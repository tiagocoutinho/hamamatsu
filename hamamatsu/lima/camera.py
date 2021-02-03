# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2021 Tiago Coutinho
# Distributed under the GPLv3 license. See LICENSE for more info.

import time
import logging
import threading

import numpy

from Lima.Core import (
    HwInterface,
    HwDetInfoCtrlObj,
    HwSyncCtrlObj,
    HwBufferCtrlObj,
    HwCap,
    HwFrameInfoType,
    SoftBufferCtrlObj,
    Size,
    FrameDim,
    Bpp8,
    Bpp12,
    Bpp16,
    RGB24,
    BGR24,
    Timestamp,
    AcqReady,
    AcqRunning,
    CtControl,
    CtSaving,
    IntTrig,
    IntTrigMult,
    ExtTrigSingle,
    ExtTrigMult,
    ExtGate,
)

from hamamatsu.dcam import (
    dcam,
    Stream,
    ETriggerSource,
    EImagePixelType,
    EIDString,
    copy_frame,
)


Status = HwInterface.StatusType


class Sync(HwSyncCtrlObj):
    def __init__(self, detector):
        self.detector = detector
        self.nb_frames = 1
        srcs = detector["trigger_source"].enum_values
        self.trigger_modes = set()
        if ETriggerSource.INTERNAL in srcs:
            self.trigger_modes.add(IntTrig)
        if ETriggerSource.SOFTWARE in srcs:
            self.trigger_modes.add(IntTrigMult)
        if ETriggerSource.EXTERNAL in srcs:
            self.trigger_modes.add(ExtTrigSingle)
            self.trigger_modes.add(ExtTrigMult)
        super().__init__()

    def checkTrigMode(self, trigger_mode):
        return trigger_mode in self.trigger_modes

    def setTrigMode(self, trigger_mode):
        if not self.checkTrigMode(trigger_mode):
            raise ValueError("Unsupported trigger mode")
        if trigger_mode == IntTrig:
            self.detector["trigger_source"] = ETriggerSource.INTERNAL
        elif trigger_mode == IntTrigMult:
            self.detector["trigger_source"] = ETriggerSource.SOFTWARE
        elif trigger_mode == ExtTrigSingle:
            raise NotImplementedError
        elif trigger_mode == ExtTrigMult:
            raise NotImplementedError
        elif trigger_mode == ExtGate:
            raise NotImplementedError

    def getTrigMode(self):
        trigger_source = self.detector["trigger_source"].value
        if trigger_source == ETriggerSource.INTERNAL:
            return IntTrig
        elif trigger_source == ETriggerSource.SOFTWARE:
            return IntTrigMult
        elif trigger_source == ETriggerSource.EXTERNAL:
            raise NotImplementedError

    def setExpTime(self, exp_time):
        self.detector["exposure_time"] = exp_time

    def getExpTime(self):
        return self.detector["exposure_time"].value

    def setLatTime(self, lat_time):
        pass

    def getLatTime(self):
        return 0

    def setNbHwFrames(self, nb_frames):
        self.nb_frames = nb_frames

    def getNbHwFrames(self):
        return self.nb_frames

    def getValidRanges(self):
        return self.ValidRangesType(10e-9, 1e6, 10e-9, 1e6)


class DetInfo(HwDetInfoCtrlObj):

    image_type = Bpp16
    ImageTypeMap = {
        EImagePixelType.MONO8: Bpp8,
        EImagePixelType.MONO12: Bpp12,
        EImagePixelType.MONO12P: Bpp12,
        EImagePixelType.MONO16: Bpp16,
        EImagePixelType.RGB24: RGB24,
        EImagePixelType.BGR24: BGR24,
    }
    PixelTypeMap = {v: k for v, k in ImageTypeMap.items()}

    def __init__(self, detector):
        self.detector = detector
        super().__init__()

    def getMaxImageSize(self):
        w = self.detector["image_width"].max_value
        h = self.detector["image_height"].max_value
        return Size(w, h)

    def getDetectorImageSize(self):
        w = self.detector["image_width"].value
        h = self.detector["image_height"].value
        return Size(w, h)

    def getDefImageType(self):
        return Bpp16

    def getCurrImageType(self):
        pixel_type = self.detector["image_pixel_type"].value
        return self.ImageTypeMap[pixel_type]

    def setCurrImageType(self, image_type):
        if image_type not in self.PixelTypeMap:
            raise ValueError(f"Unsupported image type {image_type!r}")
        pixel_type = self.PixelTypeMap[image_type]
        self.detector["image_pixel_type"] = pixel_type

    def getPixelSize(self):
        """Pixel size (x, y). Units in meter"""
        return self.detector.pixel_size

    def getDetectorType(self):
        return self.detector.info[EIDString.VENDOR]

    def getDetectorModel(self):
        return self.detector.info[EIDString.MODEL]

    def registerMaxImageSizeCallback(self, cb):
        pass

    def unregisterMaxImageSizeCallback(self, cb):
        pass


def gen_buffer(buffer_manager, nb_frames, frame_size):
    for frame_nb in range(nb_frames):
        buff = buffer_manager.getFrameBufferPtr(frame_nb)
        # don't know why the sip.voidptr has no size
        buff.setsize(frame_size)
        yield numpy.frombuffer(buff, dtype=numpy.byte)


class Acquisition:
    def __init__(self, detector, buffer_manager, nb_frames, frame_dim, trigger_mode):
        self.detector = detector
        self.buffer_manager = buffer_manager
        self.nb_frames = nb_frames
        self.trigger_mode = trigger_mode
        self.nb_frames_triggered = 0
        self.frame_dim = frame_dim
        self.nb_acquired_frames = 0
        self.status = Status.Ready
        self.stopped = False
        self.prepared = threading.Event()
        self.start_event = threading.Event()
        self.acq_thread = threading.Thread(target=self.acquire)
        self.acq_thread.daemon = True
        self.acq_thread.start()

    def wait_until_prepared(self):
        self.prepared.wait()

    def start(self):
        if self.start_event.is_set():
            self.nb_frames_triggered += 1
            self.detector.fire_software_trigger()
        else:
            self.start_event.set()

    def stop(self):
        self.stopped = True
        self.detector.stop()
        self.start_event.set()
        self.acq_thread.join()

    def acquire(self):
        buffer_manager = self.buffer_manager
        detector = self.detector
        nb_frames = self.nb_frames
        frame_dim = self.frame_dim
        buffers = gen_buffer(buffer_manager, nb_frames, frame_dim.getMemSize())
        frame_infos = []
        for frame_nb in range(nb_frames):
            frame_info = HwFrameInfoType()
            frame_info.acq_frame_nb = frame_nb
            frame_infos.append(frame_info)

        with Stream(detector, nb_frames) as stream:
            # From now we are fully prepared.
            # Notify of that and wait for trigger to start
            self.prepared.set()
            self.start_event.wait()
            if self.stopped:
                self.status = Status.Ready
                return
            start_time = time.time()
            self.detector.start()
            buffer_manager.setStartTimestamp(Timestamp(start_time))
            if self.trigger_mode != IntTrigMult:
                self.status = Status.Exposure
            for frame, buff, frame_info in zip(stream, buffers, frame_infos):
                print("new frame arrived")
                if self.stopped:
                    self.status = Status.Ready
                    return
                self.status = Status.Readout
                copy_frame(frame, buff)
                buffer_manager.newFrameReady(frame_info)
                self.nb_acquired_frames += 1
                if self.trigger_mode == IntTrigMult:
                    self.status = Status.Ready
                else:
                    Status.Exposure
            self.status = Status.Ready


class Interface(HwInterface):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.det_info = DetInfo(detector)
        self.sync = Sync(detector)
        self.buff = SoftBufferCtrlObj()
        self.caps = list(map(HwCap, (self.det_info, self.sync, self.buff)))
        self.acq = None

    def getCapList(self):
        return self.caps

    def reset(self, reset_level):
        pass

    def prepareAcq(self):
        nb_frames = self.sync.getNbHwFrames()
        frame_dim = self.buff.getFrameDim()
        buffer_manager = self.buff.getBuffer()
        trigger_mode = self.sync.getTrigMode()
        self.acq = Acquisition(
            self.detector, buffer_manager, nb_frames, frame_dim, trigger_mode
        )
        self.acq.wait_until_prepared()

    def startAcq(self):
        self.acq.start()

    def stopAcq(self):
        if self.acq:
            self.acq.stop()

    def getStatus(self):
        s = Status()
        s.set(self.acq.status if self.acq else Status.Ready)
        return s

    def getNbHwAcquiredFrames(self):
        return self.acq.nb_acquired_frames if self.acq else 0


def get_interface(camera_id):
    dcam.open()
    camera = dcam[camera_id] if isinstance(camera_id, int) else camera_id
    camera.open()
    interface = Interface(camera)
    return interface


def get_control(camera_id):
    interface = get_interface(camera_id)
    return CtControl(interface)
