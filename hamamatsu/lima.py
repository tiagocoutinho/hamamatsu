import time
import logging
import threading

import numpy

from Lima.Core import (
    HwInterface, HwDetInfoCtrlObj, HwSyncCtrlObj, HwBufferCtrlObj, HwCap,
    HwFrameInfoType, SoftBufferCtrlObj, Size, FrameDim, Bpp8, Bpp12, Bpp16,
    RGB24, Timestamp, AcqReady, AcqRunning, CtControl, CtSaving,
    IntTrig, ExtTrigSingle, ExtTrigMult, ExtGate)

from hamamatsu.dcam import dcam, Stream, DCAM_PIXELTYPE, DCAMIDSTR


Status = HwInterface.StatusType


class Sync(HwSyncCtrlObj):

    trig_mode = IntTrig

    def __init__(self, detector):
        self.detector = detector
        super().__init__()

    def checkTrigMode(self, trig_mode):
        return trig_mode in {IntTrig, ExtTrigSingle, ExtTrigMult, ExtGate}

    def setTrigMode(self, trig_mode):
        if not self.checkTrigMode(trig_mode):
            raise ValueError('Unsupported trigger mode')
        if trig_mode == IntTrig:
            self.detector.triggermode = False
            self.detector.continuoustrigger = False
            self.detector.gatemode = False
        elif trig_mode == ExtTrigSingle:
            self.detector.triggermode = True
            self.detector.continuoustrigger = False
            self.detector.gatemode = False
        elif trig_mode == ExtTrigMult:
            self.detector.triggermode = True
            self.detector.continuoustrigger = True
            self.detector.gatemode = False
        elif trig_mode == ExtGate:
            self.detector.gatemode = True

    def getTrigMode(self):
        if self.detector.gatemode:
            return ExtGate
        elif self.detector.triggermode:
            return ExtTrigMult if self.detector.continuoustrigger else ExtTrigSingle
        return IntTrig

    def setExpTime(self, exp_time):
        self.detector.inttime = exp_time

    def getExpTime(self):
        return self.detector.inttime

    def setLatTime(self, lat_time):
        self.detector.set_delay_frame(lat_time)
        self.latency_time = lat_time

    def getLatTime(self):
        return self.latency_time

    def setNbHwFrames(self, nb_frames):
        self.detector.frames = nb_frames

    def getNbHwFrames(self):
        return self.detector.frames

    def getValidRanges(self):
        return self.ValidRangesType(10E-9, 1E6, 10E-9, 1E6)


class DetInfo(HwDetInfoCtrlObj):

    image_type = Bpp16
    ImageTypeMap = {
        DCAM_PIXELTYPE.MONO8: Bpp8,
        DCAM_PIXELTYPE.MONO12: Bpp12,
        DCAM_PIXELTYPE.MONO16: Bpp16,
        DCAM_PIXELTYPE.RGB24: RGB24,
        DCAM_PIXELTYPE.BGR24: BGR24,
    }
    PixelTypeMap = {v:k for v, k in ImageTypeMap.items()}

    def __init__(self, detector):
        self.detector = detector
        super().__init__()

    def getMaxImageSize(self):
        raise NotImplementedError
        return Size(w, h)

    def getDetectorImageSize(self):
        raise NotImplementedError
        return Size(w, h)

    def getDefImageType(self):
        return Bpp16

    def getCurrImageType(self):
        raise NotImplementedError

    def setCurrImageType(self, image_type):
        if image_type not in self.PixelTypeMap:
            raise ValueError(f"Unsupported image type {image_type!r}")
        itype = PixelTypeMap[image_type]
        # TODO: Activate image type
        raise NotImplementedError

    def getPixelSize(self):
        raise NotImplementedError
        # in micrometer
        width = self.detector.module_sensor_widths.mean()
        # is height 4/8 mm or the sensor thickness ?
        height = self.detector.module_sensor_thicknesses.mean()
        return width, height

    def getDetectorType(self):
        return self.detector.get_info()[DCAMIDSTR.VENDOR]

    def getDetectorModel(self):
        return self.detector.get_info()[DCAMIDSTR.MODEL]

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

    def __init__(self, detector, buffer_manager, nb_frames, frame_dim):
        self.detector = detector
        self.buffer_manager = buffer_manager
        self.nb_frames= nb_frames
        self.frame_dim = frame_dim
        self.nb_acquired_frames = 0
        self.status = Status.Ready
        self.stopped = False
        self.prepared = threading.Event()
        self.start_trigger = threading.Event()
        self.acq_thread = threading.Thread(target=self.acquire)
        self.acq_thread.daemon = True
        self.acq_thread.start()

    def start(self):
        self.start_trigger.set()

    def stop(self):
        self.stopped = True
        self.detector.stop()
        self.start_trigger.set()
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
            self.prepared.set()
            self.start_trigger.wait()
            if self.stopped:
                self.status = Status.Ready
                return
            start_time = time.time()
            self.detector.start(nb_frames)
            self.status = Status.Exposure
            buffer_manager.setStartTimestamp(Timestamp(start_time))
            for frame, buff, frame_info in zip(stream, buffers, frame_infos):
                if self.stopped:
                    self.status = Status.Ready
                    return
                self.status = Status.Readout
                copy_frame(frame, buff)
                buffer_manager.newFrameReady(frame_info)
                self.nb_acquired_frames += 1
                self.status = Status.Exposure
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
        self.acq = Acquisition(self.detector, buffer_manager, nb_frames, frame_dim)

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
    camera = dcam[camera_id]
    interface = Interface(camera)
    return interface


def get_control(url):
    interface = get_interface(url)
    return CtControl(interface)
