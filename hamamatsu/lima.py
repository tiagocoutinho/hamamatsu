# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2019 Tiago Coutinho & Vicente Rey Bakaikoa
# Distributed under the MIT. See LICENSE for more info.

from Lima.Core import HwInterface
from Lima.Core import HwDetInfoCtrlObj, HwSyncCtrlObj, HwBufferCtrlObj
from Lima.Core import HwCap, Size, FrameDim
from Lima.Core import Bpp8, IntTrig, IntTrigMult


class Buffer(HwBufferCtrlObj):

    frame_dim = FrameDim()
    nb_buffers = 1
    nb_concat_frames = 1

    def setFrameDim(self, dim):
        self.frame_dim = dim

    def getFrameDim(self):
        return self.frame_dim

    def getNbBuffers(self):
        return self.nb_buffers

    def setNbBuffers(self, nb_buffers):
        self.nb_buffers = nb_buffers

    def getNbConcatFrames(self):
        return self.nb_concat_frames

    def setNbConcatFrames(self, nb_concat_frames):
        self.nb_concat_frames = nb_concat_frames

    def getMaxNbBuffers(self):
        return 10

    def registerFrameCallback(self, cb):
        pass

    def unregisterFrameCallback(self, cb):
        pass


class Sync(HwSyncCtrlObj):

    trig_mode = IntTrig

    def checkTrigMode(self, trig_mode):
        return trig_mode in (IntTrig, IntTrigMult)

    def setTrigMode(self, trig_mode):
        if not self.checkTrigMode(trig_mode):
            raise ValueError('Unsupported trigger mode')
        self.trig_mode = trig_mode

    def getTrigMode(self):
        return self.trig_mode

    def setExpTime(self, exp_time):
        pass

    def getExpTime(self):
        return 1.0

    def setLatTime(self, lat_time):
        pass

    def getLatTime(self):
        return 0.0

    def setNbHwFrames(self, nb_frames):
        pass

    def getNbHwFrames(self):
        return 1

    def getValidRanges(self):
        return self.ValidRangesType(1e-8, 1e6, 1e-8, 1e6)


class DetInfo(HwDetInfoCtrlObj):

    image_type = Bpp8

    def getMaxImageSize(self):
        return Size(1024, 768)

    def getDetectorImageSize(self):
        return Size(1024, 768)

    def getDefImageType(self):
        return type(self).image_type

    def getCurrImageType(self):
        return self.image_type

    def setCurrImageType(image_type):
        self.image_type = image_type

    def getPixelSize(self):
        return 1.0, 1.0

    def getDetectorType(self):
        return "Hamamatsu StreakCamera"

    def getDetectorModel():
        return "C19002"

    def registerMaxImageSizeCallback(self, cb):
        pass

    def unregisterMaxImageSizeCallback(self, cb):
        pass


class Interface(HwInterface):

    def __init__(self):
        super(Interface, self).__init__()
        self.det_info = DetInfo()
        self.sync = Sync()
        self.buff = Buffer()
        self.caps = map(HwCap, (self.det_info, self.sync, self.buff))

    def getCapList(self):
        return self.caps

    def reset(self, reset_level):
        pass

    def prepareAcq(self):
        pass

    def startAcq(self):
        pass

    def stopAcq(self):
        pass

    def getStatus(self):
        s = self.StatusType()
        s.set(self.StatusType.Ready)
        return s

    def getNbHwAcquiredFrames(self):
        return 1


def main(args=None):
    from Lima.Core import CtControl
    hw_iface = Interface()
    ctrl = CtControl(hw_iface)
    acq = ctrl.acquisition()

    return hw_iface, ctrl, acq

if __name__ == '__main__':
    iface, ctrl, acq = main()
