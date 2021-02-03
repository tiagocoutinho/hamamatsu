# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2021 Tiago Coutinho
# Distributed under the GPLv3 license. See LICENSE for more info.

import enum
import ctypes
import logging
import weakref
import functools
import contextlib

import numpy

TIMEOUT_INFINITE = 0x80000000


class EError(enum.IntEnum):

    # status error
    BUSY = 0x80000101  # API cannot process in busy state.
    NOTREADY = 0x80000103  # API requires ready state.
    NOTSTABLE = 0x80000104  # API requires stable or unstable state.
    UNSTABLE = 0x80000105  # API does not support in unstable state.
    NOTBUSY = 0x80000107  # API requires busy state.

    EXCLUDED = 0x80000110  # some resource is exclusive and already used

    COOLINGTROUBLE = 0x80000302  # something happens near cooler
    NOTRIGGER = (
        0x80000303  # no trigger when necessary. Some camera supports this error.
    )
    TEMPERATURE_TROUBLE = 0x80000304  # camera warns its temperature
    TOOFREQUENTTRIGGER = (
        0x80000305  # input too frequent trigger. Some camera supports this error.
    )

    # wait error
    ABORT = 0x80000102  # abort process
    TIMEOUT = 0x80000106  # timeout
    LOSTFRAME = 0x80000301  # frame data is lost
    MISSINGFRAME_TROUBLE = (
        0x80000F06  # frame is lost but reason is low lever driver's bug
    )
    INVALIDIMAGE = 0x80000321  # hpk format data is invalid data

    # initialization error
    NORESOURCE = 0x80000201  # not enough resource except memory
    NOMEMORY = 0x80000203  # not enough memory
    NOMODULE = 0x80000204  # no sub module
    NODRIVER = 0x80000205  # no driver
    NOCAMERA = 0x80000206  # no camera
    NOGRABBER = 0x80000207  # no grabber
    NOCOMBINATION = 0x80000208  # no combination on registry

    FAILOPEN = 0x80001001  # DEPRECATED
    INVALIDMODULE = 0x80000211  # dcam_init() found invalid module
    INVALIDCOMMPORT = 0x80000212  # invalid serial port
    FAILOPENBUS = 0x81001001  # the bus or driver are not available
    FAILOPENCAMERA = 0x82001001  # camera report error during opening
    FRAMEGRABBER_NEEDS_FIRMWAREUPDATE = (
        0x80001002  # need to update frame grabber firmware to use the camera
    )

    # calling error
    INVALIDCAMERA = 0x80000806  # invalid camera
    INVALIDHANDLE = 0x80000807  # invalid camera handle
    INVALIDPARAM = 0x80000808  # invalid parameter
    INVALIDVALUE = 0x80000821  # invalid property value
    OUTOFRANGE = 0x80000822  # value is out of range
    NOTWRITABLE = 0x80000823  # the property is not writable
    NOTREADABLE = 0x80000824  # the property is not readable
    INVALIDPROPERTYID = 0x80000825  # the property id is invalid
    NEWAPIREQUIRED = 0x80000826  # old API cannot present the value because only new API need to be used
    WRONGHANDSHAKE = (
        0x80000827  # this error happens DCAM get error code from camera unexpectedly
    )
    NOPROPERTY = (
        0x80000828  # there is no altenative or influence id, or no more property id
    )
    INVALIDCHANNEL = (
        0x80000829  # the property id specifies channel but channel is invalid
    )
    INVALIDVIEW = 0x8000082A  # the property id specifies channel but channel is invalid
    INVALIDSUBARRAY = 0x8000082B  # the combination of subarray values are invalid. e.g. SUBARRAYHPOS + SUBARRAYHSIZE is greater than the number of horizontal pixel of sensor.
    ACCESSDENY = 0x8000082C  # the property cannot access during this DCAM STATUS
    NOVALUETEXT = 0x8000082D  # the property does not have value text
    WRONGPROPERTYVALUE = 0x8000082E  # at least one property value is wrong
    DISHARMONY = 0x80000830  # the paired camera does not have same parameter
    FRAMEBUNDLESHOULDBEOFF = (
        0x80000832  # framebundle mode should be OFF under current property settings
    )
    INVALIDFRAMEINDEX = 0x80000833  # the frame index is invalid
    INVALIDSESSIONINDEX = 0x80000834  # the session index is invalid
    NOCORRECTIONDATA = 0x80000838  # not take the dark and shading correction data yet.
    CHANNELDEPENDENTVALUE = 0x80000839  # each channel has own property value so can't return overall property value.
    VIEWDEPENDENTVALUE = 0x8000083A  # each view has own property value so can't return overall property value.
    INVALIDCALIBSETTING = 0x8000083E  # the setting of properties are invalid on sampling calibration data. some camera has the limitation to make calibration data. e.g. the trigger source is INTERNAL only and read out direction isn't trigger.
    LESSSYSTEMMEMORY = 0x8000083F  # the sysmte memory size is too small. PC doesn't have enough memory or is limited memory by 32bit OS.
    NOTSUPPORT = 0x80000F03  # camera does not support the function or property with current settings

    # camera or bus trouble
    FAILREADCAMERA = 0x83001002  # failed to read data from camera
    FAILWRITECAMERA = 0x83001003  # failed to write data to the camera
    CONFLICTCOMMPORT = 0x83001004  # conflict the com port name user set
    OPTICS_UNPLUGGED = 0x83001005  # Optics part is unplugged so please check it.
    FAILCALIBRATION = 0x83001006  # fail calibration

    # 0x84000100 - 0x840001FF, INVALIDMEMBER_x
    INVALIDMEMBER_3 = 0x84000103  # 3th member variable is invalid value
    INVALIDMEMBER_5 = 0x84000105  # 5th member variable is invalid value
    INVALIDMEMBER_7 = 0x84000107  # 7th member variable is invalid value
    INVALIDMEMBER_8 = 0x84000108  # 7th member variable is invalid value
    INVALIDMEMBER_9 = 0x84000109  # 9th member variable is invalid value
    FAILEDOPENRECFILE = 0x84001001  # DCAMREC failed to open the file
    INVALIDRECHANDLE = 0x84001002  # DCAMREC is invalid handle
    FAILEDWRITEDATA = 0x84001003  # DCAMREC failed to write the data
    FAILEDREADDATA = 0x84001004  # DCAMREC failed to read the data
    NOWRECORDING = 0x84001005  # DCAMREC is recording data now
    WRITEFULL = 0x84001006  # DCAMREC writes full frame of the session
    ALREADYOCCUPIED = 0x84001007  # DCAMREC handle is already occupied by other HDCAM
    TOOLARGEUSERDATASIZE = (
        0x84001008  # DCAMREC is set the large value to user data size
    )
    NOIMAGE = 0x84001804  # not stored image in buffer on bufrecord
    INVALIDWAITHANDLE = 0x84002001  # DCAMWAIT is invalid handle
    NEWRUNTIMEREQUIRED = 0x84002002  # DCAM Module Version is older than the version that the camera requests
    VERSIONMISMATCH = (
        0x84002003  # Camre returns the error on setting parameter to limit version
    )
    RUNAS_FACTORYMODE = 0x84002004  # Camera is running as a factory mode
    IMAGE_UNKNOWNSIGNATURE = (
        0x84003001  # sigunature of image header is unknown or corrupted
    )
    IMAGE_NEWRUNTIMEREQUIRED = 0x84003002  # version of image header is newer than version that used DCAM supports
    IMAGE_ERRORSTATUSEXIST = 0x84003003  # image header stands error status
    IMAGE_HEADERCORRUPTED = 0x84004004  # image header value is strange
    IMAGE_BROKENCONTENT = 0x84004005  # image content is corrupted

    # calling error for DCAM-API 2.1.3
    UNKNOWNMSGID = 0x80000801  # unknown message id
    UNKNOWNSTRID = 0x80000802  # unknown string id
    UNKNOWNPARAMID = 0x80000803  # unkown parameter id
    UNKNOWNBITSTYPE = 0x80000804  # unknown bitmap bits type
    UNKNOWNDATATYPE = 0x80000805  # unknown frame data type

    # internal error
    NONE = 0  # no error, nothing to have done
    INSTALLATIONINPROGRESS = 0x80000F00  # installation progress
    UNREACH = 0x80000F01  # internal error
    UNLOADED = 0x80000F04  # calling after process terminated
    THRUADAPTER = 0x80000F05  #
    NOCONNECTION = 0x80000F07  # HDCAM lost connection to camera

    NOTIMPLEMENT = 0x80000F02  # not yet implementation

    APIINIT_INITOPTIONBYTES = 0xA4010003  # DCAMAPI_INIT::initoptionbytes is invalid
    APIINIT_INITOPTION = 0xA4010004  # DCAMAPI_INIT::initoption is invalid

    INITOPTION_COLLISION_BASE = 0xA401C000
    INITOPTION_COLLISION_MAX = 0xA401FFFF

    SUCCESS = 1


class EPropOption(enum.IntEnum):

    # direction flag for dcam_getnextpropertyid(), dcam_querypropertyvalue()
    PRIOR = 0xFF000000  #  prior value
    NEXT = 0x01000000  #  next value or id

    # direction flag for dcam_querypropertyvalue()
    NEAREST = 0x80000000  #  nearest value        #  reserved

    # option for dcam_getnextpropertyid()
    SUPPORT = 0x00000000  #  default option
    UPDATED = 0x00000001  #  UPDATED and VOLATILE can be used at same time
    VOLATILE = 0x00000002  #  UPDATED and VOLATILE can be used at same time
    ARRAYELEMENT = 0x00000004  #  ARRAYELEMENT

    # ** for all option parameter **
    NONE = 0x00000000  #  no option


class EIDString(enum.IntEnum):
    BUS = 0x04000101
    CAMERAID = 0x04000102
    VENDOR = 0x04000103
    MODEL = 0x04000104
    CAMERAVERSION = 0x04000105
    DRIVERVERSION = 0x04000106
    MODULEVERSION = 0x04000107
    DCAMAPIVERSION = 0x04000108

    CAMERA_SERIESNAME = 0x0400012C

    OPTICALBLOCK_MODEL = 0x04001101
    OPTICALBLOCK_ID = 0x04001102
    OPTICALBLOCK_DESCRIPTION = 0x04001103
    OPTICALBLOCK_CHANNEL_1 = 0x04001104
    OPTICALBLOCK_CHANNEL_2 = 0x04001105


class EStatus(enum.IntEnum):
    ERROR = 0x0000
    BUSY = 0x0001
    READY = 0x0002
    STABLE = 0x0003
    UNSTABLE = 0x0004


class EAttach(enum.IntEnum):
    FRAME = 0
    TIMESTAMP = 1
    FRAMESTAMP = 2
    PRIMARY_TIMESTAMP = 3
    PRIMARY_FRAMESTAMP = 4


class ETransfer(enum.IntEnum):
    FRAME = 0


class EWaitEvent(enum.IntFlag):
    CAP_TRANSFERRED = 0x0001
    CAP_FRAMEREADY = 0x0002  # all modules support
    CAP_CYCLEEND = 0x0004  # all modules support
    CAP_EXPOSUREEND = 0x0008
    CAP_STOPPED = 0x0010

    REC_STOPPED = 0x0100
    REC_WARNING = 0x0200
    REC_MISSED = 0x0400
    REC_DISKFULL = 0x1000
    REC_WRITEFAULT = 0x2000
    REC_SKIPPED = 0x4000
    REC_WRITEFRAME = 0x8000  # DCAMCAP_START_BUFRECORD only


class EImagePixelType(enum.IntEnum):
    MONO8 = 0x00000001
    MONO16 = 0x00000002
    MONO12 = 0x00000003
    MONO12P = 0x00000005

    RGB24 = 0x00000021
    RGB48 = 0x00000022
    BGR24 = 0x00000029
    BGR48 = 0x0000002A

    NONE = 0x00000000

    def bytes_per_pixel(self):
        if self is self.MONO8:
            return 1
        elif self is self.MONO16:
            return 2
        elif self in {self.MONO12, self.MONO12P}:
            return 1.5
        elif self in {self.RGB24, self.BGR24}:
            return 3
        elif self in {self.RGB48, self.BGR48}:
            return 6
        elif self is self.NONE:
            return 0

    def dtype(self):
        if self is self.MONO8:
            return numpy.uint8
        elif self is self.MONO16:
            return numpy.uint16
        elif self is self.NONE:
            return 0
        else:
            return None


EBufferPixelType = EImagePixelType


class EStart(enum.IntEnum):
    SEQUENCE = -1
    SNAP = 0


class EPropAttr(enum.IntFlag):
    # supporting information of DCAM_PROPERTYATTR
    HASRANGE = 0x80000000
    HASSTEP = 0x40000000
    HASDEFAULT = 0x20000000
    HASVALUETEXT = 0x10000000

    # property id information
    HASCHANNEL = 0x08000000  # value can set the value for each channels

    # property attribute
    AUTOROUNDING = 0x00800000
    # The dcam_setproperty() or dcam_setgetproperty() will failure if this bit exists.
    # If this flag does not exist the value will be round up when it is not supported.
    STEPPING_INCONSISTENT = 0x00400000
    # The valuestep of DCAM_PROPERTYATTR is not consistent across the entire range of
    # values.
    DATASTREAM = 0x00200000  # value is releated to image attribute

    HASRATIO = 0x00100000  # value has ratio control capability

    VOLATILE = 0x00080000  # value may be changed by user or automatically

    WRITABLE = 0x00020000  # value can be set when state is manual
    READABLE = 0x00010000  # value is readable when state is manual

    HASVIEW = 0x00008000  # value can set the value for each views
    _SYSTEM = 0x00004000  # system id                                    # reserved

    ACCESSREADY = 0x00002000  # This value can get or set at READY status
    ACCESSBUSY = 0x00001000  # This value can get or set at BUSY status

    ADVANCED = 0x00000800  # User has to take care to change this value # reserved
    ACTION = 0x00000400  # writing value takes related effect            # reserved
    EFFECTIVE = 0x00000200  # value is effective                            # reserved

    # property value type
    TYPE_NONE = 0x00000000  # undefined
    TYPE_MODE = 0x00000001  # 01:    mode, 32bit integer in case of 32bit OS
    TYPE_LONG = 0x00000002  # 02:    32bit integer in case of 32bit OS
    TYPE_REAL = 0x00000003  # 03:    64bit float
    #      no 32bit float

    # application has to use double-float type variable even the property is not REAL.

    TYPE_MASK = 0x0000000F  # mask for property value type


class EUnit(enum.IntEnum):
    SECOND = 1  # sec
    CELSIUS = 2  # for sensor temperature
    KELVIN = 3  # for color temperature
    METERPERSECOND = 4  # for LINESPEED
    PERSECOND = 5  # for FRAMERATE and LINERATE
    DEGREE = 6  # for OUTPUT ROTATION
    MICROMETER = 7  # for length
    NONE = 0  # no unit

    def to_SI(self, value):
        if self in {
            self.SECOND,
            self.KELVIN,
            self.METERPERSECOND,
            self.NONE,
            self.PERSECOND,
        }:
            return value
        elif self == self.CELSIUS:
            return value + 273.15
        elif self == self.MICROMETER:
            return value * 1e-6
        elif self == self.DEGREE:
            return numpy.radians(value)
        return value


class ESensorMode(enum.IntEnum):
    AREA = 1
    SLIT = 2
    LINE = 3
    TDI = 4
    FRAMING = 5
    PARTIALAREA = 6
    SLITLINE = 9
    TDI_EXTENDED = 10
    PANORAMIC = 11
    PROGRESSIVE = 12
    SPLITVIEW = 14
    DUALLIGHTSHEET = 16


class ESystemAlive(enum.IntEnum):
    OFFLINE = 1
    ONLINE = 2


class EColorType(enum.IntEnum):
    BW = 0x00000001
    RGB = 0x00000002
    BGR = 0x00000003


class EShutterMode(enum.IntEnum):
    GLOBAL = 1
    ROLLING = 2


class EReadoutSpeed(enum.IntEnum):
    SLOWEST = 1
    FASTEST = 0x7FFFFFFF


class EReadoutDirection(enum.IntEnum):
    FORWARD = 1
    BACKWARD = 2
    BYTRIGGER = 3
    DIVERGE = 5


class ETriggerActive(enum.IntEnum):
    EDGE = 1
    LEVEL = 2
    SYNCREADOUT = 3
    POINT = 4


class ETriggerSource(enum.IntEnum):
    INTERNAL = 1
    EXTERNAL = 2
    SOFTWARE = 3
    MASTERPULSE = 4


class EReadoutUnit(enum.IntEnum):
    FRAME = 2
    BUNDLEDLINE = 3
    BUNDLEDFRAME = 4


class ECCDMode(enum.IntEnum):
    NORMALCCD = 1
    EMCCD = 2


class ECMOSMode(enum.IntEnum):
    NORMAL = 1
    NONDESTRUCTIVE = 2


class EOutputIntensity(enum.IntEnum):
    NORMAL = 1
    TESTPATTERN = 2


class EOutputDataOrientation(enum.IntEnum):
    NORMAL = 1
    MIRROR = 2
    FLIP = 3


class EOutputDataOperation(enum.IntEnum):
    RAW = 1
    ALIGNED = 2


class ETestPatternKind(enum.IntEnum):
    FLAT = 2
    IFLAT = 3
    HORZGRADATION = 4
    IHORZGRADATION = 5
    VERTGRADATION = 6
    IVERTGRADATION = 7
    LINE = 8
    ILINE = 9
    DIAGONAL = 10
    IDIAGONAL = 11
    FRAMECOUNT = 12


class EDigitalBinningMethod(enum.IntEnum):
    MINIMUM = 1
    MAXIMUM = 2
    ODD = 3
    EVEN = 4
    SUM = 5
    AVERAGE = 6

class EBusSpeed(enum.IntEnum):
    SLOWEST = 1
    FASTEST = 0x7FFFFFFF


class ETriggerMode(enum.IntEnum):
    NORMAL = 1
    PIV = 3
    START = 6
    MULTIGATE = 7
    MULTIFRAME = 8


class ETriggerPolarity(enum.IntEnum):
    NEGATIVE = 1
    POSITIVE = 2


class ETriggerConnector(enum.IntEnum):
    INTERFACE = 1
    BNC = 2
    MULTI = 3


class EInternalTriggerHandling(enum.IntEnum):
    SHORTEREXPOSURETIME = 1
    FASTERFRAMERATE = 2
    ABANDONWRONGFRAME = 3
    BURSTMODE = 4
    INDIVIDUALEXPOSURE = 7


class ESyncReadoutSystemBlank(enum.IntEnum):
    STANDARD = 1
    MINIMUM = 2


class ETriggerEnableActive(enum.IntEnum):
    DENY = 1
    ALWAYS = 2
    LEVEL = 3
    START = 4


class ETriggerEnablePolarity(enum.IntEnum):
    NEGATIVE = 1
    POSITIVE = 2
    INTERLOCK = 3


class EOutputTriggerChannelSync(enum.IntEnum):
    ONE_CHANNEL = 1
    TWO_CHANNELS = 2
    THREE_CHANNELS = 3


class EOutputTriggerProgramableStart(enum.IntEnum):
    FIRSTEXPOSURE = 1
    FIRSTREADOUT = 2


class EOutputTriggerSource(enum.IntEnum):
    EXPOSURE = 1
    READOUTEND = 2
    VSYNC = 3
    HSYNC = 4
    TRIGGER = 6


class EOutputTriggerPolarity(enum.IntEnum):
    NEGATIVE = 1
    POSITIVE = 2


class EOutputTriggerActive(enum.IntEnum):
    EDGE = 1
    LEVEL = 2


class EOutputTriggerKind(enum.IntEnum):
    LOW = 1
    EXPOSURE = 2
    PROGRAMABLE = 3
    TRIGGERREADY = 4
    HIGH = 5


class EBinning(enum.IntEnum):
    ONExONE = 1
    TWOxTWO = 2
    FOURxFOUR = 4
    EIGHTxEIGHT = 8
    SIXTEENxSIXTEEN = 16


class ESubArrayMode(enum.IntEnum):
    OFF = 1
    ON = 2


class ETimmingExposure(enum.IntEnum):
    AFTERREADOUT = 1
    OVERLAPREADOUT = 2
    ROLLING = 3
    ALWAYS = 4
    TDI = 5


class EFrameBundleMode(enum.IntEnum):
    OFF = 1
    ON = 2


class EExposureTimeControl(enum.IntEnum):
    OFF = 1
    NORMAL = 2


class EDirectGainMode(enum.IntEnum):
    OFF = 1
    ON = 2


class EDefectCorrectMode(enum.IntEnum):
    OFF = 1
    ON = 2


class ETimeStampProducer(enum.IntEnum):
    NONE = 1
    DCAMMODULE = 2
    KERNELDRIVER = 3
    CAPTUREDEVICE = 4
    IMAGINGDEVICE = 5


class EFrameStampProducer(enum.IntEnum):
    NONE = 1
    DCAMMODULE = 2
    KERNELDRIVER = 3
    CAPTUREDEVICE = 4
    IMAGINGDEVICE = 5


class EProp(enum.IntEnum):
    # Group: TIMING
    TRIGGERSOURCE = 0x00100110  # R/W, mode,    "TRIGGER SOURCE"
    TRIGGERACTIVE = 0x00100120  # R/W, mode,    "TRIGGER ACTIVE"
    TRIGGER_MODE = 0x00100210  # R/W, mode,    "TRIGGER MODE"
    TRIGGERPOLARITY = 0x00100220  # R/W, mode,    "TRIGGER POLARITY"
    TRIGGER_CONNECTOR = 0x00100230  # R/W, mode,    "TRIGGER CONNECTOR"
    TRIGGERTIMES = 0x00100240  # R/W, long,    "TRIGGER TIMES"
    #      0x00100250 is reserved
    TRIGGERDELAY = 0x00100260  # R/W, sec,    "TRIGGER DELAY"
    INTERNALTRIGGER_HANDLING = 0x00100270  # R/W, mode,    "INTERNAL TRIGGER HANDLING"
    TRIGGERMULTIFRAME_COUNT = 0x00100280  # R/W, long,    "TRIGGER MULTI FRAME COUNT"
    SYNCREADOUT_SYSTEMBLANK = 0x00100290  # R/W, mode,    "SYNC READOUT SYSTEM BLANK"

    TRIGGERENABLE_ACTIVE = 0x00100410  # R/W, mode,    "TRIGGER ENABLE ACTIVE"
    TRIGGERENABLE_POLARITY = 0x00100420  # R/W, mode,    "TRIGGER ENABLE POLARITY"

    TRIGGERNUMBER_FORFIRSTIMAGE = (
        0x00100810  # R/O, long,    "TRIGGER NUMBER FOR FIRST IMAGE"
    )
    TRIGGERNUMBER_FORNEXTIMAGE = (
        0x00100820  # R/O, long,    "TRIGGER NUMBER FOR NEXT IMAGE"
    )

    BUS_SPEED = 0x00180110  # R/W, long,    "BUS SPEED"

    NUMBEROF_OUTPUTTRIGGERCONNECTOR = (
        0x001C0010  # R/O, long,    "NUMBER OF OUTPUT TRIGGER CONNECTOR"
    )
    OUTPUTTRIGGER_CHANNELSYNC = (
        0x001C0030  # R/W, mode,    "OUTPUT TRIGGER CHANNEL SYNC"
    )
    OUTPUTTRIGGER_PROGRAMABLESTART = (
        0x001C0050  # R/W, mode,    "OUTPUT TRIGGER PROGRAMABLE START"
    )
    OUTPUTTRIGGER_SOURCE = 0x001C0110  # R/W, mode,    "OUTPUT TRIGGER SOURCE"
    OUTPUTTRIGGER_POLARITY = 0x001C0120  # R/W, mode,    "OUTPUT TRIGGER POLARITY"
    OUTPUTTRIGGER_ACTIVE = 0x001C0130  # R/W, mode,    "OUTPUT TRIGGER ACTIVE"
    OUTPUTTRIGGER_DELAY = 0x001C0140  # R/W, sec,    "OUTPUT TRIGGER DELAY"
    OUTPUTTRIGGER_PERIOD = 0x001C0150  # R/W, sec,    "OUTPUT TRIGGER PERIOD"
    OUTPUTTRIGGER_KIND = 0x001C0160  # R/W, mode,    "OUTPUT TRIGGER KIND"
    OUTPUTTRIGGER_BASESENSOR = 0x001C0170  # R/W, mode,    "OUTPUT TRIGGER BASE SENSOR"
    OUTPUTTRIGGER_PREHSYNCCOUNT = (
        0x001C0190  # R/W, mode,    "OUTPUT TRIGGER PRE HSYNC COUNT"
    )
    #                 - 0x001C10FF for 16 output trigger connector, reserved
    _OUTPUTTRIGGER = 0x00000100  # the offset of ID for Nth OUTPUT TRIGGER parameter

    MASTERPULSE_MODE = 0x001E0020  # R/W, mode,    "MASTER PULSE MODE"
    MASTERPULSE_TRIGGERSOURCE = (
        0x001E0030  # R/W, mode,    "MASTER PULSE TRIGGER SOURCE"
    )
    MASTERPULSE_INTERVAL = 0x001E0040  # R/W, sec,    "MASTER PULSE INTERVAL"
    MASTERPULSE_BURSTTIMES = 0x001E0050  # R/W, long,    "MASTER PULSE BURST TIMES"

    # Group: FEATURE
    # exposure period
    EXPOSURETIME = 0x001F0110  # R/W, sec,    "EXPOSURE TIME"
    SYNC_MULTIVIEWEXPOSURE = (
        0x001F0120  # R/W, mode,    "SYNCHRONOUS MULTI VIEW EXPOSURE"
    )
    EXPOSURETIME_CONTROL = 0x001F0130  # R/W, mode,    "EXPOSURE TIME CONTROL"
    TRIGGER_FIRSTEXPOSURE = 0x001F0200  # R/W, mode,    "TRIGGER FIRST EXPOSURE"
    TRIGGER_GLOBALEXPOSURE = 0x001F0300  # R/W, mode,    "TRIGGER GLOBAL EXPOSURE"
    FIRSTTRIGGER_BEHAVIOR = 0x001F0310  # R/W, mode,    "FIRST TRIGGER BEHAVIOR"
    MULTIFRAME_EXPOSURE = 0x001F1000  # R/W, sec,    "MULTI FRAME EXPOSURE TIME"
    #                     - 0x001F1FFF for 256 MULTI FRAME
    _MULTIFRAME = 0x00000010  # the offset of ID for Nth MULTIFRAME

    # anti-blooming
    LIGHTMODE = 0x00200110  # R/W, mode,    "LIGHT MODE"
    #      0x00200120 is reserved

    # sensitivity
    SENSITIVITYMODE = 0x00200210  # R/W, mode,    "SENSITIVITY MODE"
    SENSITIVITY = 0x00200220  # R/W, long,    "SENSITIVITY"
    SENSITIVITY2_MODE = (
        0x00200230  # R/W, mode,    "SENSITIVITY2 MODE"            # reserved
    )
    SENSITIVITY2 = 0x00200240  # R/W, long,    "SENSITIVITY2"

    DIRECTEMGAIN_MODE = 0x00200250  # R/W, mode,    "DIRECT EM GAIN MODE"
    EMGAINWARNING_STATUS = 0x00200260  # R/O, mode,    "EM GAIN WARNING STATUS"
    EMGAINWARNING_LEVEL = 0x00200270  # R/W, long,    "EM GAIN WARNING LEVEL"
    EMGAINWARNING_ALARM = 0x00200280  # R/W, mode,    "EM GAIN WARNING ALARM"
    EMGAINPROTECT_MODE = 0x00200290  # R/W, mode,    "EM GAIN PROTECT MODE"
    EMGAINPROTECT_AFTERFRAMES = (
        0x002002A0  # R/W, long,    "EM GAIN PROTECT AFTER FRAMES"
    )

    MEASURED_SENSITIVITY = 0x002002B0  # R/O, real,    "MEASURED SENSITIVITY"

    PHOTONIMAGINGMODE = 0x002002F0  # R/W, mode,    "PHOTON IMAGING MODE"

    # sensor cooler
    SENSORTEMPERATURE = 0x00200310  # R/O, celsius,"SENSOR TEMPERATURE"
    SENSORCOOLER = 0x00200320  # R/W, mode,    "SENSOR COOLER"
    SENSORTEMPERATURETARGET = 0x00200330  # R/W, celsius,"SENSOR TEMPERATURE TARGET"
    SENSORCOOLERSTATUS = 0x00200340  # R/O, mode,    "SENSOR COOLER STATUS"
    SENSORCOOLERFAN = 0x00200350  # R/W, mode,    "SENSOR COOLER FAN"
    SENSORTEMPERATURE_AVE = 0x00200360  # R/O, celsius,"SENSOR TEMPERATURE AVE"
    SENSORTEMPERATURE_MIN = 0x00200370  # R/O, celsius,"SENSOR TEMPERATURE MIN"
    SENSORTEMPERATURE_MAX = 0x00200380  # R/O, celsius,"SENSOR TEMPERATURE MAX"
    SENSORTEMPERATURE_STATUS = 0x00200390  # R/O, mode,    "SENSOR TEMPERATURE STATUS"
    SENSORTEMPERATURE_PROTECT = 0x00200400  # R/W, mode,    "SENSOR TEMPERATURE MODE"

    # mechanical shutter
    MECHANICALSHUTTER = 0x00200410  # R/W, mode,    "MECHANICAL SHUTTER"
    #    MECHANICALSHUTTER_AUTOMODE        = 0x00200420 # R/W, mode,    "MECHANICAL SHUTTER AUTOMODE"        # reserved

    # contrast enhance
    #    CONTRAST_CONTROL                = 0x00300110 # R/W, mode,    "CONTRAST CONTROL"            # reserved
    CONTRASTGAIN = 0x00300120  # R/W, long,    "CONTRAST GAIN"
    CONTRASTOFFSET = 0x00300130  # R/W, long,    "CONTRAST OFFSET"
    #      0x00300140 is reserved
    HIGHDYNAMICRANGE_MODE = 0x00300150  # R/W, mode,    "HIGH DYNAMIC RANGE MODE"
    DIRECTGAIN_MODE = 0x00300160  # R/W, mode,    "DIRECT GAIN MODE"

    REALTIMEGAINCORRECT_MODE = (
        0x00300170  # R/W,    mode,    "REALTIME GAIN CORRECT MODE"
    )
    REALTIMEGAINCORRECT_LEVEL = (
        0x00300180  # R/W,    mode,    "REALTIME GAIN CORRECT LEVEL"
    )
    REALTIMEGAINCORRECT_INTERVAL = (
        0x00300190  # R/W,    mode,    "REALTIME GAIN CORRECT INTERVAL"
    )
    NUMBEROF_REALTIMEGAINCORRECTREGION = (
        0x003001A0  # R/W,    long,    "NUMBER OF REALTIME GAIN CORRECT REGION"
    )

    # color features
    VIVIDCOLOR = 0x00300200  # R/W, mode,    "VIVID COLOR"                #[C7780]
    WHITEBALANCEMODE = 0x00300210  # R/W, mode,    "WHITEBALANCE MODE"
    WHITEBALANCETEMPERATURE = 0x00300220  # R/W, color-temp., "WHITEBALANCE TEMPERATURE"
    WHITEBALANCEUSERPRESET = 0x00300230  # R/W, long,    "WHITEBALANCE USER PRESET"
    #      0x00300310 is reserved

    REALTIMEGAINCORRECTREGION_HPOS = (
        0x00301000  # R/W,    long,    "REALTIME GAIN CORRECT REGION HPOS"
    )
    REALTIMEGAINCORRECTREGION_HSIZE = (
        0x00302000  # R/W,    long,    "REALTIME GAIN CORRECT REGION HSIZE"
    )

    _REALTIMEGAINCORRECTIONREGION = (
        0x00000010  # the offset of ID for Nth REALTIME GAIN CORRECT REGION parameter
    )

    # Group: ALU
    # ALU
    INTERFRAMEALU_ENABLE = 0x00380010  # R/W, mode,    "INTERFRAME ALU ENABLE"
    RECURSIVEFILTER = 0x00380110  # R/W, mode,    "RECURSIVE FILTER"
    RECURSIVEFILTERFRAMES = 0x00380120  # R/W, long,    "RECURSIVE FILTER FRAMES"
    SPOTNOISEREDUCER = 0x00380130  # R/W, mode,    "SPOT NOISE REDUCER"
    SUBTRACT = 0x00380210  # R/W, mode,    "SUBTRACT"
    SUBTRACTIMAGEMEMORY = 0x00380220  # R/W, mode,    "SUBTRACT IMAGE MEMORY"
    STORESUBTRACTIMAGETOMEMORY = (
        0x00380230  # W/O, mode,    "STORE SUBTRACT IMAGE TO MEMORY"
    )
    SUBTRACTOFFSET = 0x00380240  # R/W, long    "SUBTRACT OFFSET"
    DARKCALIB_STABLEMAXINTENSITY = (
        0x00380250  # R/W, long,    "DARKCALIB STABLE MAX INTENSITY"
    )
    SUBTRACT_DATASTATUS = 0x003802F0  # R/W    mode,    "SUBTRACT DATA STATUS"
    SHADINGCALIB_DATASTATUS = 0x00380300  # R/W    mode,    "SHADING CALIB DATA STATUS"
    SHADINGCORRECTION = 0x00380310  # R/W, mode,    "SHADING CORRECTION"
    SHADINGCALIBDATAMEMORY = 0x00380320  # R/W, mode,    "SHADING CALIB DATA MEMORY"
    STORESHADINGCALIBDATATOMEMORY = (
        0x00380330  # W/O, mode,    "STORE SHADING DATA TO MEMORY"
    )
    SHADINGCALIB_METHOD = 0x00380340  # R/W, mode,    "SHADING CALIB METHOD"
    SHADINGCALIB_TARGET = 0x00380350  # R/W, long,    "SHADING CALIB TARGET"
    SHADINGCALIB_STABLEMININTENSITY = (
        0x00380360  # R/W, long,    "SHADING CALIB STABLE MIN INTENSITY"
    )
    SHADINGCALIB_SAMPLES = 0x00380370  # R/W, long,    "SHADING CALIB SAMPLES"
    SHADINGCALIB_STABLESAMPLES = (
        0x00380380  # R/W, long,    "SHADING CALIB STABLE SAMPLES"
    )
    SHADINGCALIB_STABLEMAXERRORPERCENT = (
        0x00380390  # R/W, long,    "SHADING CALIB STABLE MAX ERROR PERCENT"
    )
    FRAMEAVERAGINGMODE = 0x003803A0  # R/W, mode,    "FRAME AVERAGING MODE"
    FRAMEAVERAGINGFRAMES = 0x003803B0  # R/W, long,    "FRAME AVERAGING FRAMES"
    DARKCALIB_STABLESAMPLES = 0x003803C0  # R/W, long,    "DARKCALIB STABLE SAMPLES"
    DARKCALIB_SAMPLES = 0x003803D0  # R/W, long,    "DARKCALIB SAMPLES"
    DARKCALIB_TARGET = 0x003803E0  # R/W, long,    "DARKCALIB TARGET"
    CAPTUREMODE = 0x00380410  # R/W, mode,    "CAPTURE MODE"
    LINEAVERAGING = 0x00380450  # R/W, long,    "LINE AVERAGING"
    INTENSITYLUT_MODE = 0x00380510  # R/W, mode,    "INTENSITY LUT MODE"
    INTENSITYLUT_PAGE = 0x00380520  # R/W, long,    "INTENSITY LUT PAGE"
    INTENSITYLUT_WHITECLIP = 0x00380530  # R/W, long,    "INTENSITY LUT WHITE CLIP"
    INTENSITYLUT_BLACKCLIP = 0x00380540  # R/W, long,    "INTENSITY LUT BLACK CLIP"
    INTENSITY_GAMMA = 0x00380560  # R/W, real,    "INTENSITY GAMMA"
    SENSORGAPCORRECT_MODE = 0x00380620  # R/W, long,    "SENSOR GAP CORRECT MODE"
    ADVANCEDEDGEENHANCEMENT_MODE = (
        0x00380630  # R/W, mode,    "ADVANCED EDGE ENHANCEMENT MODE"
    )
    ADVANCEDEDGEENHANCEMENT_LEVEL = (
        0x00380640  # R/W, long,    "ADVANCED EDGE ENHANCEMENT LEVEL"
    )

    # TAP CALIBRATION
    TAPGAINCALIB_METHOD = 0x00380F10  # R/W, mode,    "TAP GAIN CALIB METHOD"
    TAPCALIB_BASEDATAMEMORY = 0x00380F20  # R/W, mode,    "TAP CALIB BASE DATA MEMORY"
    STORETAPCALIBDATATOMEMORY = (
        0x00380F30  # W/O, mode,    "STORE TAP CALIB DATA TO MEMORY"
    )
    TAPCALIBDATAMEMORY = 0x00380F40  # W/O, mode,    "TAP CALIB DATA MEMORY"
    NUMBEROF_TAPCALIB = 0x00380FF0  # R/W, long,    "NUMBER OF TAP CALIB"
    TAPCALIB_GAIN = 0x00381000  # R/W, mode,    "TAP CALIB GAIN"
    TAPCALIB_OFFSET = 0x00382000  # R/W, mode,    "TAP CALIB OFFSET"
    _TAPCALIB = 0x00000010  # the offset of ID for Nth TAPCALIB

    # Group: READOUT
    # readout speed
    READOUTSPEED = 0x00400110  # R/W, long,    "READOUT SPEED"
    #      0x00400120 is reserved
    READOUT_DIRECTION = 0x00400130  # R/W, mode,    "READOUT DIRECTION"
    READOUT_UNIT = 0x00400140  # R/O, mode,    "READOUT UNIT"

    SHUTTER_MODE = 0x00400150  # R/W, mode,    "SHUTTER MODE"

    # sensor mode
    SENSORMODE = 0x00400210  # R/W, mode,    "SENSOR MODE"
    SENSORMODE_SLITHEIGHT = (
        0x00400220  # R/W, long,    "SENSOR MODE SLIT HEIGHT"            # reserved
    )
    SENSORMODE_LINEBUNDLEHEIGHT = (
        0x00400250  # R/W, long,    "SENSOR MODE LINE BUNDLEHEIGHT"
    )
    SENSORMODE_FRAMINGHEIGHT = (
        0x00400260  # R/W, long,    "SENSOR MODE FRAMING HEIGHT"        # reserved
    )
    SENSORMODE_PANORAMICSTARTV = (
        0x00400280  # R/W, long,    "SENSOR MODE PANORAMIC START V"
    )

    # other readout mode
    CCDMODE = 0x00400310  # R/W, mode,    "CCD MODE"
    EMCCD_CALIBRATIONMODE = 0x00400320  # R/W, mode,    "EM CCD CALIBRATION MODE"
    CMOSMODE = 0x00400350  # R/W, mode,    "CMOS MODE"

    # output mode
    OUTPUT_INTENSITY = 0x00400410  # R/W, mode,    "OUTPUT INTENSITY"
    OUTPUTDATA_ORIENTATION = (
        0x00400420  # R/W, mode,    "OUTPUT DATA ORIENTATION"        # reserved
    )
    OUTPUTDATA_ROTATION = (
        0x00400430  # R/W, degree,    "OUTPUT DATA ROTATION"            # reserved
    )
    OUTPUTDATA_OPERATION = 0x00400440  # R/W, mode,    "OUTPUT DATA OPERATION"

    TESTPATTERN_KIND = 0x00400510  # R/W, mode,    "TEST PATTERN KIND"
    TESTPATTERN_OPTION = 0x00400520  # R/W, long,    "TEST PATTERN OPTION"

    EXTRACTION_MODE = 0x00400620  # R/W    mode,    "EXTRACTION MODE    "

    # Group: ROI
    # binning and subarray
    BINNING = 0x00401110  # R/W, mode,    "BINNING"
    BINNING_INDEPENDENT = 0x00401120  # R/W, mode,    "BINNING INDEPENDENT"
    BINNING_HORZ = 0x00401130  # R/W, long,    "BINNING HORZ"
    BINNING_VERT = 0x00401140  # R/W, long,    "BINNING VERT"
    SUBARRAYHPOS = 0x00402110  # R/W, long,    "SUBARRAY HPOS"
    SUBARRAYHSIZE = 0x00402120  # R/W, long,    "SUBARRAY HSIZE"
    SUBARRAYVPOS = 0x00402130  # R/W, long,    "SUBARRAY VPOS"
    SUBARRAYVSIZE = 0x00402140  # R/W, long,    "SUBARRAY VSIZE"
    SUBARRAYMODE = 0x00402150  # R/W, mode,    "SUBARRAY MODE"
    DIGITALBINNING_METHOD = 0x00402160  # R/W, mode,    "DIGITALBINNING METHOD"
    DIGITALBINNING_HORZ = 0x00402170  # R/W, long,    "DIGITALBINNING HORZ"
    DIGITALBINNING_VERT = 0x00402180  # R/W, long,    "DIGITALBINNING VERT"

    # Group: TIMING
    # synchronous timing
    TIMING_READOUTTIME = 0x00403010  # R/O, sec,    "TIMING READOUT TIME"
    TIMING_CYCLICTRIGGERPERIOD = (
        0x00403020  # R/O, sec,    "TIMING CYCLIC TRIGGER PERIOD"
    )
    TIMING_MINTRIGGERBLANKING = (
        0x00403030  # R/O, sec,    "TIMING MINIMUM TRIGGER BLANKING"
    )
    #      0x00403040 is reserved
    TIMING_MINTRIGGERINTERVAL = (
        0x00403050  # R/O, sec,    "TIMING MINIMUM TRIGGER INTERVAL"
    )
    TIMING_EXPOSURE = 0x00403060  # R/O, mode,    "TIMING EXPOSURE"
    TIMING_INVALIDEXPOSUREPERIOD = 0x00403070  # R/O, sec,    "INVALID EXPOSURE PERIOD"
    TIMING_FRAMESKIPNUMBER = 0x00403080  # R/W, long,    "TIMING FRAME SKIP NUMBER"
    TIMING_GLOBALEXPOSUREDELAY = (
        0x00403090  # R/O, sec,    "TIMING GLOBAL EXPOSURE DELAY"
    )

    INTERNALFRAMERATE = 0x00403810  # R/W, 1/sec,    "INTERNAL FRAME RATE"
    INTERNAL_FRAMEINTERVAL = 0x00403820  # R/W, sec,    "INTERNAL FRAME INTERVAL"
    INTERNALLINERATE = 0x00403830  # R/W, 1/sec,    "INTERNAL LINE RATE"
    INTERNALLINESPEED = 0x00403840  # R/W, m/sec,    "INTERNAL LINE SPEEED"
    INTERNAL_LINEINTERVAL = 0x00403850  # R/W, sec,    "INTERNAL LINE INTERVAL"

    # system information

    TIMESTAMP_PRODUCER = 0x00410A10  # R/O, mode,    "TIME STAMP PRODUCER"
    FRAMESTAMP_PRODUCER = 0x00410A20  # R/O, mode,    "FRAME STAMP PRODUCER"

    # Group: READOUT

    # image information
    #      0x00420110 is reserved
    COLORTYPE = 0x00420120  # R/W, mode,    "COLORTYPE"
    BITSPERCHANNEL = 0x00420130  # R/W, long,    "BIT PER CHANNEL"
    #      0x00420140 is reserved
    #      0x00420150 is reserved

    NUMBEROF_CHANNEL = 0x00420180  # R/O, long,    "NUMBER OF CHANNEL"
    ACTIVE_CHANNELINDEX = 0x00420190  # R/W, mode,    "ACTIVE CHANNEL INDEX"
    NUMBEROF_VIEW = 0x004201C0  # R/O, long,    "NUMBER OF VIEW"
    ACTIVE_VIEWINDEX = 0x004201D0  # R/W, mode,    "ACTIVE VIEW INDEX"

    IMAGE_WIDTH = 0x00420210  # R/O, long,    "IMAGE WIDTH"
    IMAGE_HEIGHT = 0x00420220  # R/O, long,    "IMAGE HEIGHT"
    IMAGE_ROWBYTES = 0x00420230  # R/O, long,    "IMAGE ROWBYTES"
    IMAGE_FRAMEBYTES = 0x00420240  # R/O, long,    "IMAGE FRAMEBYTES"
    IMAGE_TOPOFFSETBYTES = (
        0x00420250  # R/O, long,    "IMAGE TOP OFFSET BYTES"        # reserved
    )
    IMAGE_PIXELTYPE = 0x00420270  # R/W, EPixelType,    "IMAGE PIXEL TYPE"
    IMAGE_CAMERASTAMP = 0x00420300  # R/W, long,    "IMAGE CAMERA STAMP"

    BUFFER_ROWBYTES = 0x00420330  # R/O, long,    "BUFFER ROWBYTES"
    BUFFER_FRAMEBYTES = 0x00420340  # R/O, long,    "BUFFER FRAME BYTES"
    BUFFER_TOPOFFSETBYTES = 0x00420350  # R/O, long,    "BUFFER TOP OFFSET BYTES"
    BUFFER_PIXELTYPE = 0x00420360  # R/O, EPixelType,    "BUFFER PIXEL TYPE"

    RECORDFIXEDBYTES_PERFILE = (
        0x00420410  # R/O,    long    "RECORD FIXED BYTES PER FILE"
    )
    RECORDFIXEDBYTES_PERSESSION = (
        0x00420420  # R/O,    long    "RECORD FIXED BYTES PER SESSION"
    )
    RECORDFIXEDBYTES_PERFRAME = (
        0x00420430  # R/O,    long    "RECORD FIXED BYTES PER FRAME"
    )

    IMAGEDETECTOR_PIXELWIDTH = (
        0x00420810  # R/O, micro-meter, "IMAGE DETECTOR PIXEL WIDTH"        # reserved
    )
    IMAGEDETECTOR_PIXELHEIGHT = (
        0x00420820  # R/O, micro-meter, "IMAGE DETECTOR PIXEL HEIGHT"        # reserved
    )

    # frame bundle
    FRAMEBUNDLE_MODE = 0x00421010  # R/W, mode,    "FRAMEBUNDLE MODE"
    FRAMEBUNDLE_NUMBER = 0x00421020  # R/W, long,    "FRAMEBUNDLE NUMBER"
    FRAMEBUNDLE_ROWBYTES = 0x00421030  # R/O,    long,    "FRAMEBUNDLE ROWBYTES"
    FRAMEBUNDLE_FRAMESTEPBYTES = (
        0x00421040  # R/O, long,    "FRAMEBUNDLE FRAME STEP BYTES"
    )

    # partial area
    NUMBEROF_PARTIALAREA = 0x00430010  # R/W, long,    "NUMBER OF PARTIAL AREA"
    PARTIALAREA_HPOS = 0x00431000  # R/W, long,    "PARTIAL AREA HPOS"
    PARTIALAREA_HSIZE = 0x00432000  # R/W, long,    "PARTIAL AREA HSIZE"
    PARTIALAREA_VPOS = 0x00433000  # R/W, long,    "PARTIAL AREA VPOS"
    PARTIALAREA_VSIZE = 0x00434000  # R/W, long,    "PARTIAL AREA VSIZE"
    _PARTIALAREA = 0x00000010  # the offset of ID for Nth PARTIAL AREA

    # multi line
    NUMBEROF_MULTILINE = 0x0044F010  # R/W, long,    "NUMBER OF MULTI LINE"
    MULTILINE_VPOS = 0x00450000  # R/W, long,    "MULTI LINE VPOS"
    MULTILINE_VSIZE = 0x00460000  # R/W, long,    "MULTI LINE VSIZE"
    #                 - 0x0046FFFF for 4096 MULTI LINEs                    # reserved
    _MULTILINE = 0x00000010  # the offset of ID for Nth MULTI LINE

    # defect
    DEFECTCORRECT_MODE = 0x00470010  # R/W, mode,    "DEFECT CORRECT MODE"
    NUMBEROF_DEFECTCORRECT = 0x00470020  # R/W, long,    "NUMBER OF DEFECT CORRECT"
    HOTPIXELCORRECT_LEVEL = 0x00470030  # R/W, mode,    "HOT PIXEL CORRECT LEVEL"
    DEFECTCORRECT_HPOS = 0x00471000  # R/W, long,    "DEFECT CORRECT HPOS"
    DEFECTCORRECT_METHOD = 0x00473000  # R/W, mode,    "DEFECT CORRECT METHOD"
    #                 - 0x0047FFFF for 256 DEFECT
    _DEFECTCORRECT = 0x00000010  # the offset of ID for Nth DEFECT

    # Group: CALIBREGION
    CALIBREGION_MODE = 0x00402410  # R/W, mode,    "CALIBRATE REGION MODE"
    NUMBEROF_CALIBREGION = 0x00402420  # R/W, long,    "NUMBER OF CALIBRATE REGION"
    CALIBREGION_HPOS = 0x004B0000  # R/W, long,    "CALIBRATE REGION HPOS"
    CALIBREGION_HSIZE = 0x004B1000  # R/W, long,    "CALIBRATE REGION HSIZE"
    #                 - 0x0048FFFF for 256 REGIONs at least
    _CALIBREGION = 0x00000010  # the offset of ID for Nth REGION

    # Group: MASKREGION
    MASKREGION_MODE = 0x00402510  # R/W, mode,    "MASK REGION MODE"
    NUMBEROF_MASKREGION = 0x00402520  # R/W, long,    "NUMBER OF MASK REGION"
    MASKREGION_HPOS = 0x004C0000  # R/W, long,    "MASK REGION HPOS"
    MASKREGION_HSIZE = 0x004C1000  # R/W, long,    "MASK REGION HSIZE"
    #                 - 0x0048FFFF for 256 REGIONs at least
    _MASKREGION = 0x00000010  # the offset of ID for Nth REGION

    # Group: Camera Status
    CAMERASTATUS_INTENSITY = 0x004D1110  # R/O, mode,    "CAMERASTATUS INTENSITY"
    CAMERASTATUS_INPUTTRIGGER = 0x004D1120  # R/O, mode,    "CAMERASTATUS INPUT TRIGGER"
    CAMERASTATUS_CALIBRATION = 0x004D1130  # R/O, mode,    "CAMERASTATUS CALIBRATION"

    # Group: Back Focus Position
    BACKFOCUSPOS_TARGET = 0x00804010  # R/W, micro-meter,"BACK FOCUS POSITION TARGET"
    BACKFOCUSPOS_CURRENT = 0x00804020  # R/O, micro-meter,"BACK FOCUS POSITION CURRENT"
    BACKFOCUSPOS_LOADFROMMEMORY = (
        0x00804050  # R/W, long, "BACK FOCUS POSITION LOAD FROM MEMORY"
    )
    BACKFOCUSPOS_STORETOMEMORY = (
        0x00804060  # W/O, long, "BACK FOCUS POSITION STORE TO MEMORY"
    )

    # Group: SYSTEM
    # system property

    SYSTEM_ALIVE = 0x00FF0010  # R/O, mode,    "SYSTEM ALIVE"

    CONVERSIONFACTOR_COEFF = 0x00FFE010  # R/O, double,    "CONVERSION FACTOR COEFF"
    CONVERSIONFACTOR_OFFSET = 0x00FFE020  # R/O, double,    "CONVERSION FACTOR OFFSET"

    # -- options --

    # option
    _RATIO = 0x80000000
    EXPOSURETIME_RATIO = _RATIO | EXPOSURETIME  # reserved
    # R/W, real,    "EXPOSURE TIME RATIO"                    # reserved
    CONTRASTGAIN_RATIO = _RATIO | CONTRASTGAIN  # reserved
    # R/W, real,    "CONTRAST GAIN RATIO"                    # reserved

    _CHANNEL = 0x00000001
    _VIEW = 0x01000000

    _MASK_CHANNEL = 0x0000000F
    _MASK_VIEW = 0x0F000000
    _MASK_BODY = 0x00FFFFF0

    # for backward compativilities
    REMOTE_VALUE = EPropAttr.VOLATILE.value

    PHOTONIMAGING_MODE__0 = 0
    PHOTONIMAGING_MODE__1 = 1
    PHOTONIMAGING_MODE__2 = 2

    SCAN_MODE = ESensorMode.AREA
    SLITSCAN_HEIGHT = SENSORMODE_SLITHEIGHT

    FRAME_BUNDLEMODE = FRAMEBUNDLE_MODE
    FRAME_BUNDLENUMBER = FRAMEBUNDLE_NUMBER
    FRAME_BUNDLEROWBYTES = FRAMEBUNDLE_ROWBYTES

    ACTIVE_VIEW = ACTIVE_VIEWINDEX  # reserved
    ACTIVE_VIEWINDEXES = ACTIVE_VIEWINDEX  # reserved
    SYNCMULTIVIEWREADOUT = SYNC_MULTIVIEWEXPOSURE  # reserved
    #    SYNC_FRAMEREADOUTTIME=TIMING_READOUTTIME,                    # reserved
    #    SYNC_CYCLICTRIGGERPERIOD = TIMING_CYCLICTRIGGERPERIOD,        # reserved
    SYNC_MINTRIGGERBLANKING = TIMING_MINTRIGGERBLANKING
    SYNC_FRAMEINTERVAL = INTERNAL_FRAMEINTERVAL
    LOWLIGHTSENSITIVITY = PHOTONIMAGINGMODE

    DARKCALIB_MAXIMUMINTENSITY = DARKCALIB_STABLEMAXINTENSITY
    SUBTRACT_SAMPLINGCOUNT = DARKCALIB_SAMPLES

    SHADINGCALIB_MINIMUMINTENSITY = SHADINGCALIB_STABLEMININTENSITY
    SHADINGCALIB_STABLEFRAMECOUNT = SHADINGCALIB_STABLESAMPLES
    SHADINGCALIB_INTENSITYMAXIMUMERRORPERCENTAGE = SHADINGCALIB_STABLEMAXERRORPERCENT
    SHADINGCALIB_AVERAGEFRAMECOUNT = SHADINGCALIB_SAMPLES

    def to_enum(self):
        return PROP_ENUM_MAP.get(self)


PROP_ENUM_MAP = {
    EProp.TRIGGERSOURCE: ETriggerSource,
    EProp.SYSTEM_ALIVE: ESystemAlive,
    EProp.SENSORMODE: ESensorMode,
    EProp.COLORTYPE: EColorType,
    EProp.SHUTTER_MODE: EShutterMode,
    EProp.READOUTSPEED: EReadoutSpeed,
    EProp.READOUT_DIRECTION: EReadoutDirection,
    EProp.TRIGGERACTIVE: ETriggerActive,
    EProp.READOUT_UNIT: EReadoutUnit,
    EProp.CCDMODE: ECCDMode,
    EProp.CMOSMODE: ECMOSMode,
    EProp.OUTPUT_INTENSITY: EOutputIntensity,
    EProp.OUTPUTDATA_ORIENTATION: EOutputDataOrientation,
    EProp.OUTPUTDATA_OPERATION: EOutputDataOperation,
    EProp.TESTPATTERN_KIND: ETestPatternKind,
    EProp.DIGITALBINNING_METHOD: EDigitalBinningMethod,
    EProp.TRIGGER_MODE: ETriggerMode,
    EProp.TRIGGERPOLARITY: ETriggerPolarity,
    EProp.TRIGGER_CONNECTOR: ETriggerConnector,
    EProp.INTERNALTRIGGER_HANDLING: EInternalTriggerHandling,
    EProp.SYNCREADOUT_SYSTEMBLANK: ESyncReadoutSystemBlank,
    EProp.TRIGGERENABLE_ACTIVE: ETriggerEnableActive,
    EProp.TRIGGERENABLE_POLARITY: ETriggerEnablePolarity,
    EProp.OUTPUTTRIGGER_CHANNELSYNC: EOutputTriggerChannelSync,
    EProp.OUTPUTTRIGGER_PROGRAMABLESTART: EOutputTriggerProgramableStart,
    EProp.OUTPUTTRIGGER_SOURCE: EOutputTriggerSource,
    EProp.OUTPUTTRIGGER_POLARITY: EOutputTriggerPolarity,
    EProp.OUTPUTTRIGGER_ACTIVE: EOutputTriggerActive,
    EProp.OUTPUTTRIGGER_KIND: EOutputTriggerKind,
    EProp.BINNING: EBinning,
    EProp.SUBARRAYMODE: ESubArrayMode,
    EProp.TIMING_EXPOSURE: ETimmingExposure,
    EProp.FRAME_BUNDLEMODE: EFrameBundleMode,
    EProp.IMAGE_PIXELTYPE: EImagePixelType,
    EProp.BUFFER_PIXELTYPE: EBufferPixelType,
    EProp.EXPOSURETIME_CONTROL: EExposureTimeControl,
    EProp.DIRECTGAIN_MODE: EDirectGainMode,
    EProp.DEFECTCORRECT_MODE: EDefectCorrectMode,
    EProp.TIMESTAMP_PRODUCER: ETimeStampProducer,
    EProp.FRAMESTAMP_PRODUCER: EFrameStampProducer,
}


class DCAMError(Exception):
    @classmethod
    def name(cls):
        return cls.__name__

    @property
    def error_code(self):
        return self.args[0]

    @property
    def location(self):
        return self.args[1] if len(self.args) > 1 else "?"

    def __repr__(self):
        code = self.error_code
        return f"{self.name()}({code.name}, {self.location!r})"

    def __str__(self):
        code = self.error_code
        return f"{self.name()}: {self.location!r} raised {code.name} ({code.value})"


# Hamamatsu structures.

## DCAMAPI_INIT
#
# The dcam initialization structure
#
class SInit(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iDeviceCount", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("initoptionbytes", ctypes.c_int32),
        ("initoption", ctypes.POINTER(ctypes.c_int32)),
        ("guid", ctypes.POINTER(ctypes.c_int32)),
    ]


## DCAMDEV_OPEN
#
# The dcam open structure
#
class SOpen(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("hdcam", ctypes.c_void_p),
    ]


## DCAMWAIT_OPEN
#
# The dcam wait open structure
#
class SWaitOpen(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("supportevent", ctypes.c_int32),
        ("hwait", ctypes.c_void_p),
        ("hdcam", ctypes.c_void_p),
    ]


## DCAMWAIT_START
#
# The dcam wait start structure
#
class SWaitStart(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("eventhappened", ctypes.c_int32),
        ("eventmask", ctypes.c_int32),
        ("timeout", ctypes.c_int32),
    ]


## DCAMCAP_TRANSFERINFO
#
# The dcam capture info structure
#
class STransferInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iKind", ctypes.c_int32),
        ("nNewestFrameIndex", ctypes.c_int32),
        ("nFrameCount", ctypes.c_int32),
    ]


## DCAMBUF_ATTACH
#
# The dcam buffer attachment structure
#
class SAttach(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iKind", ctypes.c_int32),
        ("buffer", ctypes.POINTER(ctypes.c_void_p)),
        ("buffercount", ctypes.c_int32),
    ]


## DCAMBUF_FRAME
#
# The dcam buffer frame structure
#
class SFrame(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iKind", ctypes.c_int32),
        ("option", ctypes.c_int32),
        ("iFrame", ctypes.c_int32),
        ("buf", ctypes.c_void_p),
        ("rowbytes", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("timestamp", ctypes.c_int32),
        ("framestamp", ctypes.c_int32),
        ("camerastamp", ctypes.c_int32),
    ]


## DCAMDEV_STRING
#
# The dcam device string structure
#
class SString(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iString", ctypes.c_int32),
        ("text", ctypes.c_char_p),
        ("textbytes", ctypes.c_int32),
    ]


## DCAMPROP_ATTR
#
# The dcam property attribute structure.
#
class Attr(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_int32),
        ("iProp", ctypes.c_int32),
        ("option", ctypes.c_int32),
        ("iReserved1", ctypes.c_int32),
        ("attribute", ctypes.c_int32),
        ("iGroup", ctypes.c_int32),
        ("iUnit", ctypes.c_int32),
        ("attribute2", ctypes.c_int32),
        ("valuemin", ctypes.c_double),
        ("valuemax", ctypes.c_double),
        ("valuestep", ctypes.c_double),
        ("valuedefault", ctypes.c_double),
        ("nMaxChannel", ctypes.c_int32),
        ("iReserved3", ctypes.c_int32),
        ("nMaxView", ctypes.c_int32),
        ("iProp_NumberOfElement", ctypes.c_int32),
        ("iProp_ArrayBase", ctypes.c_int32),
        ("iPropStep_Element", ctypes.c_int32),
    ]


## DCAMPROP_VALUETEXT
#
# The dcam text property structure.
#
class ValueText(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_int32),
        ("iProp", ctypes.c_int32),
        ("value", ctypes.c_double),
        ("text", ctypes.c_char_p),
        ("textbytes", ctypes.c_int32),
    ]


class FrameStream:
    def __init__(self, device, nb_frames):
        self.device = device
        self.nb_frames = nb_frames
        self.device._buf_alloc(nb_frames)
        self.stream = device.frame_stream(nb_frames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.device._buf_release()
        except DCAMError as error:
            logging.error("Could not release buffer. Reason %r", error)

    def __len__(self):
        return self.nb_frames

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.stream)


class EventStream:

    DefaultMask = EWaitEvent.CAP_FRAMEREADY | EWaitEvent.CAP_STOPPED

    def __init__(self, device, mask=DefaultMask, timeout=TIMEOUT_INFINITE):
        self.device = device
        self.mask = mask
        self.timeout = timeout
        self._handle = device._wait_open()
        self.stream = self.device.event_stream(
            self.mask.value, self.timeout, self._handle
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.stream)

    def abort(self):
        if self._handle is not None:
            self.device._wait_abort(self._handle)

    def close(self):
        if self._handle is not None:
            # should call abort in case another thread is waiting to the event ?
            self.abort()
            self.device._wait_close(self._handle)
            self._handle = None


class Stream:
    def __init__(self, device, nb_frames):
        self.device = device
        self.nb_frames = nb_frames
        self.frame_stream = FrameStream(device, nb_frames)
        self.event_stream = EventStream(device)
        self.context_stack = contextlib.ExitStack()
        tstream = self.device.transfer_stream()
        self.stream = stream(self.frame_stream, self.event_stream, tstream)

    def __enter__(self):
        self.context_stack.enter_context(self.frame_stream)
        self.context_stack.enter_context(self.event_stream)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def close(self):
        self.context_stack.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.stream)


def stream(fstream, estream, tstream):
    last_frame_index = -1
    for event in estream:
        if event is EWaitEvent.CAP_STOPPED:
            break
        elif event is EWaitEvent.CAP_FRAMEREADY:
            transfer = next(tstream)
            while transfer.nNewestFrameIndex > last_frame_index:
                yield next(fstream)
                last_frame_index += 1


def copy_frame(frame, into=None):
    pixel_type = EImagePixelType(frame.type)
    if into is None:
        pixel_bytes = pixel_type.bytes_per_pixel()
        nbytes = frame.width * frame.height * pixel_bytes
        into = numpy.empty(nbytes, dtype=numpy.uint8)
    ctypes.memmove(into.ctypes.data, frame.buf, into.nbytes)
    dtype = pixel_type.dtype()
    if dtype:
        into.dtype = dtype
        into.shape = frame.width, frame.height
    return into


class Attribute(dict):
    def __getattr__(self, name):
        return self[name]

    def __dir__(self):
        return sorted(self)

    @property
    def value(self):
        return self.read()


class Device:
    def __init__(self, lib, camera_id):
        self._lib = lib
        self.camera_id = camera_id
        self._handle = None
        self.capabilities = None
        self.capability_names = None
        self._info = None

    def __del__(self):
        self.close()

    def __enter__(self):
        if not self.is_open():
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.close()
        except:
            pass

    def _build_capabilities(self):
        prop_id = ctypes.c_int32(0)
        buff_size = ctypes.c_int32(64)
        buff = ctypes.create_string_buffer(buff_size.value)

        capabilities = {}
        capability_names = {}
        while True:
            try:
                self._lib.dcamprop_getnextid(
                    self._handle,
                    ctypes.byref(prop_id),
                    ctypes.c_int32(EPropOption.SUPPORT),
                )
            except Exception:
                break
            if not prop_id.value:
                break
            self._lib.dcamprop_getname(self._handle, prop_id, buff, buff_size)
            eprop = EProp(prop_id.value)
            prop_name = buff.value.decode()
            prop_uname = prop_name.lower().replace(" ", "_")
            attr_dict = Attribute(name=prop_name, uname=prop_uname, prop=eprop)
            capabilities[eprop] = attr_dict
            capability_names[prop_name] = attr_dict
            capability_names[prop_uname] = attr_dict
            attr = Attr()
            attr.cbSize = ctypes.sizeof(attr)
            attr.iProp = prop_id.value
            self._lib.dcamprop_getattr(self._handle, ctypes.byref(attr))
            attr_dict["attribute"] = EPropAttr(attr.attribute)
            attr_dict["id"] = attr.iProp
            attr_dict["unit"] = EUnit(attr.iUnit)
            attr_dict["min_value"] = attr.valuemin
            attr_dict["max_value"] = attr.valuemax
            attr_dict["step_value"] = attr.valuestep
            attr_dict["default_value"] = attr.valuedefault
            attr_dict["max_view"] = attr.nMaxView
            attr_dict["max_channel"] = attr.nMaxChannel
            attr_dict["dtype"] = dtype = attr.attribute & EPropAttr.TYPE_MASK
            attr_dict["read"], attr_dict["write"] = self._make_read_write(attr_dict)
            if EPropAttr.HASVALUETEXT in attr_dict["attribute"]:
                attr_dict["enum"] = self._get_property_options(attr_dict)
                attr_dict["enum_values"] = {v: k for k, v in attr_dict["enum"].items()}
        self.capabilities = capabilities
        self.capability_names = capability_names

    def _make_read_write(self, cap):
        eprop = cap["prop"]
        dtype = cap["dtype"]
        cid = cap["id"]
        name = cap["uname"]
        enum_type = eprop.to_enum()
        if enum_type is not None:
            decode = lambda v: enum_type(int(v))
        elif dtype in {EPropAttr.TYPE_LONG, EPropAttr.TYPE_MODE, EPropAttr.TYPE_MASK}:
            decode = int
        elif dtype == EPropAttr.TYPE_REAL:
            decode = lambda x: x

        def read():
            c_value = ctypes.c_double(0)
            self._lib.dcamprop_getvalue(self._handle, cid, ctypes.byref(c_value))
            return decode(c_value.value)

        def write(value):
            c_value = ctypes.c_double(value)
            self._lib.dcamprop_setgetvalue(self._handle, cid, ctypes.byref(c_value), 0)
            return decode(c_value.value)

        return read, write

    def _get_property_options(self, cap):
        curr_value = ctypes.c_double(cap["min_value"])
        prop_text = ValueText()
        c_buf_len = 64
        c_buf = ctypes.create_string_buffer(c_buf_len)
        prop_text.cbSize = ctypes.sizeof(prop_text)
        prop_text.iProp = cap["id"]
        prop_text.value = curr_value
        prop_text.text = ctypes.addressof(c_buf)
        prop_text.textbytes = c_buf_len
        # Collect text options.
        text_options = {}
        while True:
            # Get text of current value.
            self._lib.dcamprop_getvaluetext(self._handle, ctypes.byref(prop_text)),
            text_options[prop_text.text.decode()] = int(curr_value.value)

            # Get next value.
            try:
                ret = self._lib.dcamprop_queryvalue(
                    self._handle,
                    cap["id"],
                    ctypes.byref(curr_value),
                    ctypes.c_int32(EPropOption.NEXT),
                )
            except DCAMError as error:
                if error.error_code == EError.OUTOFRANGE:
                    break
                raise
            prop_text.value = curr_value
        return text_options

    def __getitem__(self, id):
        try:
            return self.capabilities[id]
        except KeyError:
            return self.capability_names[id]

    def __setitem__(self, id, value):
        return self[id].write(value)

    def __contains__(self, id):
        return id in self.capabilities or id in self.capability_names

    def __len__(self):
        return len(self.capabilities)

    def __iter__(self):
        return iter(self.capabilities)

    def __repr__(self):
        try:
            info = self.info
            name = f"{info[EIDString.VENDOR]} {info[EIDString.MODEL]}"
            details = [f"  {k.name}: {v}" for k, v in info.items()]
            return name + "\n" + "\n".join(details)
        except Exception as error:
            name = type(self).__name__
            return f"{name}: {error!r}"

    def __str__(self):
        try:
            info = self.info
            return f"{info[EIDString.VENDOR]}({info[EIDString.MODEL]})"
        except Exception as error:
            name = type(self).__name__
            return f"{name}: {error!r}"

    def values(self):
        return self.capabilities.values()

    def keys(self):
        return self.capabilities.keys()

    def items(self):
        return self.capabilities.items()

    def _lock_frame_index(self, frame_index):
        frame = SFrame(0, 0, 0, frame_index, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        frame.size = ctypes.sizeof(frame)
        return self._lock_frame(frame)

    def _lock_frame(self, frame):
        self._lib.dcambuf_lockframe(self._handle, ctypes.byref(frame))
        return frame

    def _get_transfer_info(self, transfer):
        self._lib.dcamcap_transferinfo(self._handle, ctypes.byref(transfer))
        return transfer

    def _wait_open(self):
        wait_open = SWaitOpen(0, 0, None, self._handle)
        wait_open.size = ctypes.sizeof(wait_open)
        self._lib.dcamwait_open(ctypes.byref(wait_open))
        return ctypes.c_void_p(wait_open.hwait)

    def _wait_abort(self, handle):
        self._lib.dcamwait_abort(handle)

    def _wait_close(self, handle):
        self._lib.dcamwait_close(handle)

    def _wait_start(self, handle, wait_start):
        try:
            self._lib.dcamwait_start(handle, ctypes.byref(wait_start))
        except:
            pass
        return EWaitEvent(wait_start.eventhappened)

    def _buf_alloc(self, nb_frames):
        self._lib.dcambuf_alloc(self._handle, nb_frames)

    def _buf_release(self):
        # can only be called when status != BUSY, so must stop acquisition first
        self._lib.dcambuf_release(self._handle, EAttach.FRAME)

    def is_open(self):
        return self._handle is not None

    def open(self):
        if self.is_open():
            return
        popen = SOpen(0, self.camera_id, None)
        popen.size = ctypes.sizeof(popen)
        self._lib.dcamdev_open(ctypes.byref(popen))
        self._handle = ctypes.c_void_p(popen.hdcam)

        # initialize capabilities
        self._build_capabilities()

    def close(self):
        if self._handle is not None:
            self._lib.dcamdev_close(self._handle)
            self._handle = None
        self.capabilities = None
        self.capability_names = None
        self._info = None

    def start(self, live=False):
        # Not to be used directly, first need to setup buffer!
        mode = EStart.SEQUENCE if live else EStart.SNAP
        self._lib.dcamcap_start(self._handle, mode)

    def stop(self):
        self._lib.dcamcap_stop(self._handle)

    def frame_stream(self, nb_frames):
        # Unsafe method: need to call dcambuf_alloc first!
        frame = SFrame(0, 0, 0, -1, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        frame.size = ctypes.sizeof(frame)
        for i in range(nb_frames):
            frame.iFrame = i
            yield self._lock_frame(frame)

    def event_stream(self, mask, timeout, handle):
        # Pass timeout in seconds
        timeout = int(timeout * 1000)
        param = SWaitStart(0, 0, mask, timeout)
        while True:
            try:
                yield self._wait_start(handle, param)
            except DCAMError as error:
                if error.error_code is EError.ABORT:
                    return
                else:
                    raise

    def transfer_stream(self):
        transfer = STransferInfo(0, ETransfer.FRAME, 0, 0)
        transfer.size = ctypes.sizeof(transfer)
        while True:
            yield self._get_transfer_info(transfer)

    @property
    def info(self):
        if self._info is None:
            buff_size = ctypes.c_int32(256)
            buff = ctypes.create_string_buffer(buff_size.value)
            pars = {
                EIDString.BUS,
                EIDString.CAMERAID,
                EIDString.VENDOR,
                EIDString.MODEL,
                EIDString.CAMERAVERSION,
                EIDString.DRIVERVERSION,
                EIDString.MODULEVERSION,
                EIDString.DCAMAPIVERSION,
                EIDString.CAMERA_SERIESNAME,
            }
            info = {}
            for par in pars:
                param = SString(
                    0, par.value, ctypes.cast(buff, ctypes.c_char_p), buff_size
                )
                param.size = ctypes.sizeof(param)
                try:
                    self._lib.dcamdev_getstring(self._handle, ctypes.byref(param))
                except Exception:
                    continue
                info[par] = buff.value.decode()
            self._info = info
        return self._info

    @property
    def status(self):
        status = ctypes.c_int32(0)
        self._lib.dcamcap_status(self._handle, ctypes.byref(status))
        return EStatus(status.value)

    @property
    def last_error(self):
        c_buf_len = 80
        c_buf = ctypes.create_string_buffer(c_buf_len)
        try:
            self._lib.dcam_getlasterror(self._handle, c_buf, c_buf_len)
        except:
            pass
        return c_buf.value.decode()

    # physical properties

    @property
    def pixel_size(self):
        """Pixel size (x, y). Units in meter"""
        w = self["image_detector_pixel_width"]
        h = self["image_detector_pixel_height"]
        width = w.unit.to_SI(w.read())
        height = h.unit.to_SI(h.read())
        return width, height

    # trigger
    def fire_software_trigger(self):
        self._lib.dcamcap_firetrigger(self._handle, 0)


class DCAM:
    def __init__(self):
        self._state = None
        self._devices = weakref.WeakValueDictionary()
        self._lib = ctypes.windll.dcamapi
        # force reference to uninit and close so that when close() is invoked
        # by __del__ it doesn't try to inject new members into a dying object
        self.dcam_uninit = self._lib.dcam_uninit
        self.dcamdev_close = self._lib.dcamdev_close

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self.nb_devices

    def __iter__(self):
        return (self[i] for i in range(self.nb_devices))

    def __getitem__(self, device_id):
        if 0 <= device_id < self.nb_devices:
            dev = self._devices.get(device_id)
            if dev is None:
                self._devices[device_id] = dev = Device(self, device_id)
            return dev
        raise KeyError(f"Device {device_id!r} not present")

    def __getattr__(self, name):
        member = getattr(self._lib, name)
        member.restype = ctypes.c_uint32
        if callable(member):

            @functools.wraps(member)
            def func(*args, **kwargs):
                r = member(*args, **kwargs)
                if r != EError.SUCCESS and ctypes.c_int32(r).value < 0:
                    raise DCAMError(EError(r), name)

            setattr(self, name, func)
            return func
        return member

    def is_open(self):
        return self._state is not None

    def open(self):
        if self.is_open():
            return
        state = SInit(0, 0, 0, 0, None, None)
        state.size = ctypes.sizeof(state)
        self.dcamapi_init(ctypes.byref(state))
        self._state = state

    @property
    def nb_devices(self):
        return self._state.iDeviceCount

    def close(self):
        if self._state is not None:
            for device in self._devices:
                device.close()
            self.dcam_uninit()
            self._state = None


dcam = DCAM()


def gen_acquire(device, exposure_time=1, nb_frames=1):
    """Simple acquisition example"""
    device["exposure_time"] = exposure_time
    with Stream(device, nb_frames) as stream:
        logging.info("start acquisition")
        device.start()
        for frame in stream:
            yield copy_frame(frame)
        logging.info("finised acquisition")


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nb-frames", default=10, type=int)
    parser.add_argument("-e", "--exposure-time", default=0.1, type=float)
    parser.add_argument(
        "--log-level",
        help="log level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
    )
    options = parser.parse_args(args)
    log_fmt = "%(levelname)s %(asctime)-15s %(name)s: %(message)s"
    logging.basicConfig(level=options.log_level.upper(), format=log_fmt)

    with dcam:
        with dcam[0] as camera:
            for i, frame in enumerate(
                gen_acquire(camera, options.exposure_time, options.nb_frames)
            ):
                logging.info(
                    f"Frame #{i+1}/{options.nb_frames} {frame.shape} {frame.dtype}"
                )


if __name__ == "__main__":
    main()
