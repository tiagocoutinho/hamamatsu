# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2021 Tiago Coutinho
# Distributed under the GPLv3 license. See LICENSE for more info.

from tango import DevState, Util
from tango.server import Device, device_property, attribute

from . import camera


class Hamamatsu(Device):

    camera_id = device_property(dtype=int, default_value=0)

    def init_device(self):
        super().init_device()
        self.ctrl = get_control()

    @property
    def mythen(self):
        return self.ctrl.hwInterface().detector

    def dev_state(self):
        status = self.mythen.status
        return DevState.RUNNING if status == "RUNNING" else DevState.ON


def get_tango_specific_class_n_device():
    return Hamamatsu


_HAMAMATSU = None


def get_control(camera_id=None):
    global _HAMAMATSU
    if _HAMAMATSU is None:
        if camera_id is None:
            # if there is no camera id use server instance
            camera_id = Util.instance().get_ds_inst_name()
        camera_id = int(camera_id)
        _HAMAMATSU = camera.get_control(camera_id)
    return _HAMAMATSU


def main():
    import Lima.Server.LimaCCDs

    Lima.Server.LimaCCDs.main()


if __name__ == "__main__":
    main()
