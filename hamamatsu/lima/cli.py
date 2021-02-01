import struct
import asyncio
import contextlib
import collections
import urllib.parse

import click
import Lima.Core
from beautifultable import BeautifulTable
from limatb.cli import camera, url, table_style, max_width
from limatb.info import info_list
from limatb.network import get_subnet_addresses, get_host_by_addr

from ..dcam import dcam, DCAMIDSTR
from .camera import Interface


@camera(name="hamamatsu")
@click.option("camera_id", "-i", "--id", type=int)
@click.pass_context
def hamamatsu(ctx, camera_id):
    """Hamamatsu specific commands"""
    if camera_id is None:
        return
    dcam.open()
    camera = dcam[camera_id]
    camera.open()
    ctx.obj['camera'] = camera
    return Interface(camera)


def find_detectors(timeout=2.0):
    dcam.open()
    return list(dcam)


def detector_table(detectors):
    import beautifultable

    width = click.get_terminal_size()[0]
    table = beautifultable.BeautifulTable(maxwidth=width)

    table.columns.header = ["ID", "Vendor", "Model", "S/N", "Bus", "Version", "Driver", "Module", "API", "Series"]
    for i, detector in enumerate(detectors):
        with detector:
            info = collections.defaultdict(str)
            info.update(detector.get_info())
        row = [i, 
            info[DCAMIDSTR.VENDOR], 
            info[DCAMIDSTR.MODEL],
            info[DCAMIDSTR.CAMERAID],
            info[DCAMIDSTR.BUS],
            info[DCAMIDSTR.CAMERAVERSION],
            info[DCAMIDSTR.DRIVERVERSION],
            info[DCAMIDSTR.MODULEVERSION],
            info[DCAMIDSTR.DCAMAPIVERSION],
            info[DCAMIDSTR.CAMERA_SERIESNAME]
        ]
        table.rows.append(row)
    return table


def scan(timeout=2.0):
    detectors = find_detectors(timeout)
    return detector_table(detectors)


@hamamatsu.command("scan")
@click.option('--timeout', default=2.0)
@table_style
@max_width
def hamamatsu_scan(timeout, table_style, max_width):
    """show accessible hamamatsu devices"""
    table = scan(timeout)
    style = getattr(table, "STYLE_" + table_style.upper())
    table.set_style(style)
    table.maxwidth = max_width
    click.echo(table)
