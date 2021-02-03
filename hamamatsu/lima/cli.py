# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2021 Tiago Coutinho
# Distributed under the GPLv3 license. See LICENSE for more info.

import enum
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

from ..dcam import dcam, EIDString
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
    ctx.obj["camera"] = camera
    return Interface(camera)


def find_detectors(timeout=2.0):
    dcam.open()
    return list(dcam)


def detector_table(detectors):
    import beautifultable

    width = click.get_terminal_size()[0]
    table = beautifultable.BeautifulTable(maxwidth=width)

    table.columns.header = [
        "ID",
        "Vendor",
        "Model",
        "S/N",
        "Bus",
        "Version",
        "Driver",
        "Module",
        "API",
        "Series",
    ]
    for i, detector in enumerate(detectors):
        info = collections.defaultdict(str)
        info.update(detector.info)
        row = [
            i,
            info[EIDString.VENDOR],
            info[EIDString.MODEL],
            info[EIDString.CAMERAID],
            info[EIDString.BUS],
            info[EIDString.CAMERAVERSION],
            info[EIDString.DRIVERVERSION],
            info[EIDString.MODULEVERSION],
            info[EIDString.DCAMAPIVERSION],
            info[EIDString.CAMERA_SERIESNAME],
        ]
        table.rows.append(row)
    return table


def scan(timeout=2.0):
    detectors = find_detectors(timeout)
    return detector_table(detectors)


@hamamatsu.command("scan")
@click.option("--timeout", default=2.0)
@table_style
@max_width
def hamamatsu_scan(timeout, table_style, max_width):
    """show accessible hamamatsu devices"""
    table = scan(timeout)
    style = getattr(table, "STYLE_" + table_style.upper())
    table.set_style(style)
    table.maxwidth = max_width
    click.echo(table)


@hamamatsu.command("dump")
@table_style
@max_width
@click.pass_context
def hamamatsu_dump(ctx, table_style, max_width):
    """dump hamamatsu properties"""
    camera = ctx.obj["camera"]

    import beautifultable
    table = beautifultable.BeautifulTable(maxwidth=max_width)
    style = getattr(table, "STYLE_" + table_style.upper())
    table.set_style(style)
    table.columns.header = ["Name", "Value", "Unit", "DType"]

    def rep(x):
        if isinstance(x, enum.Enum):
            return x.name
        return x

    pmap = {}
    for prop in camera.values():
        row = prop["name"], rep(prop.value), rep(prop.unit), rep(prop.dtype).lstrip("TYPE_").lower()
        pmap[row[0]] = row

    for name in sorted(pmap):
        table.rows.append(pmap[name])

    click.echo(table)
