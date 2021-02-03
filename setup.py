# -*- coding: utf-8 -*-
#
# This file is part of the hamamatsu project
#
# Copyright (c) 2021 Tiago Coutinho
# Distributed under the GPLv3 license. See LICENSE for more info.

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

extras = {
    'simulator': ['sinstruments>=1'],
    'lima': ['lima-toolbox>=1', 'beautifultable>=1', 'click>=7'],
}
extras["all"] = list(set.union(*(set(i) for i in extras.values())))

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Tiago Coutinho",
    author_email='coutinhotiago@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Python library to access Hamamatsu using DCAM or RemoteEX interface",
    install_requires=requirements,
    extras_require=extras,
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=['hamamatsu', 'remoteex', 'dcam', 'lima', 'simulator'],
    name='hamamatsu',
    packages=find_packages(include=['hamamatsu']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tiagocoutinho/hamamatsu',
    version='0.1.2',
    zip_safe=False,
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'hamamatsu-simulator = hamamatsu.simulator [simulator]',
            'hamamatsu-lima = hamamatsu.lima [Lima]',
        ],
        "Lima_camera": [
            "Hamamatsu=hamamatsu.lima.camera"
        ],
        "Lima_tango_camera": [
            "Hamamatsu=hamamatsu.lima.tango"
        ],
        "limatb.cli.camera": [
            'Hamamatsu=hamamatsu.lima.cli:hamamatsu [lima]'
        ],
        "limatb.cli.camera.scan": [
            "Hamamatsu=hamamatsu.lima.cli:scan [lima]"
        ],
    }
)
