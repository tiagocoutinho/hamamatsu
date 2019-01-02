#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ 'gevent', 'PyYAML', 'enum34; python_version < "3.4"' ]

extras = {'simulator': ['sinstruments'],
          'lima': ['Lima'] }

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Tiago Coutinho",
    author_email='coutinhotiago@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python library to access Hamamatsu RemoteEX interface",
    install_requires=requirements,
    extras_require=extras,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['hamamatsu', 'remoteex', 'lima', 'simulator'],
    name='hamamatsu',
    packages=find_packages(include=['hamamatsu']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.com/tiagocoutinho/hamamatsu',
    version='0.1.0',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hamamatsu-simulator = hamamatsu.simulator [simulator]',
            'hamamatsu-lima = hamamatsu.lima [Lima]',
            ]
        },
)
