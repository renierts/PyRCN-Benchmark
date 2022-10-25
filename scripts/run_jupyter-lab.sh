#!/bin/sh

# "PyRCN: A Toolbox for Exploration and Application of Reservoir Computing Networks"
#
# Copyright (C) 2022 Peter Steiner
# License: GPLv3

python3 -m venv .virtualenv
source .virtualenv/bin/activate
python3 -m pip install -r requirements.txt
jupyter-lab

deactivate
