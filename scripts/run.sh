#!/bin/sh

# "Template Repository for Research Papers with Python Code"
#
# Copyright (C) 2022 Peter Steiner
# License: GPLv3

python3 -m venv .virtualenv
source .virtualenv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/main.py --plot --export --serialize

deactivate
