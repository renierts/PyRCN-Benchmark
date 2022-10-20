# "Template Repository for Research Papers with Python Code"
#
# Copyright (C) 2021 Peter Steiner
# License: GPLv3

python.exe -m venv .virtualenv

.\.virtualenv\Scripts\activate.ps1
python.exe -m pip install -r requirements.txt
python.exe .\src\main.py --fit_reservoirpy_esn

deactivate
