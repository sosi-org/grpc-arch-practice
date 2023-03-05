#!/bin//bash
set -exu

# from: backend/ml/heuristic/pyinstall.bash
# from: https://github.com/Opteran/sohail-dev-notes/blob/main/localisation_heuristic/pyinstall.bash
# from: /home/sohail/opteran/sohail-dev-notes/gold_digger/pyinstall.bash
#  See https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
#  https://github.com/sosi-org/scientific-code/blob/256365e82b97fc529fc3626f312848e55eacc3c0/timescales-state/run-script.bash

# On Debian/Ubuntu systems, you need to install the python3-venv package using the following command: (You may need to use `sudo`)

# apt install python3.10-venv

VNAME="p3-for-me"

# rm -rf "$VNAME"

ls "$VNAME" || \
    python3 -m venv "$VNAME"
#--python=python3.5

source "./$VNAME/bin/activate"

pip install numpy
pip install matplotlib
# pip install cppyy
#pip install pygame
#pip install sympy
pip install torch
pip install torchvision

# dev-only:
pip install autopep8

python --version

# Python 3.10.6

# source "./$VNAME/bin/activate"
# python approach_1_demo.py
