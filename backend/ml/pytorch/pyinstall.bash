#!/bin//bash
set -exu

# from: backend/ml/heuristic/pyinstall.bash
# from: https://github.com/Opteran/sohail-dev-notes/blob/main/localisation_heuristic/pyinstall.bash
# from: /home/sohail/opteran/sohail-dev-notes/gold_digger/pyinstall.bash
#  See https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
#  https://github.com/sosi-org/scientific-code/blob/256365e82b97fc529fc3626f312848e55eacc3c0/timescales-state/run-script.bash

# On Debian/Ubuntu systems, you need to install the python3-venv package using the following command: (You may need to use `sudo`)

# apt install python3.10-venv


#The virtual environment was not created successfully because ensurepip is not available.  On Debian/Ubuntu systems, you need to install the python3-venv package using the following command.
#
#    apt install python3.8-venv



#apt install python3.
# python3.8           python3.8-dev       python3.8-examples  python3.8-minimal   python3.9           python3.9-dev       python3.9-examples  python3.9-minimal python3.8-dbg       python3.8-doc       python3.8-full      python3.8-venv      python3.9-dbg       python3.9-doc       python3.9-full      python3.9-venv
# Only 3.8 works:
#   sudo apt install python3.8-venv
# needs sudo

# It is a path, not a name. again and again in bash.
VNAME_DEFAULT="./p3-for-me"
export VNAME=${1:-$VNAME_DEFAULT}

echo "The venv is at: $VNAME/bin/activate"

# rm -rf "$VNAME"

ls "$VNAME" || \
    python3 -m venv --upgrade --copies "$VNAME"
# https://docs.python.org/3/library/venv.html#creating-virtual-environments
# --system-site-packages
# cannot work:
# --upgrade-deps

#--python=python3.5

echo "use: ----> source $VNAME/bin/activate"
echo
echo
source "$VNAME/bin/activate"
echo

pip install numpy
pip install matplotlib
# pip install cppyy
#pip install pygame
#pip install sympy
pip install torch
pip install torchvision

# Why does it get torch each time?
# todo upgeade that python

# dev-only:
pip install autopep8

python --version

# Python 3.10.6

# source "./$VNAME/bin/activate"
# python approach_1_demo.py
