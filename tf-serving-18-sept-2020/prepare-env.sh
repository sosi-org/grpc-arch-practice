# run using `source`

if false; then

    none work:
    #assert folder PWD ~/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020
    #on "my MacOS"

    # SLOWNESS: 20 sec, DOWNLOADLOAD: 100MB
    #pip install -I keras
    echo '
    pip install --upgrade pip setuptools wheel
    pip install -I tensorflow
    pip install -I keras

    # AttributeError: module 'tensorflow' has no attribute 'keras'
    '

    pip install tensorflow-gpu==2.0.0-beta

    export GAP=~/cs/ml--nn/grpc-arch-practice
    source $GAP/tensorf2/bin/activate
    #cd tfserving-example


    # deactivate
fi


# clean start from scratcch


set -ex

if true; then

    # see (based on) source-tf1.sh

    # Installation on MacOS: (First time only)
    export GAP=~/cs/ml--nn/grpc-arch-practice
    export VENV=~/cs/ml--nn/grpc-arch-practice/tf2b
    cd $GAP

    # virtualenv --version
    #      # If error, install virsualenv . see https://www.tensorflow.org/install/pip
    #virtualenv -v --python=python3  ./tf2b

    # source /tf2b/bin/activate
    source $VENV/bin/activate

    # pip install tensorflow==2.3.0

    #pip install -Uq grpcio==1.26.0

    #pip install numpy
    # Requirement already satisfied: numpy in /Users/a9858770/cs/ml--nn/grpc-arch-practice/tf2b/lib/python3.7/site-packages (1.19.0)

    # ÷pip install scipy
    #* pip install imageio
    pip install  matplotlib
    #* pip install scikit-image
    #Unsure: cython PyHamcrest

fi
