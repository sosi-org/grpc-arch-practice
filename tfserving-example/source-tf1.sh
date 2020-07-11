
# . ../../neural-networks-sandbox/glyphnet/tensorf1/bin/activate
# source ./tensorf2/bin/activate


# Using https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

if false; then

    # Installation on MacOS: (First time only)
    cd grpc-arch-practice
    virtualenv --version
    # If error, install virsualenv . see https://www.tensorflow.org/install/pip
    virtualenv -v --python=python3  ./tensorf2
    source ./tensorf2/bin/activate

    pip install tensorflow
    # pip install tensorflow==2.2.0
    #▷ pip install tensorflow
    #Collecting tensorflow
    #  Downloading tensorflow-2.2.0-cp37-cp37m-macosx_10_11_x86_64.whl (175.3 MB)

    pip install -Uq grpcio==1.26.0

    pip install numpy
    # Requirement already satisfied: numpy in /Users/a9858770/cs/ml--nn/grpc-arch-practice/tensorf2/lib/python3.7/site-packages (1.19.0)

    # ÷pip install scipy
    #* pip install imageio
    pip install  matplotlib
    #* pip install scikit-image
    #Unsure: cython PyHamcrest

fi

# Run on MacOS
cd grpc-arch-practice/
source ./tensorf2/bin/activate
cd tfserving-example
#    * python ??.py
