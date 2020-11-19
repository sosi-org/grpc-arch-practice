
# This is my attempt to rememerb things I did on 4 October 2020.
# However, was it based doocker??
# I need to turn my model into this format.
# now: 18 Nov 2020 23:02
# . ~/core/env-me-mine.sh

# from: /Users/a9858770/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020/prepare-env.sh
export GAP=~/cs/ml--nn/grpc-arch-practice
export VENV=~/cs/ml--nn/grpc-arch-practice/tf2b
cd $GAP
source $VENV/bin/activate

# cd ~/cs/ml--nn/grpc-arch-practice/nov-2020

if false; then

  cd ~/cs/ml--nn/grpc-arch-practice/nov-2020

  ln -s ~/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020/dl/tmp/resnet /tmp

  ls /tmp/resnet/resnet_client.py

  docker run -p 8501:8501 --name tfserving_resnet --mount type=bind,source=/tmp/resnet,target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving &

  python /tmp/resnet/resnet_client.py

fi
