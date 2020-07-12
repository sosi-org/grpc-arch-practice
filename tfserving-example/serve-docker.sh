
# ln -s /var/folders/5g/kz1p_241503bfrqndt8qy7640000gn/T/1 ./model_dir
#export MODEL_DIR=/var/folders/5g/kz1p_241503bfrqndt8qy7640000gn/T/1
# shared volume cannot be a symbolic (soft) link
export MODEL_DIR=$(pwd)/model_dir2
echo MODEL_DIR: $MODEL_DIR
ls -alt $MODEL_DIR

#docker run --rm -p 8501:8501 \
#    --mount type=bind,source=$(pwd),target=$(pwd) \
#    -e MODEL_DIR=$MODEL_DIR \
#    -e MODEL_NAME=saved_model_half_plus_three -t tensorflow/serving:latest

#  -v, --volume=[host-src:]container-dest[:<options>]:

#  tensorflow/serving:latest

echo 'docker---'
set -ex

#     --volume=$MODEL_DIR:/MODEL_DIR \
docker run --rm -p 8501:8501 \
    --volume=$MODEL_DIR:/MODEL_DIR \
    -e MODEL_DIR=/MODEL_DIR \
    -t \
    ubuntu:latest \
    echo 'hi'

