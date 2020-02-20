# from:
# https://www.tensorflow.org/tfx/serving/api_rest

# docker pull tensorflow/serving:latest


MODEL_BASE_PATH=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata
#MODEL_BASE_PATH=$(pwd)/testdata
docker run --rm -p 8501:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    -e MODEL_BASE_PATH=$MODEL_BASE_PATH \
    -e MODEL_NAME=saved_model_half_plus_three -t tensorflow/serving:latest

# ...
# .... Exporting HTTP/REST API at:localhost:8501 ...

# curl -d '{"instances": [1.0,2.0,5.0]}' -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:predict


# curl -d '{"signature_name": "tensorflow/serving/regress", "examples": [{"x": 1.0}, {"x": 2.0}]}' \
#  -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:regress


# curl -d '{"instances": [1.0,2.0,5.0]}' -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:predict
# curl -d '{"signature_name": "tensorflow/serving/regress", "examples": [{"x": 1.0}, {"x": 2.0}]}'   -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:regress
# curl -i -d '{"instances": [1.0,5.0]}' -X POST http://localhost:8501/v1/models/half:predict



