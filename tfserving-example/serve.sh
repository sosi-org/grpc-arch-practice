# tensorflow-model-server

echo "SERVER PROCESS:"
# This is the same as you would do from your command line, but without the [arch=amd64], and no sudo
# You would instead do:
# echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
apt update

# install the tf serving
apt-get install tensorflow-model-server

# os.environ["MODEL_DIR"] = MODEL_DIR
# export MODEL_DIR=/var/folders/5g/kz1p_241503bfrqndt8qy7640000gn/T/1

# export MODEL_DIR=/MODEL_DIR

echo "MODEL_DIR: $MODEL_DIR"

# %%bash --bg
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="$MODEL_DIR" >server.log 2>&1

# tail server.log
