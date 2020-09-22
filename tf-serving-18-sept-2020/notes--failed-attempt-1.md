## Yet another attempt
(Maybe 4th?) 18 Sept 2020


Base on this page:
	* Docker
	* Model from xor

### did
```sh
cd ~/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020

git clone --depth=1 https://github.com/tensorflow/serving

docker pull tensorflow/serving:latest

ls $(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata
# saved_model_half_plus_two_tf2_cpu

export MNAME=saved_model_half_plus_three

# set -ex
docker run --rm -p 8501:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    -e MODEL_BASE_PATH=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata \
    -e MODEL_NAME=$MNAME -t tensorflow/serving:latest




+ docker run --rm -p 8501:8501 --mount type=bind,source=/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020,target=/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020 -e MODEL_BASE_PATH=/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020/serving/tensorflow_serving/servables/tensorflow/testdata -e MODEL_NAME=saved_model_half_plus_three -t tensorflow/serving:latest



curl http://localhost:8501/v1/models/saved_model_half_plus_three
# some metadata
```

### regress call = ?

```bash

curl -d '{"instances": [1.0,2.0,5.0]}' -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:predict

# "predictions": [3.5, 4.0, 5.5

curl -d '{"signature_name": "tensorflow/serving/regress", "examples": [{"x": 1.0}, {"x": 2.0}]}' \
  -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:regress

# results": [3.5, 4.0]
```

Now, let's try an image:

```json
{
  "signature_name": "classify_objects",
  "examples": [
    {
      "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
      "caption": "seaside"
    },
    {
      "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
      "caption": "mountains"
    }
  ]
}
```

searching:

classify_objects rest

classify_objects rest tensorflow 2

https://stackoverflow.com/questions/57964394/how-to-a-make-a-model-ready-for-tensorflow-serving-rest-interface-with-a-base64


killing:
docker exec -it 16768cef0df8  bash

switching to another

searching:
tutorial object recognitiom in tensorflow webcam

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html
