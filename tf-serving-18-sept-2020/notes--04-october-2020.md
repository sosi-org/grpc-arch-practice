
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

`pip install -Uq grpcio==1.26.0`

```
ImportError: cannot import name 'keras' from 'tensorflow' (unknown location)
```

`pip install tensorflow --upgrade`
output:
```
    Found existing installation: tensorflow 2.3.0
    Uninstalling tensorflow-2.3.0:
      Successfully uninstalled tensorflow-2.3.0
Successfully installed tensorflow-2.3.1
```


`print(tf.version)`
```
<module 'tensorflow._api.v2.version' from '/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf2b/lib/python3.8/site-packages/tensorflow/_api/v2/version/__init__.py'>
```

If it shows 3.7.0, then tensorflow won't work since python 3.7 doesn't support tensorflow as of now.

In [13]: sys.version_info
Out[13]: sys.version_info(major=3, minor=8, micro=4, releaselevel='final', serial=0)
OOPS


`pip freeze > requirements.txt`

pip freeze > requirements.freeze.txt
WARNING: Could not generate requirement for distribution -rpcio 1.32.0 (/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf2b/lib/python3.8/site-packages): Parse error at "'-rpcio=='": Expected W:(abcd...)


pip install --upgrade tensorflow


# tensorflow==1.15 —The final version of TensorFlow 1.x.
https://www.tensorflow.org/install/pip


docker
https://www.tensorflow.org/install



Breakthrough: See:
https://hub.docker.com/r/tensorflow/serving
links to: [http://www.tensorflow.org/serving](serving) and [https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md](md)

[https://www.tensorflow.org/tfx/guide/serving](This) (is TFX?), links to related: [https://www.tensorflow.org/tfx/serving/architecture](Architecture Overview),
[https://www.tensorflow.org/tfx/serving/api_docs/cc/](Server API),
[https://www.tensorflow.org/tfx/serving/api_rest](REST Client API).


WORKS
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md
THIS IS THE SOLUTION ^

`ModelServer` = ?

https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
(But neesd keras)

What does it mean:
    "This will pull the latest TensorFlow Serving image with ModelServer installed."


outcome: these twp pages.
You need to know:
1. ho wo instalal keras (including python version)
2. What is the `ModelServer` claass in tf



Good search: install ModelServer

https://medium.com/@noone7791/how-to-install-tensorflow-serving-load-a-saved-tf-model-and-connect-it-to-a-rest-api-in-ubuntu-48e2a27b8c2a
Apr 9, 2018

saved_model_cli
?!

https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008
Nov 2, 2018

Finally this works
https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008


model:
https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz
