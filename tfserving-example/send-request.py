
import numpy as np
import matplotlib.pyplot as plt

import random

def show(idx, title, test_images):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})

def sss(test_images, test_labels, class_names):
  rando = random.randint(0,len(test_images)-1)
  show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]), test_images)

## sss(?)

def rq(test_images, test_labels, class_names):
  import json
  data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
  print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


  # !pip install -q requests

  import requests
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
  predictions = json.loads(json_response.text)['predictions']

  tx = 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0])

  show(0, tx, test_images)
