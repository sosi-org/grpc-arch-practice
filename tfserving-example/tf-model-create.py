# TensorFlow and tf.keras

# Bsaed on: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

# print("Installing dependencies for Colab environment")
# !pip install -Uq grpcio==1.26.0

import tensorflow as tf
from tensorflow import keras

# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt
import os
import subprocess

#def tf_version():
print('TensorFlow version: {}'.format(tf.__version__))
# TensorFlow version: 2.2.0
# ----- tf version settled -----------------

from load_dataset import load_dataset

# Train and evaluate the model
def train_eval(train_images, train_labels, test_images, test_labels):
    model = keras.Sequential([
      keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3,
                          strides=2, activation='relu', name='Conv1'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])
    model.summary()

    testing = False
    epochs = 5

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))

    return model


def save_model(model):
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors,
    # and stored with the default serving key
    import tempfile

    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model:')
    #!ls -l {export_path}
    os.system(f'ls -l {export_path}')
    return export_path

    """
    assets
    saved_model.pb
    variables
    """

def create_model():
  train_images, train_labels, test_images, test_labels, class_names = load_dataset()
  model = train_eval(train_images, train_labels, test_images, test_labels)
  export_path = save_model(model)
  return export_path

# export_path = create_model()

export_path = '/var/folders/5g/kz1p_241503bfrqndt8qy7640000gn/T/1'
os.system(f'saved_model_cli show --dir {export_path} --all')
