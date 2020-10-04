#
# from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html
#
if True:
    import os
    import tarfile
    import urllib.request
    MODELS_DIR="/Users/a9858770/cs/ml--nn/grpc-arch-practice/tf-serving-18-sept-2020/models"

    # Download and extract model
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    # above hasa a problem. replacing it.
    PATH_TO_MODEL_TAR = './models/dlod.model.ta.gz'
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    print('aa')
    print('PATH_TO_CKPT', PATH_TO_CKPT)
    if not os.path.exists(PATH_TO_CKPT):
        print('bb')
        print('MODEL_DOWNLOAD_LINK', MODEL_DOWNLOAD_LINK)
        print('PATH_TO_MODEL_TAR', PATH_TO_MODEL_TAR)
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')

    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE = \
        'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')


