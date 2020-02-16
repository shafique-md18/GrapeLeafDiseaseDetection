"""
Initiate/Load model
Get the local image
Predict the image on the loaded model
Print predictions
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from pathlib import Path
import numpy as np
import gc
import matplotlib.pyplot as plt
from PIL import Image
# import pandas as pd
# from PIL import Image
print(tf.__version__)


# def singleton(cls, *args, **kw):
#     instances = {}
#
#     def _singleton():
#         if cls not in instances:
#             instances[cls] = cls(*args, **kw)
#         return instances[cls]
#
#     return _singleton

def singleton(cls):
    instance = [None]
    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper

@singleton
class Predict:
    def __init__(self, path_to_model=None, shape=(256, 256)):
        """
        load model and print summary
        """
        (self.IMG_HEIGHT, self.IMG_WIDTH) = shape
        self.classes = ['Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                         'Grape___healthy'] # can be generated from tensorflow image_generator.class_indices
        CURR_DIR = Path('..')
        if path_to_model is None:
            path_to_model = CURR_DIR / 'model_new.h5'
        self.model = tf.keras.models.load_model(path_to_model)
        self.model.summary()

    def predict_image(self, image=None):
        try:
            if image is None:
                # no image path is provided
                raise TypeError
            else:
                im = Path(image)
                # wrong path provided
                if not im.exists():
                    raise FileNotFoundError
                else:
                    # file exists
                    self.img = Image.open(im)
                    self.img = self.img.convert('RGB')
                    self.im_array = np.asarray(self.img.resize((self.IMG_HEIGHT, self.IMG_WIDTH))) / 255.0
                    # shape of im_array is 150x150x3 but Conv2D expects 1x150x150x3 as input size
                    # as batch size is 1
                    self.im_array = np.expand_dims(self.im_array, axis=0)
                    self.predictions = self.model.predict(self.im_array)
                    print('Predictions = ', self.predictions)
                    print('Predicted class: ', self.classes[np.argmax(self.predictions)])
                    return (self.classes[np.argmax(self.predictions)], self.predictions)
        except TypeError:
            print('Usage: predict_image(image=\'path_to_image_file\')')


    def end_sess(self):
        gc.collect()

# p = Predict()
# p.predict_image('/home/shafique/PycharmProjects/LLD/LeafDiseaseDetector/dataset/testing_dataset/Grape___Esca_(Black_Measles)/Black_Measles9.JPG')

# GRAPH_PB_PATH = '../retrained_graph.pb'
# with tf.compat.v1.Session() as sess:
#     print("load graph")
#     with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#         graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
#     graph_nodes=[n for n in graph_def.node]
#     names = []
#     for t in graph_nodes:
#         names.append(t.name)
#     print(names)