import os
import re
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

frozen_graph_filename = 'models/classify_image_graph_def.pb' # define with libraries 

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name = "prefix", 
            op_dict=None, 
            producer_op_list=None)
    return graph

graph = load_graph(frozen_graph_filename)

sess_inc = tf.Session(graph=graph)
next_to_last_tensor = sess_inc.graph.get_tensor_by_name('prefix/pool_3:0')

FEATURE_SIZE = 2048

def face_feature_extract(img) :
  cv2.imwrite('input_incp_v3.jpg', img)
  features = np.empty((1,FEATURE_SIZE))  
  image_data = gfile.FastGFile('input_incp_v3.jpg', 'rb').read()
  net_output = sess_inc.run(next_to_last_tensor,{'prefix/DecodeJpeg/contents:0': image_data})
  features = np.squeeze(net_output)
  return features
