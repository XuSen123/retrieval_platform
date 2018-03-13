#!/usr/bin/env python
# coding=utf-8

# -----------------------
# retrieval system written by XuSenhai
# -----------------------

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

__C.DATA_DIR = osp.join(__C.ROOT_DIR, 'data')

__C.FEATURE_DIR = osp.join(__C.ROOT_DIR, 'feature')

__C.MODEL_DIR = osp.join(__C.ROOT_DIR, 'models')

__C.RESULT_DIR = osp.join(__C.ROOT_DIR, 'results')

__C.PIXEL_MEANS = np.array([[[104.0, 117.0, 123.0]]])


# Setting distance metric e.g: 'L2' or 'cosin'
__C.DISTANCE_METRIC = 'cosin'

# Setting image process
__C.RESIZE = True

# Setting top-K list
__C.TOP_K = 100

# Setting image width
__C.IMAGE_WIDTH = 224

# Setting image height
__C.IMAGE_HEIGHT = 224

# Setting gpu id
__C.GPU_ID = 0

# Setting save_name
__C.SAVE_NAME = 'softmax_features.pkl'

# Setting data
__C.DATASET = 'VehicleID_V1.0'

# Setting model
__C.MODEL = 'softmax'

# Setting caffe models
__C.CAFFEMODEL = 'carReID_softmax.caffemodel'

# Setting prototxt
__C.PROTOTXT = 'deploy.prototxt'

# Setting feature name
__C.FEATURE_NAME = ['global_pool']
