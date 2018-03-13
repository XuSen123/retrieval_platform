# -----------------------------
# GENERATE IMAGE FEATURES AND STORE THEM
# Written by XuSenhai
# ------------------------------

import os
from config.config import cfg
import argparse
import cPickle
import cv2
import caffe
import numpy as np
import pdb

class model(object):
    '''Summaery of class here

    model class is utilized to initialize caffemodel, forward image 
    to get deep features
    
    Attributes:
        model: trained model name        
        prototxt: test model prototxt
        layername: the layer name features

    '''

    def __init__(self, model, caffemodel, prototxt, layername, gpu_id):
        ''' Init model here '''
        self.model_path = os.path.join(cfg.MODEL_DIR, model, caffemodel)
        self.prototxt_path = os.path.join(cfg.MODEL_DIR, model, prototxt)
        self.layernames = layername
 
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        #pdb.set_trace()
        self.net = caffe.Net(self.prototxt_path, self.model_path, caffe.TEST)

        print('{} model initialize Done!'.format(model))
    
    def image_preprocess(self, image):
        image = image.astype(np.float32, copy=False)
        image = image - cfg.PIXEL_MEANS
    
        if cfg.RESIZE:
            image = cv2.resize(image, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        
        image_ = np.transpose(image, (2, 0, 1))
        image_ = image_[np.newaxis, ...]

        return image_

    def forward(self, image, layername):
        image = self.image_preprocess(image) 
        #pdb.set_trace()
        self.net.blobs['data'].data[...] = image
        self.net.forward()
        feature = self.net.blobs[layername].data[0]

        return feature
