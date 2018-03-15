#!/usr/bin/env python
# coding=utf-8

# -----------------------------
# GENERATE IMAGE FEATURES AND STORE THEM
# Written by XuSenhai
# ------------------------------

import os
from config.config import cfg
from src.model import model
import argparse
import cPickle
import cv2
import caffe
import numpy as np
import pdb
from src.utils import view_bar
from src.utils import Timer
from src.dataset import Dataset
import copy

if __name__ == '__main__':
    dataset = cfg.DATASET
    layername = cfg.FEATURE_NAME[0]
    modelname = cfg.MODEL
    caffemodel = cfg.CAFFEMODEL
    prototxt = cfg.PROTOTXT 
    gpu_id = cfg.GPU_ID
    
    print('model: {}, caffemodel: {}, prototxt: {}'.format(modelname, caffemodel, prototxt))

    net = model(modelname, caffemodel, prototxt, layername, gpu_id)

    dataset = Dataset(dataset)

    image_iters = dataset.get_image_list()
    
    #image_iters = image_iters[0:100]

    for idx, image_iter in enumerate(image_iters):
        image_name = image_iter.split('.')[0]

        image_id = dataset.get_image_id(image_name)

        if image_id:
            image_path = os.path.join(dataset.data_path, image_name + '.jpg')
            image = cv2.imread(image_path)
            net_time = Timer()
            
            net_time.tic()
            feature = copy.deepcopy(net.forward(image, layername))
            net_time.toc()

            dataset.add_feature(image_name, feature, image_id)
            
            view_bar(idx, len(image_iters))

            #if idx % 5000 == 0:
            #    print('{} image have done!'.format(idx))
            #print image_name

    saveroot = os.path.join(cfg.DATA_DIR, cfg.FEATURE_DIR, 
                            cfg.DATASET, cfg.MODEL)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    savepath = os.path.join(saveroot, cfg.SAVE_NAME)
    #pdb.set_trace()
    dataset.store_feature(savepath)
