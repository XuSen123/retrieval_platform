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
from src.utils import norm_cosin_distance
import copy

class Dataset(object):
    def __init__(self, dataset):
        self.data_path = os.path.join(cfg.DATA_DIR, dataset, 'image')
        self.dataset = dataset
        self.features = {}
        self.image_id = self.image_id()

    def get_image_list(self):
        #test_file = os.path.join(cfg.DATA_DIR, )
        image_iters = os.listdir(os.path.join(self.data_path))

        return image_iters
    
    def image_id(self):
        pkl_file = open(os.path.join(cfg.DATA_DIR, self.dataset, 'image_id.pkl'), 'rb')
        image_id = cPickle.load(pkl_file)

        return image_id
    
    def get_image_id(self, image_name):
        if image_name in self.image_id.keys(): 
            return self.image_id[image_name]
        else:
            print('Image: {}'.format(image_name))
            return None

    def add_feature(self, name, feature, id):
        if name not in self.features.keys():
            self.features[name] = {}
            self.features[name]['feature'] = feature 
            self.features[name]['id'] = id
    
    def store_feature(self, savepath):
        with open(savepath, 'wb') as f:
            cPickle.dump(self.features, f, cPickle.HIGHEST_PROTOCOL)
        print('Dump features Done!')
        print('Path: {}'.format(savepath))

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

    image_iters = ['0267313', '0034106']

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

    saveroot = os.path.join(cfg.DATA_DIR, cfg.FEATURE_DIR, cfg.MODEL)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    savepath = os.path.join(saveroot, cfg.SAVE_NAME)
    pdb.set_trace()
    dataset.store_feature(savepath)
