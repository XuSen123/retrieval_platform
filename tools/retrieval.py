#!/usr/bin/env python
# coding=utf-8

# ------------------------------
# RETRIEVAL FEATURES
# ------------------------------

import os
from config.config import cfg 
from src.model import model
from src.utils import view_bar
import cPickle
import caffe
import numpy as np
import cv2
import pdb
import copy

def L2_distance(a, b):
    c = a - b
    c = c * c
    return c.sum()

def norm_L2_dictance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    return L2_distance(a, b)

def norm_cosin_distance(a, b):
    upper = np.sum(np.multiply(a, b))
    bottom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return upper / bottom
    #return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def main():
    query_file = os.path.join(cfg.DATA_DIR, 'retrieval', 'retrieval_list.txt')
    image_root = os.path.join(cfg.DATA_DIR, cfg.DATASET, 'image')
    query_lists = open(query_file, 'r').readlines()
    save_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, cfg.MODEL, cfg.DISTANCE_METRIC)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    net = model(cfg.MODEL, cfg.CAFFEMODEL, cfg.PROTOTXT, cfg.FEATURE_NAME[0], cfg.GPU_ID)    
    
    feature_file = open(os.path.join(cfg.DATA_DIR, cfg.DATASET, cfg.SAVE_NAME), 'rb')
    feature_dict = cPickle.load(feature_file)
    
    retrieval_results = []

    for idx, query_list in enumerate(query_lists):
        query_name, query_id = query_list.strip().split(' ')
        #query_name = '0226953'
        image_path = os.path.join(image_root, query_name + '.jpg')
        image = cv2.imread(image_path)
        q_feature = copy.deepcopy(net.forward(image, cfg.FEATURE_NAME[0]))
    
        retrieval_results = []

        for key in feature_dict.keys():
            p_feature = feature_dict[key]['feature']
            p_id = feature_dict[key]['id']
            #tmp = norm_L2_dictance(q_feature, p_feature)
            
            tmp = norm_cosin_distance(q_feature, p_feature)
            #pdb.set_trace()
            retrieval_results.append({'filename': key, 'distance': tmp, 'id': p_id})
            
        retrieval_results.sort(lambda x, y: cmp(x['distance'], y['distance']))

        f = open(os.path.join(save_root, query_name + '.txt'), 'w')
        
        query_name = retrieval_results[0]['filename']

        for i in range(0, cfg.TOP_K + 1):
            filename = retrieval_results[i]['filename']
            distance = retrieval_results[i]['distance']
            f.writelines(str(query_name) + '_' + str(filename) + ' ' + str(distance) + '\n')
        f.close()
        #pdb.set_trace() 
        view_bar(idx, len(query_lists))

if __name__ == '__main__':
    main()
