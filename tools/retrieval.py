#!/usr/bin/env python
# coding=utf-8

# ------------------------------
# RETRIEVAL FEATURES
# ------------------------------

import os
from config.config import cfg 
from src.model import model
from src.dataset import Dataset
from src.utils import view_bar
from src.utils import norm_L2_distance
from src.utils import norm_cosin_distance
import cPickle
import caffe
import numpy as np
import cv2
import pdb
import copy
import pdb

def main():
    image_root = os.path.join(cfg.DATA_DIR, cfg.DATASET, 'image')
    datasetname = cfg.DATASET
    dataset = Dataset(datasetname)
    query_lists, _= dataset.get_retrieval_list()
    feature_file = open(os.path.join(cfg.FEATURE_DIR, cfg.DATASET, \
                                     cfg.MODEL, cfg.DISTANCE_METRIC, \
                                     cfg.SAVE_NAME), 'rb')
    feature_dict = cPickle.load(feature_file)
    
    save_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, cfg.MODEL, cfg.DISTANCE_METRIC)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    net = model(cfg.MODEL, cfg.CAFFEMODEL, cfg.PROTOTXT, cfg.FEATURE_NAME[0], cfg.GPU_ID)    
     
    for idx, query_list in enumerate(query_lists):
        query_name, query_id = query_list.strip().split(' ')
        
        if query_name in feature_dict.keys():
            q_feature = copy.deepcopy(feature_dict[query_name]['feature'])
            q_id = copy.deepcopy(feature_dict[query_name]['id'])
        else:
            image_path = os.path.join(image_root, query_name + '.jpg')
            image = cv2.imread(image_path)
            q_feature = copy.deepcopy(net.forward(image, cfg.FEATURE_NAME[0]))
            q_id = dataset.get_image_id(query_name)
    
        retrieval_results = []

        for key in feature_dict.keys():
            p_feature = feature_dict[key]['feature']
            p_id = feature_dict[key]['id']
            
            if cfg.DISTANCE_METRIC == 'L2':
                tmp = norm_L2_distance(q_feature, p_feature)
            elif cfg.DISTANCE_METRIC == 'cosin':
                tmp = norm_cosin_distance(q_feature, p_feature)

            retrieval_results.append({'filename': key, 'distance': tmp, 'id': p_id})
            
        retrieval_results.sort(lambda x, y: cmp(y['distance'], x['distance']))
        #pdb.set_trace()

        f = open(os.path.join(save_root, query_name + '.txt'), 'w')
        
        # query image is also store in dataset
        for i in range(0, cfg.TOP_K + 1):
            filename = retrieval_results[i]['filename']
            distance = retrieval_results[i]['distance']
            f.writelines(str(query_name) + '_' + str(filename) + ' ' + str(distance) + '\n')
        f.close()

        view_bar(idx, len(query_lists))

if __name__ == '__main__':
    main()
