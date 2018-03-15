#!/usr/bin/env python
# coding=utf-8

import os
import cv2
from src.utils import view_bar
from src.utils import Timer
from src.utils import eval
from src.utils import visualize
from src.dataset import Dataset
from config.config import cfg
import copy
import pdb

def main():
    retrieval_results_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, \
                                          cfg.MODEL, cfg.DISTANCE_METRIC)
    retrieval_lists = os.listdir(retrieval_results_root)
    
    visualize_root = os.path.join(cfg.VISUALIZE_DIR, cfg.DATASET, \
                                  cfg.MODEL, cfg.DISTANCE_METRIC)
    
    if not os.path.exists(visualize_root):
        os.makedirs(visualize_root)

    dataset = Dataset(cfg.DATASET)
    
    AP_boyun = 0.0
    AP_peking = 0.0

    for idx, retrieval_list in enumerate(retrieval_lists):
        #print('retrieval: {}'.format(retrieval_list))
        lines = open(os.path.join(retrieval_results_root, retrieval_list), 'r')
        lines = [iter.strip() for iter in lines]
        
        # evaluatw AP for each query
        ap_boyun, ap_peking = eval(lines, dataset) 
        AP_boyun += ap_boyun
        AP_peking += ap_peking
        
        # visualize results
        if cfg.VISUALIZE:
            visualize(lines, dataset, visualize_root)
        
        view_bar(idx + 1, len(retrieval_lists))

    mAP_boyun = AP_boyun / len(retrieval_lists)
    mAP_peking = AP_peking / len(retrieval_lists)

    print('mAP_boyun: {:.4f}, mAP_peking: {:.4f}'.format(mAP_boyun, mAP_peking))

if __name__ == '__main__':
    main()
