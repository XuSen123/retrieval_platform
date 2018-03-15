#!/usr/bin/env python
# coding=utf-8

import os
import pdb
import sys
import time
import numpy as np
from config.config import cfg
import cv2
from time import strftime, localtime
import os.path as osp

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%% (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

def L2_distance(a, b):
    c = a - b
    c = c * c

    return c.sum()

def norm_L2_distance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    return L2_distance(a, b)

def norm_cosin_distance(a, b):
    upper = np.sum(np.multiply(a, b))
    bottom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))

    return upper / bottom

def eval(lines, datasets):
    ap = 0.0
    k = 1.0
    ap_boyun = 0.0
    ap_peking = 0.0

    for i in range(1, cfg.RANK_TOP_K+1):
        line = lines[i]
        pair_name, distance = line.split(' ')
        query_name, candidate_name = pair_name.split('_')
        query_id = datasets.get_image_id(query_name)
        candidate_id = datasets.get_image_id(candidate_name)
        
        query_length = datasets.get_id_length(query_id)

        if candidate_id == query_id:
            ap += k / (i)
            k += 1
    
    if k == 1:
        ap = 0.0
    else:
        ap_boyun = ap / (k - 1)
        ap_peking = ap / query_length

    return ap_boyun, ap_peking

def eval_dict(retrieval_results, datasets):
    ap = 0.0
    k = 1.0
    ap_boyun = 0.0
    ap_peking = 0.0
    
    sort_retrieval_results = sorted(retrieval_results.items(), key=lambda x:x[1], reverse=True)
    
    for i in range(len(sort_retrieval_results)):
        pair_name = sort_retrieval_results[i][0]
        distance = sort_retrieval_results[i][1]
        query_name, candidate_name = pair_name.split('_')
        query_id = datasets.get_image_id(query_name)
        candidate_id = datasets.get_image_id(candidate_name)

        query_length = datasets.get_id_length(query_id)

        if candidate_id == query_id:
            ap += k / (i+1)
            k += 1

    if k == 1:
        ap = 0.0
    else:
        ap_boyun = ap / (k-1)
        ap_peking = ap / query_length

    return ap_boyun, ap_peking

def draw(sample_image, concate_image, rows, cols, cell_width, cell_height, margin):
    concate_image[rows*(cell_height+margin)+5:(rows+1)*(cell_height+margin)-5, \
                  cols*(cell_width+margin)+5:(cols+1)*(cell_width+margin)-5, \
                  :] = sample_image
    return concate_image

def visualize_dict(retrieval_results, dataset, saveroot):
    rows = np.floor(cfg.VISUALIZE_TOP_K / cfg.VISUALIZE_COLS) + 1
    cell_width = 100
    cell_height = 100
    margin = 10
    
    sort_retrieval_results = sorted(retrieval_results.items(), key=lambda x:x[1], reverse=True)
 
    concate_image = np.zeros(((cell_height+margin) * rows, \
                              (cell_width+margin) * cfg.VISUALIZE_COLS, 3), \
                              dtype = np.float32)
    concate_image = concate_image + 255.0
    
    query_name = sort_retrieval_results[0][0].split('_')[0]
    query_image = cv2.imread(os.path.join(dataset.data_path, \
                                          query_name + '.jpg')) 
    query_image = cv2.resize(query_image, (cell_width, cell_height))
    query_id = dataset.get_image_id(query_name)
    query_length = dataset.get_id_length(query_id)
    concate_image = draw(query_image, concate_image, 0, \
                           0, cell_width, cell_height, margin)
    cv2.putText(concate_image,\
                'gt_number: {}'.format(query_length), (150,50), \
                cv2.FONT_HERSHEY_COMPLEX, 0.5, \
                (0,0,0), 2)
    cv2.putText(concate_image, \
                'red circle for right candidate', (300, 50), \
                cv2.FONT_HERSHEY_COMPLEX, 0.5, \
                (0,0,0), 2)
    cv2.putText(concate_image, \
                'blue circle for wrong candidate', (600, 50), \
                cv2.FONT_HERSHEY_COMPLEX, 0.5, \
                (0,0,0), 2)
    
    for i in range(cfg.VISUALIZE_TOP_K):
        candidate_name = sort_retrieval_results[i][0].split('_')[1]

        candidate_id = dataset.get_image_id(candidate_name)
        candidate_image = cv2.imread(os.path.join(dataset.data_path, \
                                                  candidate_name + '.jpg'))
        candidate_image = cv2.resize(candidate_image, (cell_width, cell_height))
    
        # right candidate with red circle
        # wrong candidate with blue circle
        if query_id == candidate_id:
            cv2.circle(candidate_image, (80, 20), \
                                         10, (0, 0, 255), 3)
        else:
            cv2.circle(candidate_image, (80, 20), \
                                         10, (255, 0, 0), 3)

        rows = np.floor(i / cfg.VISUALIZE_COLS)
        cols = i - rows * cfg.VISUALIZE_COLS
 
        concate_image = draw(candidate_image, concate_image, rows+1, \
                             cols, cell_width, cell_height, margin)
    
    concate_image = concate_image.astype(np.uint8)
    savepath = os.path.join(saveroot, \
                            '_'.join((query_name, str(cfg.VISUALIZE_TOP_K)))+\
                            '.jpg')

    cv2.imwrite(savepath, concate_image)

def visualize(lines, dataset, saveroot):
    rows = np.floor(cfg.VISUALIZE_TOP_K / cfg.VISUALIZE_COLS) + 1
    cell_width = 100
    cell_height = 100
    margin = 10

    concate_image = np.zeros(((cell_height+margin) * rows, \
                              (cell_width+margin) * cfg.VISUALIZE_COLS, 3), \
                              dtype = np.float32)
    concate_image = concate_image + 255.0

    query_name = lines[0].split(' ')[0].split('_')[0]
    query_image = cv2.imread(os.path.join(dataset.data_path, \
                                          query_name + '.jpg')) 
    query_image = cv2.resize(query_image, (cell_width, cell_height))
    query_id = dataset.get_image_id(query_name)
    query_length = dataset.get_id_length(query_id)
    concate_image = draw(query_image, concate_image, 0, \
                           0, cell_width, cell_height, margin)
    cv2.putText(concate_image,\
                'gt_number: {}'.format(query_length), (150,50), \
                cv2.FONT_HERSHEY_COMPLEX, 0.5, \
                (0,0,0), 2)

    for i in range(cfg.VISUALIZE_TOP_K):
        candidate_name = lines[i+1].split(' ')[0].split('_')[1]
        candidate_id = dataset.get_image_id(candidate_name)
        candidate_image = cv2.imread(os.path.join(dataset.data_path, \
                                                  candidate_name + '.jpg'))
        candidate_image = cv2.resize(candidate_image, (cell_width, cell_height))
        
        # right candidate with red circle
        # wrong candidate with blue circle
        if query_id == candidate_id:
            cv2.circle(candidate_image, (80, 20), \
                                         10, (255, 0, 0), 3)
        else:
            cv2.circle(candidate_image, (80, 20), \
                                         10, (0, 0, 255), 3)

        rows = np.floor(i / cfg.VISUALIZE_COLS)
        cols = i - rows * cfg.VISUALIZE_COLS

        concate_image = draw(candidate_image, concate_image, rows+1, \
                             cols, cell_width, cell_height, margin)
    
    concate_image = concate_image.astype(np.uint8)
    savepath = os.path.join(saveroot, \
                            '_'.join((query_name, str(cfg.VISUALIZE_TOP_K)))+\
                            '.jpg')
    cv2.imwrite(savepath, concate_image)

def get_log_dir():
    present_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    log_dir = osp.abspath(osp.join(cfg.VISUALIZE_DIR, cfg.DATASET, \
                                   cfg.MODEL, cfg.DISTANCE_METRIC+'_analysis', \
                                   present_time))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0.0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff
