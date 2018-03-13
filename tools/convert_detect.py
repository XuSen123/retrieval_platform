#!/usr/bin/env python
# coding=utf-8

import os
import argparse
from config.config import cfg
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='convert_detect_list')
    parser.add_argument('--data', dest='data', help='dataset_name', \
                        default='VehicleID_V1.0', type=str)
    parser.add_argument('--model', dest='model', help='model_name', \
                        default='Peking', type=str)
    parser.add_argument('--metric', dest='metric', help='metric', \
                        default='L2', type=str)
    parser.add_argument('-top', dest='top', help='top', \
                        default=30, type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    retrieval_root = os.path.join(cfg.RESULT_DIR, args.data, \
                                  args.model, args.metric)

    iters = os.listdir(retrieval_root)
    
    saveroot = os.path.join(cfg.DATA_DIR, 'retrieval', 'detect_list')

    cfg.TOP_K = args.top

    for iter in iters:
        retrieval_lists = open(os.path.join(retrieval_root, iter), 'r').readlines()
        file = open(os.path.join(saveroot, iter), 'w')
        for i in range(1, cfg.TOP_K + 1):
            query_name = iter.split('.')[0]
            candidate_name = retrieval_lists[i].split(' ')[0]
            file.writelines(query_name + '_' + candidate_name + '\n')
        file.close()

    print('Convert detect list done!')
    
if __name__ == '__main__':
    main()
