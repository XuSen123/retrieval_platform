#!/usr/bin/env python
# coding=utf-8

import os 
from config.config import cfg
from src.model import model
from src.utils import eval
from src.utils import visualize_dict
from src.dataset import Dataset
from src.utils import eval_dict
from src.utils import get_log_dir
import copy
import pdb
import cPickle
import numpy as np

def main():
    retrieval_results_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, \
                                          cfg.MODEL, cfg.DISTANCE_METRIC)
    retrieval_lists = os.listdir(retrieval_results_root)
    difference_dets_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, \
                                        cfg.MODEL, 'cosin_difference', 'dets')

    rerank_results_root = os.path.join(cfg.RESULT_DIR, cfg.DATASET, \
                                       cfg.MODEL, 'cosin_rerank')
    rerank_save_root = os.path.join(cfg.VISUALIZE_DIR, cfg.DATASET, \
                                    cfg.MODEL, 'rerank_'+cfg.DISTANCE_METRIC, \
                                    str(cfg.RANK_TOP_K))
    save_root = os.path.join(cfg.VISUALIZE_DIR, cfg.DATASET, \
                             cfg.MODEL, cfg.DISTANCE_METRIC, str(cfg.RANK_TOP_K))
    #analysis_root = os.path.join(cfg.VISUALIZE_DIR, cfg.DATASET, \
    #                             cfg.MODEL, cfg.DISTANCE_METRIC+'_analysis')
     
    analysis_root = get_log_dir()

    # mAP > r_mAP
    f_hard = open(os.path.join(analysis_root, 'cosin_hard.txt'), 'w')
    # mAP = r_mAP
    f_medium = open(os.path.join(analysis_root, 'cosin_medium.txt'), 'w')
    # mAP < r_mAP
    f_easy = open(os.path.join(analysis_root, 'cosin_easy.txt'), 'w')
    # all mAP and r_mAP records
    f_records = open(os.path.join(analysis_root, 'cosin_all.txt'), 'w')

    if not os.path.exists(rerank_save_root):
        os.makedirs(rerank_save_root)
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    dataset = Dataset(cfg.DATASET)

    AP_boyun = 0.0
    r_AP_boyun = 0.0
    AP_peking = 0.0
    r_AP_peking = 0.0
    
    print('theta1: {}, theta2: {}'.format(cfg.THETA1, cfg.THETA2))
    f_records.writelines('theta1: {}, theta2: {}\n'.format(cfg.THETA1, cfg.THETA2))

    #retrieval_lists = retrieval_lists[0:10]
    for idx, retrieval_list in enumerate(retrieval_lists):
        lines = open(os.path.join(retrieval_results_root, retrieval_list), 'r').readlines()
        lines = [iter.strip() for iter in lines]
        query_name = retrieval_list.split('.')[0]
        dets_pkl = os.path.join(difference_dets_root, query_name+'.pkl')
        f = open(dets_pkl, 'rb')
        dets = cPickle.load(f)
        
        retrieval_results = {}
        rerank_retrieval_results = {}
        
        # update image pair distance
        for i in range(1, cfg.RANK_TOP_K+1):
            ds_rate = 0.0

            line = lines[i]
            pair_name, distance = line.split(' ')
            query_name, candidate_name = pair_name.split('_')
            
            retrieval_results[pair_name] = distance
            
            # update image pair distance
            pair_dets = dets[pair_name]
            indexs = np.where(pair_dets[:, -1] > cfg.THRESHOLD)
            dets_number = len(indexs[0])
            pair_dets = pair_dets[indexs[0], :]

            if dets_number == 0:
                ds_rate = 0.0 
            else:
                ds_rate = np.max(pair_dets[:, -1])
            
            distance = float(distance)
            update_rate = np.power(distance, cfg.THETA1) * \
                          np.power((1-ds_rate), cfg.THETA2)

            rerank_retrieval_results[pair_name] = update_rate

        ap_boyun, ap_peking = eval_dict(retrieval_results, dataset)
        r_ap_boyun, r_ap_peking = eval_dict(rerank_retrieval_results, dataset)
        
        analysis_line = ' '.join(('query_name: {}'.format(query_name), \
                                  'ap_boyun: {:.4f}'.format(ap_boyun), \
                                  'r_ap_boyun: {:.4f}'.format(r_ap_boyun), \
                                  'ap_peking: {:.4f}'.format(ap_peking), \
                                  'r_ap_peking: {:.4f}'.format(r_ap_peking)))

        if ap_boyun > r_ap_boyun:
            f_hard.writelines(analysis_line+'\n')
        elif ap_boyun == r_ap_boyun:
            f_medium.writelines(analysis_line+'\n')
        elif ap_boyun < r_ap_boyun:
            f_easy.writelines(analysis_line+'\n')

        visualize_dict(retrieval_results, dataset, save_root)
        visualize_dict(rerank_retrieval_results, dataset, rerank_save_root)
 
        output_line = ' '.join(('query_name: {}'.format(query_name), \
                                'ap_boyun: {:.4f}'.format(ap_boyun), \
                                'r_ap_boyun: {:.4f}'.format(r_ap_boyun), \
                                'ap_peking: {:.4f}'.format(ap_peking), \
                                'r_ap_peking: {:.4f}'.format(r_ap_peking)))
        f_records.writelines(output_line + '\n')

        print('query_name: {},'.format(query_name)),
        print('ap_boyun: {:.4f}, r_ap_boyun: {:.4f},'.format(ap_boyun, r_ap_boyun)),
        print('ap_peking: {:.4f}, r_ap_peking:{:.4f}'.format(ap_peking, r_ap_peking))

        AP_boyun += ap_boyun 
        AP_peking += ap_peking
        r_AP_boyun += r_ap_boyun
        r_AP_peking += r_ap_peking
    
    mAP_boyun = AP_boyun / len(retrieval_lists)
    mAP_peking = AP_peking / len(retrieval_lists)
    r_mAP_boyun = r_AP_boyun / len(retrieval_lists)
    r_mAP_peking = r_AP_peking / len(retrieval_lists)
    
    f_hard.close()
    f_medium.close()
    f_easy.close()
    
    output_line = ' '.join(('mAP_boyun: {:.4f}'.format(mAP_boyun), \
                            'r_mAP_boyun: {:.4f}'.format(r_mAP_boyun), \
                            'mAP_peking: {:.4f}'.format(mAP_peking), \
                            'r_mAP_peking: {:.4f}'.format(r_mAP_peking)))
    f_records.writelines(output_line + '\n')
    
    f_records.close()
 
    print('mAP_boyun: {:.4f}, r_mAP_boyun: {:.4f},'.format(mAP_boyun, r_mAP_boyun)),
    print('mAP_peking: {:.4f}, r_mAP_peking: {:.4f}'.format(mAP_peking, r_mAP_peking))


if __name__ == '__main__':
    main()
