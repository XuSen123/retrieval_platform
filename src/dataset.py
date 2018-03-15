#!/usr/bin/env python
# coding=utf-8
import os
from config.config import cfg
import cPickle
import pdb

class Dataset(object):
    def __init__(self, dataset):
        self.data_path = os.path.join(cfg.DATA_DIR, dataset, 'image')
        self.dataset = dataset
        self.features = {}
        self.image_id = self.image_id()
        self.id_image = self.id_image()

    def get_image_list(self):
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
    
    def id_image(self):
        pkl_file = open(os.path.join(cfg.DATA_DIR, self.dataset, 'id_image.pkl'), 'rb')
        id_image = cPickle.load(pkl_file)

        return id_image
    
    def get_id_length(self, image_id):
        if image_id in self.id_image.keys():
            return len(self.id_image[image_id])
        else:
            print('Image: {}'.format(image_id))
            return None

    def get_retrieval_list(self):
        retrieval_file_path = os.path.join(cfg.DATA_DIR, cfg.DATASET, \
                                          'train_test_split', 'test_list_800.txt')
        retrieval_lines = open(retrieval_file_path, 'r').readlines()
        
        retrieval_list_dict = {}

        for line in retrieval_lines:
            image_name, image_id = line.strip().split(' ')
            retrieval_list_dict[image_name] = image_id

        return retrieval_lines, retrieval_list_dict
        
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
