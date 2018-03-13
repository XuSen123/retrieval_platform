#!/usr/bin/env python
# coding=utf-8

import os
import pdb
import sys
import time
import numpy as np

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
