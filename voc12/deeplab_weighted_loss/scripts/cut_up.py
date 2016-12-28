# coding: utf-8

import sys

import caffe
import numpy as np

def surgery(net, threshold):
    print 'threshold {} ...'.format(threshold)    
    
    total = 0
    cut = 0  
    for p in net.params:
        print 'processing {} ...'.format(p)
        for i in range(len(net.params[p])):
            data = net.params[p][i].data
            total += data.size
            cut += np.sum(np.abs(data) <= threshold)
            data[np.abs(data) <= threshold] = 0
    print '{}% weights was cut !!!'.format(cut/float(total)*100)

######################################################
import os

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'

NET_ID = 'deeplab_weighted_loss'
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, 'model', NET_ID)
MODEL = os.path.join(MODEL_DIR, 'train_weights_iter_12000.caffemodel')
SAVE_MODEL = os.path.join(MODEL_DIR, 'save.caffemodel')

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(os.path.join(CONFIG_DIR, 'train_train_aug.prototxt'), MODEL, caffe.TEST)
surgery(net, 0.002)
net.save(SAVE_MODEL)
