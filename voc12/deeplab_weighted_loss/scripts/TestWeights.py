# coding: utf-8

"""
    python Test.py [gpu_id] [exp_folder] [model_name] [feature_dir]
"""

import sys

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
DEV_ID = int(sys.argv[1])
NET_ID = str(sys.argv[2])
import setproctitle
setproctitle.setproctitle(NET_ID+'-test')

# Create dirs
import os
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, 'model', NET_ID)

## Run
## Test #1 specification (on val or test)
TEST_SET = 'val'
TEST_LIST = os.path.join(EXP, 'list', '{}_weights.txt'.format(TEST_SET))

MODEL = os.path.join(MODEL_DIR, '{}.caffemodel'.format(str(sys.argv[3])))

print 'Testing net {}/{}'.format(EXP, NET_ID)

FEATURE_DIR = os.path.join(EXP, 'features', NET_ID, TEST_SET, str(sys.argv[4]))
if not os.path.isdir(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)

from NetCreatorWeights import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'test_{}.prototxt'.format(TEST_SET))
deeplab_vgg16(
    NET_NAME,
    False,
    DATA_ROOT,
    TEST_LIST,
    NUM_LABELS
)

source_id=os.path.join(EXP, 'list', '{}_id.txt'.format(TEST_SET))

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

net = caffe.Net(NET_NAME, MODEL, caffe.TEST)
with open(source_id) as f:
    lines = f.readlines()
    TEST_ITER = len(lines)

import scipy.io
import numpy as np
for i, line in enumerate(lines):
    print 'processing {}/{} ...'.format(i, TEST_ITER)
    net.forward()
    score = net.blobs['score'].data[0]
    score = np.transpose(score)
    scipy.io.savemat(os.path.join(FEATURE_DIR, line.strip()+'_blob_0.mat'), {'data': score})