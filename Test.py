# coding: utf-8

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/deeplab-public-ver2/python/')

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
NET_ID = 'deeplab_v2_vgg16'
import setproctitle
setproctitle.setproctitle(NET_ID)
DEV_ID = int(sys.argv[1])

# Create dirs
import os
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, 'model', NET_ID)
LOG_DIR = os.path.join(EXP, NET_ID, 'log')
os.environ['GLOG_log_dir'] = LOG_DIR

## Run
## Test #1 specification (on val or test)
TEST_SET = 'val'
TEST_LIST = os.path.join(EXP, 'list', '{}.txt'.format(TEST_SET))

with open(TEST_LIST) as f:
    TEST_ITER = len(f.readlines())

MODEL = os.path.join(MODEL_DIR, 'test.caffemodel')

print 'Testing net {}/{}'.format(EXP, NET_ID)

FEATURE_DIR = os.path.join(EXP, 'features', NET_ID, TEST_SET, 'fc8')
if not os.path.isdir(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)

sys.path.insert(0, os.path.join(EXP, NET_ID, 'scripts'))
from NetCreator import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'test_{}.prototxt'.format(TEST_SET))
deeplab_vgg16(
    NET_NAME,
    False,
    DATA_ROOT,
    TEST_LIST,
    NUM_LABELS,
    1,
    FEATURE_DIR+'/',
    os.path.join(EXP, 'list', '{}_id.txt'.format(TEST_SET))
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

net = caffe.Net(NET_NAME, MODEL, caffe.TEST)
for i in xrange(TEST_ITER):
    print 'processing {}/{} ...'.format(i, TEST_ITER)
    net.forward()
