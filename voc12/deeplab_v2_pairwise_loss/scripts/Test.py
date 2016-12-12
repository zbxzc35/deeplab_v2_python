# coding: utf-8

"""
    python Test.py [gpu_id] [exp_folder] [model_name] [feature_dir]
"""

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/deeplab-public-ver2/python/')

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
DEV_ID = int(sys.argv[1])
NET_ID = str(sys.argv[2])
import setproctitle
setproctitle.setproctitle(NET_ID)

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

MODEL = os.path.join(MODEL_DIR, '{}.caffemodel'.format(str(sys.argv[3])))

print 'Testing net {}/{}'.format(EXP, NET_ID)

FEATURE_DIR = os.path.join(EXP, 'features', NET_ID, TEST_SET, str(sys.argv[4]))
if not os.path.isdir(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)

PAIR_FEATURE_DIR = os.path.join(FEATURE_DIR, 'pairwise')
if not os.path.isdir(PAIR_FEATURE_DIR):
    os.makedirs(PAIR_FEATURE_DIR)

sys.path.insert(0, os.path.join(EXP, NET_ID, 'scripts'))
from NetCreator import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'test_{}.prototxt'.format(TEST_SET))
deeplab_vgg16(
    NET_NAME,
    False,
    DATA_ROOT,
    TEST_LIST,
    NUM_LABELS,
    batch_size=1,
    p_prefix=PAIR_FEATURE_DIR,
    prefix=FEATURE_DIR+'/',
    source_id=os.path.join(EXP, 'list', '{}_id.txt'.format(TEST_SET))
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

net = caffe.Net(NET_NAME, MODEL, caffe.TEST)
for i in xrange(TEST_ITER):
    print 'processing {}/{} ...'.format(i, TEST_ITER)
    net.forward()