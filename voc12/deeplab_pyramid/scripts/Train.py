# coding: utf-8

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/exper')

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
NET_ID = 'deeplab_pyramid'
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
## Training #1 (on train_aug)

LIST_DIR = os.path.join(EXP, 'list')
TRAIN_SET = 'train_aug'
MODEL = os.path.join(MODEL_DIR, 'vgg.caffemodel')

print 'Training net {}/{}'.format(EXP, NET_ID)

sys.path.insert(0, os.path.join(EXP, NET_ID, 'scripts'))

from NetCreator import deeplab_pyramid_refine
NET_NAME = os.path.join(CONFIG_DIR, 'train_{}.prototxt'.format(TRAIN_SET))
deeplab_pyramid_refine(
    NET_NAME,
    True,
    DATA_ROOT,
    os.path.join(LIST_DIR, '{}.txt'.format(TRAIN_SET)),
    NUM_LABELS,
    batch_size=10
)

from SolverCreator import create_solver
SOLVER_NAME = os.path.join(CONFIG_DIR, 'solver_{}.prototxt'.format(TRAIN_SET))
create_solver(
    SOLVER_NAME,
    NET_NAME,
    os.path.join(MODEL_DIR, 'train'),
    snapshot=1000,
    base_lr=1e-3,
    weight_decay=0.0005,
    momentum=0.9,
    max_iter=16000
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(SOLVER_NAME)
solver.net.copy_from(MODEL)

STEP = 20
for i in xrange(0, 16000, STEP):
    solver.step(STEP)
