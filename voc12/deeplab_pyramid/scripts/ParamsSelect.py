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
MODEL = os.path.join(MODEL_DIR, 'init.caffemodel')

print 'Training net {}/{}'.format(EXP, NET_ID)

sys.path.insert(0, os.path.join(EXP, NET_ID, 'scripts/TrainableModel'))

from NetCreator import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'train_{}.prototxt'.format(TRAIN_SET))
deeplab_vgg16(
    NET_NAME,
    True,
    DATA_ROOT,
    os.path.join(LIST_DIR, '{}.txt'.format(TRAIN_SET)),
    NUM_LABELS,
    batch_size=5
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

import numpy as np
for i in xrange(100):
    base_lr = 10 ** np.random.uniform(-1, -5)
    momentum = np.random.choice([0.5, 0.9, 0.95, 0.99])
    weight_decay = 10 ** np.random.uniform(-1, -6)

    from SolverCreator import create_solver
    SOLVER_NAME = os.path.join(CONFIG_DIR, 'solver_{}.prototxt'.format(TRAIN_SET))
    create_solver(
        SOLVER_NAME,
        NET_NAME,
        os.path.join(MODEL_DIR, 'train'),
        snapshot=2000,
        base_lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


    solver = caffe.SGDSolver(SOLVER_NAME)
    solver.net.copy_from(MODEL)

    solver.step(200)
