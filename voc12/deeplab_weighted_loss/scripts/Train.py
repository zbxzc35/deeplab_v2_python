# coding: utf-8

import sys

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
NET_ID = 'deeplab_weighted_loss'
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

sys.path.insert(0, os.path.join(EXP, NET_ID, 'scripts'))

from NetCreator import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'train_{}.prototxt'.format(TRAIN_SET))
deeplab_vgg16(
    NET_NAME,
    True,
    DATA_ROOT,
    os.path.join(LIST_DIR, '{}.txt'.format(TRAIN_SET)),
    NUM_LABELS
)

MAX_ITER = 20000
from SolverCreator import create_solver
SOLVER_NAME = os.path.join(CONFIG_DIR, 'solver_{}.prototxt'.format(TRAIN_SET))
create_solver(
    SOLVER_NAME,
    NET_NAME,
    os.path.join(MODEL_DIR, 'train'),
    max_iter=MAX_ITER,
    snapshot=2000
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(SOLVER_NAME)
solver.net.copy_from(MODEL)

# surgeries
import surgery
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for _ in xrange(MAX_ITER):
    solver.step(1)