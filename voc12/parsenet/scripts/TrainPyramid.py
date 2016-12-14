# coding: utf-8

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/exper')

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
NUM_LABELS = 21
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

# Specify model name to train
########### voc12 ###########
NET_ID = 'parsenet'
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

from NetCreatorPyramid import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'train_{}.prototxt'.format(TRAIN_SET))
deeplab_vgg16(
    NET_NAME,
    True,
    DATA_ROOT,
    os.path.join(LIST_DIR, '{}.txt'.format(TRAIN_SET)),
    NUM_LABELS,
    batch_size=20
)

from SolverCreator import create_solver
SOLVER_NAME = os.path.join(CONFIG_DIR, 'solver_{}.prototxt'.format(TRAIN_SET))
create_solver(
    SOLVER_NAME,
    NET_NAME,
    os.path.join(MODEL_DIR, 'train_pyramid'),
    snapshot=100,
    base_lr=1e-3,
    weight_decay=0.0005,
    momentum=0.9,
    max_iter=4000
)

import caffe
caffe.set_device(DEV_ID)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(SOLVER_NAME)
solver.net.copy_from(MODEL)

STEP = 20
from subprocess import call
for i in xrange(0, 4000, STEP):
    solver.step(STEP)
    """
    if i%(STEP*100) == 0:
        name = 'train_iter_{}'.format(i) if i !=0 else 'init'
        call('python Test.py {} {} {} {}'.format(DEV_ID, NET_ID, name, 'train'), shell=True)
        call('python Eval.py {} {} {} {}'.format(4, NET_ID, 'train', 'train'), shell=True)
    """
