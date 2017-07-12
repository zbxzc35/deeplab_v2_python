# coding: utf-8

import os
import sys
import argparse
import setproctitle

from NetCreator import deeplab_vgg16
from SolverCreator import create_solver

import caffe

"""
    python Train.py --gpu [gpu_id] --batchsize [batch_size] --maxiter [max iter] --trainset [training set name] --model [init model name]
"""
parser = argparse.ArgumentParser('Train deeplab on cityscapes')
parser.add_argument('--gpu', dest='gpu_id', help='gpu id for training', type=int, default=0)
parser.add_argument('--batchsize', dest='batch_size', help='batch_size', type=int, default=10)
parser.add_argument('--maxiter', dest='max_iter', help='max train iteration', type=int, default=20000)
parser.add_argument('--trainset', dest='train_set', help='training set name', type=str, default='train_target')
parser.add_argument('--model', dest='model', help='init model name', type=str, default='init')
parser.add_argument('--datafolder', dest='data_folder', help='data folder', type=str, default='')
parser.add_argument('--modelprefix', dest='model_prefix', help='prefix for saved model', type=str, default='deeplab-v2-vgg16')
args = parser.parse_args()

"""
    Global Env setting
"""
# Label num
NUM_LABELS = 34

# Base dirs
EXP = '/data1/wuhuikai/DT_GAN/deeplab_v2/exper/cityscapes'
DATA_ROOT = '/data1/wuhuikai/DT_GAN/benchmark/cityscapes/' + args.data_folder + '/'

# Specify model name to train
########### cityscapes ###########
NET_ID = 'deeplab-v2-vgg16'
print 'Training net {}/{}'.format(EXP, NET_ID)
setproctitle.setproctitle('cityscapes---' + NET_ID + '---TRAIN')

# Create dirs
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, NET_ID, 'model')
LIST_DIR = os.path.join(EXP, NET_ID,'list')
MODEL = os.path.join(MODEL_DIR, '{}.caffemodel'.format(args.model))

# Create trainning prototxt
NET_NAME = os.path.join(CONFIG_DIR, 'train_{}.prototxt'.format(args.train_set))
deeplab_vgg16(
    NET_NAME,
    True,
    DATA_ROOT,
    os.path.join(LIST_DIR, '{}.txt'.format(args.train_set)),
    NUM_LABELS,
    batch_size=args.batch_size
)

# Create solver prototxt
SOLVER_NAME = os.path.join(CONFIG_DIR, 'solver_{}.prototxt'.format(args.train_set))
create_solver(
    SOLVER_NAME,
    NET_NAME,
    os.path.join(MODEL_DIR, args.model_prefix),
    max_iter=args.max_iter
)

# Training
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(SOLVER_NAME)
solver.net.copy_from(MODEL)

for _ in xrange(args.max_iter):
    solver.step(1)