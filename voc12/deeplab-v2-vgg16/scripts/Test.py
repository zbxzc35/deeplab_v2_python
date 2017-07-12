# coding: utf-8

import os
import sys
import argparse
import setproctitle

from NetCreator import deeplab_vgg16

import caffe

from skimage import io

import numpy as np

from LargeImageHelper import split, concate, padding, unpadding

"""
    python Test.py --gpu [gpu_id]  --crop [crop_size] --step [step_size] --model [init model name] --feature [feature name] --testset [test set]
"""
parser = argparse.ArgumentParser('Test deeplab on cityscapes')
parser.add_argument('--gpu', dest='gpu_id', help='gpu id for testing', type=int, default=0)
parser.add_argument('--crop', dest='crop_size', help='crop size for testing', type=int, default=513)
parser.add_argument('--step', dest='step_size', help='step size when crop images for testing', type=int, default=512)
parser.add_argument('--model', dest='model', help='test model name', type=str, default='final')
parser.add_argument('--feature', dest='feature_name', help='prob feature name', type=str, default='fc8_interp')
parser.add_argument('--testset', dest='test_set', help='test set name', type=str, default='val')
args = parser.parse_args()

"""
    Global Env Setting
"""
NUM_LABELS = 34

EXP = '/data1/wuhuikai/DT_GAN/deeplab_v2/exper/cityscapes'
DATA_ROOT = '/data1/wuhuikai/DT_GAN/benchmark/cityscapes'

# Specify model name to train
########### cityscapes ###########
NET_ID = 'deeplab-v2-vgg16'
print 'Testing net {}/{}'.format(EXP, NET_ID)
setproctitle.setproctitle('cityscapes---' + NET_ID + '---TEST')

# Create dirs
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, NET_ID, 'model')

FEATURE_DIR = os.path.join(EXP, NET_ID, 'features', args.test_set, args.model+'___'+args.feature_name)
if not os.path.isdir(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)

"""
    Test specification (on val or test)
"""
TEST_LIST = os.path.join(EXP, NET_ID, 'list', '{}.txt'.format(args.test_set))

# Create testing prorotxt
NET_NAME = os.path.join(CONFIG_DIR, 'test_{}.prototxt'.format(args.test_set))
deeplab_vgg16(
    NET_NAME,
    False,
    DATA_ROOT,
    TEST_LIST,
    NUM_LABELS,
    batch_size=1,
    crop_size=args.crop_size
)

# Init caffe
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()

MODEL = os.path.join(MODEL_DIR, '{}.caffemodel'.format(args.model))
net = caffe.Net(NET_NAME, MODEL, caffe.TEST)

# Test
with open(TEST_LIST) as f:
    im_list = f.readlines()
TEST_ITER = len(im_list)

im_mean = np.array([104.008, 116.669, 122.675])

for i, im_pair in enumerate(im_list):
    print 'processing {}/{} ...'.format(i, TEST_ITER)

    im_path = os.path.join(DATA_ROOT, im_pair.strip().split(' ')[0])
    im = io.imread(im_path)[:,:,::-1] - im_mean

    im_group = split(im, args.crop_size, args.step_size)
    data = np.array([np.transpose(padding(p_im, args.crop_size), (2, 0, 1)) for p_im in im_group])
    
    net.blobs['data'].reshape(*data.shape)
    net.blobs['data'].data[...] = data
    net.forward(start='conv1_1')

    top = net.blobs[args.feature_name].data
    prob_group = [ unpadding(im_group[id], np.squeeze(prob), args.crop_size) for id, prob in enumerate(np.split(top, len(im_group), axis = 0)) ]
    prob = concate(prob_group, args.crop_size, args.step_size, im)
    
    im_name = os.path.basename(im_path)
    save_path = os.path.join(FEATURE_DIR, im_name)
    
    label = np.asarray(np.argmax(prob, 0), dtype=np.uint8)

    io.imsave(os.path.join(FEATURE_DIR, im_name), label)