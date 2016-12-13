# coding: utf-8

import os

import numpy as np

import caffe

from skimage.util import img_as_float
from skimage import io

import cPickle

def unpadding(im, out, crop_size):
    (height, width, _) = im.shape
    h_diff, w_diff = crop_size-height, crop_size-width
    h_pad, w_pad = h_diff/2, w_diff/2
    return np.copy(out[:, h_pad:h_pad+height, w_pad:w_pad+width])

def padding(im, crop_size):
    (height, width, _) = im.shape
    h_diff, w_diff = crop_size-height, crop_size-width
    h_pad, w_pad = h_diff/2, w_diff/2
    return np.pad(im, ((h_pad, h_diff-h_pad), (w_pad, w_diff-w_pad), (0, 0)), 'constant', constant_values=0)

def combine(ims, crop_size, im):
    (height, width, _) = im.shape
    (channel, _, _) = ims[0].shape
    out = np.zeros([channel, height, width])
    
    out[:, :crop_size, :crop_size] += ims[0]
    out[:, :crop_size, -crop_size:] += ims[1]
    out[:, -crop_size:, :crop_size] += ims[2]
    out[:, -crop_size:, -crop_size:] += ims[3]
    
    return out

def split_im(im, crop_size):
    (height, width, _) = im.shape
    return im[:crop_size, :crop_size, :], im[:crop_size, -crop_size:, :], im[-crop_size:, :crop_size, :], im[-crop_size:, -crop_size:, :]

def TestAll(img_list, data_root, save_dir, proto_path, model_path, gpu_id, crop_size=473):
    im_mean = np.array([104.008, 116.669, 122.675])
    
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    net = caffe.Net(proto_path, model_path, caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *(3, crop_size, crop_size))
    
    total = len(img_list)
    for idx, img_pair in enumerate(img_list):
        print 'processing {}/{} ...'.format(idx, total)
        
        img_name = img_pair.split(' ')[0].strip()
        img_path = '{}/{}'.format(data_root, img_name)
        im = io.imread(img_path)[:,:,::-1] - im_mean

        im_blocks = split_im(im, crop_size)
        results = []
        for im_block in im_blocks:
            im_pad = np.transpose(padding(im_block, crop_size), (2, 0, 1))
            net.blobs['data'].data[...] = im_pad
            net.forward(start='conv1_1')
            out = unpadding(im_block, net.blobs['fc8_interp'].data[0], crop_size)
            results.append(out)
        
        result = combine(results, crop_size, im)
        
        img_prename = os.path.splitext(os.path.basename(img_name))[0]
        with open('{}/{}.pkl'.format(save_dir, img_prename), 'w') as f:
            cPickle.dump(result, f)


"""
    python Test.py [gpu_id] [exp_name] [model_name]
"""

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/PSPNet/python')

import os

NUM_LABELS = 21
EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'
DATA_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'

DEV_ID = int(sys.argv[1])
NET_ID = str(sys.argv[2])
import setproctitle
setproctitle.setproctitle(NET_ID+'_test')

CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, 'model', NET_ID)

TEST_SET = 'val'
TEST_LIST = os.path.join(EXP, 'list', '{}.txt'.format(TEST_SET))
with open(TEST_LIST) as f:
    TEST_IMG = f.readlines()

MODEL = os.path.join(MODEL_DIR, '{}.caffemodel'.format(str(sys.argv[3])))
FEATURE_DIR = os.path.join(EXP, 'features', NET_ID, TEST_SET, str(sys.argv[3]))
if not os.path.isdir(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)
    
from NetCreator import deeplab_vgg16
NET_NAME = os.path.join(CONFIG_DIR, 'test_{}.prototxt'.format(TEST_SET))
deeplab_vgg16(
    NET_NAME,
    False,
    DATA_ROOT,
    TEST_LIST,
    NUM_LABELS,
    batch_size=1
)

TestAll(TEST_IMG, DATA_ROOT, FEATURE_DIR, NET_NAME, MODEL, DEV_ID)