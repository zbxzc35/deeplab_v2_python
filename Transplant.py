import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/deeplab-public-ver2/python/')

import caffe
import numpy as np

def transplant(new_net, net, suffix=''):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.
    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.
    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i

            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat[:new_net.params[p_new][i].data.size]

######################################################
import os

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12'

NET_ID = 'deeplab_v2_no_pooling'
CONFIG_DIR = os.path.join(EXP, NET_ID, 'config')
MODEL_DIR = os.path.join(EXP, 'model', NET_ID)
MODEL = os.path.join(MODEL_DIR, 'init.caffemodel')
SAVE_MODEL = os.path.join(MODEL_DIR, 'save.caffemodel')

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()
net = caffe.Net(os.path.join(CONFIG_DIR, 'transplate.prototxt'), caffe.TEST)
vgg_net = caffe.Net(os.path.join(CONFIG_DIR, 'init.prototxt'), MODEL, caffe.TEST)
transplant(net, vgg_net)
net.save(SAVE_MODEL)
