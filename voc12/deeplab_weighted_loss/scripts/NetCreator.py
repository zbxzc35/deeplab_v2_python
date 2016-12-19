# coding: utf-8

import sys

import caffe
from caffe import layers as L, params as P

# VGG 16-layer network convolutional finetuning
# Network modified to have smaller receptive field (128 pixels)
# nand smaller stride (8 pixels) when run in convolutional mode.
#
# In this model we also change max pooling size in the first 4 layers
# from 2 to 3 while retaining stride = 2
# which makes it easier to exactly align responses at different layers.
#
# For alignment to work, we set (we choose 32x so as to be able to evaluate
# the model for all different subsampling sizes):
# (1) input dimension equal to
# $n = 32 * k - 31$, e.g., 321 (for k = 11)
# Dimension after pooling w. subsampling:
# (16 * k - 15); (8 * k - 7); (4 * k - 3); (2 * k - 1); (k).
# For k = 11, these translate to
#           161;          81;          41;          21;  11
# 

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, dilation=1):
    conv = L.Convolution(
        bottom, 
        kernel_size=ks, 
        stride=stride,
        num_output=nout, 
        pad=pad,
        dilation=dilation,
        param=[
            dict(
                lr_mult=1, 
                decay_mult=1
            ), 
            dict(
                lr_mult=2, 
                decay_mult=0
            )
        ]
    )
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=3, stride=2, pad=1):
    return L.Pooling(
        bottom, 
        pool=P.Pooling.MAX, 
        kernel_size=ks, 
        stride=stride,
        pad = pad
    )

def drop_out(bottom):
    return L.Dropout(
        bottom,
        dropout_ratio=0.5,
        in_place=True
    )

"""
    proto_path: path for saving prototxt
    train: True if for training networks else for testing networks
    
    data_root: folder path for training images, e.g. "/home/wuhuikai/DataBase/VOC2012/VOCdevkit/VOC2012"
    source: file path for file containing a list of training images,
            e.g. "voc12/list/train_aug.txt"
    num_labels: class count to segment
    batch_size: training batch size
"""
def deeplab_vgg16(proto_path, train, data_root, source, num_labels):
    # name: "${NET_ID}"
    
    # Data Layer
    n = caffe.NetSpec()
    pydata_params = dict(
        root_folder=data_root,
        source=source,
        mean=(104.00699, 116.66877, 122.67892),
        shuffle=True if train else False
    )
    n.data, n.label = L.Python(
        module='data_layers',
        layer='VOCSegDataLayer',
        ntop=2, 
        param_str=str(pydata_params)
    )

    # ###################### DeepLab ####################
    
    # Pool 1
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2) # 3
    
    # Pool 2
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128) # 7
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128) # 11
    n.pool2 = max_pool(n.relu2_2) # 15
    
    # Pool 3
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256) # 23
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256) # 31
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256) # 39
    n.pool3 = max_pool(n.relu3_3)
    
    # Pool 4
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3, stride=1)
    
    # Pool 5
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, pad=2, dilation=2)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, pad=2, dilation=2)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, pad=2, dilation=2)
    n.pool5 = max_pool(n.relu5_3, stride=1)
    
    # ### hole = 6
    # fc 6
    n.fc6_1, n.relu6_1 = conv_relu(n.pool5, 1024, pad=6, dilation=6)
    n.drop6_1 = drop_out(n.relu6_1)
    
    # fc 7
    n.fc7_1, n.relu7_1 = conv_relu(n.drop6_1, 1024, ks=1, pad=0)
    n.drop7_1 = drop_out(n.relu7_1)
    
    # fc 8
    n.fc8_voc12_1 = L.Convolution(
        n.drop7_1,
        num_output=num_labels,
        kernel_size=1,
        weight_filler=dict(
            type='xavier'
        ),
        bias_filler=dict(
            type='constant',
            value=0
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ), 
            dict(
                lr_mult=20, 
                decay_mult=0
            )
        ]
    )
    
    # ### hole = 12
    # fc 6
    n.fc6_2, n.relu6_2 = conv_relu(n.pool5, 1024, pad=12, dilation=12)
    n.drop6_2 = drop_out(n.relu6_2)
    
    # fc 7
    n.fc7_2, n.relu7_2 = conv_relu(n.drop6_2, 1024, ks=1, pad=0)
    n.drop7_2 = drop_out(n.relu7_2)
    
    # fc 8
    n.fc8_voc12_2 = L.Convolution(
        n.drop7_2,
        num_output=num_labels,
        kernel_size=1,
        weight_filler=dict(
            type='xavier'
        ),
        bias_filler=dict(
            type='constant',
            value=0
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ), 
            dict(
                lr_mult=20, 
                decay_mult=0
            )
        ]
    )

    # ### hole = 18
    # fc 6
    n.fc6_3, n.relu6_3 = conv_relu(n.pool5, 1024, pad=18, dilation=18)
    n.drop6_3 = drop_out(n.relu6_3)
    
    # fc 7
    n.fc7_3, n.relu7_3 = conv_relu(n.drop6_3, 1024, ks=1, pad=0)
    n.drop7_3 = drop_out(n.relu7_3)
    
    # fc 8
    n.fc8_voc12_3 = L.Convolution(
        n.drop7_3,
        num_output=num_labels,
        kernel_size=1,
        weight_filler=dict(
            type='xavier'
        ),
        bias_filler=dict(
            type='constant',
            value=0
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ), 
            dict(
                lr_mult=20, 
                decay_mult=0
            )
        ]
    )

    # ### hole = 24
    # fc 6
    n.fc6_4, n.relu6_4 = conv_relu(n.pool5, 1024, pad=24, dilation=24)
    n.drop6_4 = drop_out(n.relu6_4)
    
    # fc 7
    n.fc7_4, n.relu7_4 = conv_relu(n.drop6_4, 1024, ks=1, pad=0)
    n.drop7_4 = drop_out(n.relu7_4)
    
    # fc 8
    n.fc8_voc12_4 = L.Convolution(
        n.drop7_4,
        num_output=num_labels,
        kernel_size=1,
        weight_filler=dict(
            type='xavier'
        ),
        bias_filler=dict(
            type='constant',
            value=0
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ), 
            dict(
                lr_mult=20, 
                decay_mult=0
            )
        ]
    )

    # ### SUM the four branches
    n.fc8_voc12 = L.Eltwise(
        n.fc8_voc12_1,
        n.fc8_voc12_2,
        n.fc8_voc12_3,
        n.fc8_voc12_4, 
        operation=P.Eltwise.SUM
    )
    
    n.up_fc8_voc12 = L.Deconvolution(
        n.fc8_voc12,
        convolution_param=dict(
            num_output=num_labels,
            kernel_size=16,
            stride=8,
            pad=4,
            group=num_labels
        ),
        param=[
            dict(
                lr_mult=1, 
                decay_mult=1
            ), 
            dict(
                lr_mult=2, 
                decay_mult=0
            )
        ]
    )
    n.score = L.Crop(n.up_fc8_voc12, n.data)
   
    # #################
    if train:
        # Loss
        n.loss = L.SoftmaxWithLoss(
            n.score, 
            n.label,
            include=dict(
                phase=0
            ),
            loss_param=dict(
                normalize=False,
                ignore_label=255
            )
        )
        
    with open(proto_path, 'w') as f:
        f.write(str(n.to_proto()))