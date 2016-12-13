# coding: utf-8

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

def drop_out(bottom, dropout_ratio=0.5):
    return L.Dropout(
        bottom,
        dropout_ratio=dropout_ratio,
        in_place=True
    )

def BN(bottom):
    return L.BN(
        bottom,
        momentum=0.95,
        frozen=True,
        in_place=True,
        slope_filler=dict(
            type='constant',
            value=1
        ),
        bias_filler=dict(
            type='constant',
            value=0
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=0
            ),
            dict(
                lr_mult=10, 
                decay_mult=0
            ),
            dict(
                lr_mult=0, 
                decay_mult=0
            ),
            dict(
                lr_mult=0, 
                decay_mult=0
            )
        ]
    )

def conv(bottom, ks, pad):
    return L.Convolution(
        bottom, 
        kernel_size=ks, 
        stride=1,
        pad=pad,
        num_output=256, 
        weight_filler=dict(
            type='msra'
        ),
        bias_term=False,
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            )
        ]
    )

def pyramid_pool(bottom, ks):
    pool = L.Pooling(
        bottom, 
        pool=P.Pooling.AVE, 
        kernel_size=ks, 
        stride=ks
    )
    
    relu = L.ReLU( BN( conv(pool, 1, 0) ), in_place=True )

    return L.Interp(
            relu,
            height=60,
            width=60
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
def deeplab_vgg16(proto_path, train, data_root, source, num_labels, batch_size=10):
    # Data Layer
    n = caffe.NetSpec()
    n.data, n.label, n.data_dim = L.ImageSegData(
        ntop=3,
        transform_param=dict(
            mirror=True if train else False,
            crop_size=473,
            mean_value=[104.008, 116.669, 122.675]
        ),
        image_data_param=dict(
            root_folder=data_root,
            source=source,
            batch_size=batch_size,
            shuffle=True if train else False,
            label_type=P.ImageData.PIXEL if train else P.ImageData.NONE
        )
    )
    
    # ###################### DeepLab ####################
    
    # Pool 1
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)
    
    # Pool 2
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)
    
    # Pool 3
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
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
    # fc 7
    n.fc7_1, n.relu7_1 = conv_relu(n.relu6_1, 1024, ks=1, pad=0)
    
    # ### hole = 12
    # fc 6
    n.fc6_2, n.relu6_2 = conv_relu(n.pool5, 1024, pad=12, dilation=12)
    # fc 7
    n.fc7_2, n.relu7_2 = conv_relu(n.relu6_2, 1024, ks=1, pad=0)
    
    # ### hole = 18
    # fc 6
    n.fc6_3, n.relu6_3 = conv_relu(n.pool5, 1024, pad=18, dilation=18)
    # fc 7
    n.fc7_3, n.relu7_3 = conv_relu(n.relu6_3, 1024, ks=1, pad=0)

    # ### hole = 24
    # fc 6
    n.fc6_4, n.relu6_4 = conv_relu(n.pool5, 1024, pad=24, dilation=24)
    # fc 7
    n.fc7_4, n.relu7_4 = conv_relu(n.relu6_4, 1024, ks=1, pad=0)

    # ### SUM the four branches
    n.fc8 = L.Eltwise(
        n.relu7_1,
        n.relu7_2,
        n.relu7_3,
        n.relu7_4, 
        operation=P.Eltwise.SUM
    )
    # ### Pyramid
    n.pyramid_pool1 = pyramid_pool(n.fc8, 60)
    n.pyramid_pool2 = pyramid_pool(n.fc8, 30)
    n.pyramid_pool3 = pyramid_pool(n.fc8, 20)
    n.pyramid_pool6 = pyramid_pool(n.fc8, 10)

    n.pyramid = L.Concat(
        n.fc8,
        n.pyramid_pool1,
        n.pyramid_pool2,
        n.pyramid_pool3,
        n.pyramid_pool6
    )

    n.fuse = drop_out(L.ReLU(BN(conv(n.pyramid, 3, 1)), in_place=True), 0.1)

    n.predict = L.Convolution(
        n.fuse, 
        kernel_size=1, 
        stride=1,
        num_output=num_labels, 
        weight_filler=dict(
            type='msra'
        ),
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ),
            dict(
                lr_mult=20, 
                decay_mult=1
            )
        ]
    )

    # #################
    if train:
        # Shrink Label
        n.label_shrink = L.Interp(
            n.label,
            shrink_factor=8,
            pad_beg=0,
            pad_end=0
        )

        # Loss
        n.loss = L.SoftmaxWithLoss(
            n.predict, 
            n.label_shrink,
            include=dict(
                phase=0
            ),
            loss_param=dict(
                ignore_label=255
            )
        )

        # Accuracy
        n.accuracy = L.SegAccuracy(
            n.predict, 
            n.label_shrink,
            ignore_label=255
        )
        with open('/home/wuhuikai/Segmentation/Deeplab_v2/exper/train_proto_template') as f:
            template = f.read()
        proto = str(n.to_proto()) + template
    else:
        n.fc8_interp = L.Interp(
            n.predict,
            zoom_factor=8
        )
        proto = str(n.to_proto())
    
    with open(proto_path, 'w') as f:
        f.write(proto)