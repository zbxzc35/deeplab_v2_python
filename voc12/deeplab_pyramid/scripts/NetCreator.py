# coding: utf-8

import caffe
from caffe import layers as L, params as P

# VGG 16-layer network

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, dilation=1, lr=1):
    conv = L.Convolution(
        bottom, 
        kernel_size=ks, 
        stride=stride,
        num_output=nout, 
        pad=pad,
        dilation=dilation,
        param=[
            dict(
                lr_mult=lr*1, 
                decay_mult=1
            ), 
            dict(
                lr_mult=lr*2, 
                decay_mult=0
            )
        ]
    )
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2, pad=0):
    return L.Pooling(
        bottom, 
        pool=P.Pooling.MAX, 
        kernel_size=ks, 
        stride=stride,
        pad=pad
    )

def conv_bias(bottom, nout, ks=1, stride=1, pad=0):
    return L.Convolution(
        bottom, 
        kernel_size=ks, 
        pad=pad,
        stride=stride,
        num_output=nout, 
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

def conv_no_bias(bottom, nout, ks=1, stride=1, pad=0):
    return L.Convolution(
        bottom, 
        kernel_size=ks, 
        stride=stride,
        pad=pad,
        num_output=nout, 
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

def pyramid_pool(bottom, nout, ks, size):
    pool = L.Pooling(
        bottom, 
        pool=P.Pooling.AVE, 
        kernel_size=ks, 
        stride=ks
    )
    
    relu = L.ReLU( BN( conv_no_bias(pool, nout) ), in_place=True )

    return L.Interp(
        relu,
        height=size,
        width=size
    )

def drop_out(bottom, dropout_ratio=0.1):
    return L.Dropout(
        bottom,
        dropout_ratio=dropout_ratio,
        in_place=True
    )

def pyramid(bottom, nout, size):
    pyramid_pool1 = pyramid_pool(bottom, nout, size, size)
    pyramid_pool2 = pyramid_pool(bottom, nout, size/2, size)
    pyramid_pool3 = pyramid_pool(bottom, nout, size/3, size)
    pyramid_pool6 = pyramid_pool(bottom, nout, size/6, size)

    pyramid = L.Concat(
        bottom,
        pyramid_pool1,
        pyramid_pool2,
        pyramid_pool3,
        pyramid_pool6
    )

    pyramid_pool5 = drop_out(
        L.ReLU(
            BN(
                conv_no_bias(pyramid, nout, ks=3, pad=1)
            ), 
            in_place=True
        )
    )

    return pyramid_pool5

"""
    proto_path: path for saving prototxt
    train: True if for training networks else for testing networks
    
    data_root: folder path for training images, e.g. "/home/wuhuikai/DataBase/VOC2012/VOCdevkit/VOC2012"
    source: file path for file containing a list of training images,
            e.g. "voc12/list/train_aug.txt"
    num_labels: class count to segment
    batch_size: training batch size
"""
def deeplab_pyramid_refine(proto_path, train, data_root, source, num_labels, batch_size=10):
    # Data Layer
    n = caffe.NetSpec()
    n.data, n.label, n.data_dim = L.ImageSegData(
        ntop=3,
        transform_param=dict(
            mirror=True if train else False,
            crop_size=480,
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
    
    # ###################### FCN-32s ####################
    
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3, ks=3, stride=1, pad=1)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 1024, ks=3, pad=6, dilation=6)
    n.fc7, _ = conv_relu(n.relu6, 1024, ks=1, pad=0)
    
    # Pyramid #1
    n.top_relu = L.ReLU( BN(n.fc7), in_place=True )
    n.pyramid_pool5 = pyramid(n.top_relu, 256, 30)

    n.predict1 = L.Convolution(
        n.pyramid_pool5, 
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
    n.predict1_large = L.Interp(
        n.predict1,
        width=480,
        height=480
    )

    # Pyramid #2
    n.pyramid_pool5_up = L.Interp(
        n.pyramid_pool5,
        width = 60,
        height = 60
    )
    n.conv4 = L.ReLU( BN(conv_bias(n.relu4_3, 256) ), in_place=True )
    n.conv4_cat = L.Concat(
        n.pyramid_pool5_up,
        n.conv4
    )
    n.pyramid_pool4 = pyramid(n.conv4_cat, 128, 60)

    n.predict = L.Convolution(
        n.pyramid_pool4,
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

    n.predict_large = L.Interp(
        n.predict,
        width=480,
        height=480
    )
    # # #################
    if train:
        # Loss
        n.loss1 = L.SoftmaxWithLoss(
            n.predict1_large, 
            n.label,
            include=dict(
                phase=0
            ),
            loss_param=dict(
                ignore_label=255
            )
        )

        n.loss = L.SoftmaxWithLoss(
            n.predict_large, 
            n.label,
            include=dict(
                phase=0
            ),
            loss_param=dict(
                ignore_label=255
            ),
            loss_weight=0.1
        )

        # Accuracy
        n.accuracy = L.SegAccuracy(
            n.predict_large, 
            n.label,
            ignore_label=255
        )
        with open('/home/wuhuikai/Segmentation/Deeplab_v2/exper/train_proto_template') as f:
            template = f.read()
        proto = str(n.to_proto()) + template
    else:
        proto = str(n.to_proto())
    
    with open(proto_path, 'w') as f:
        f.write(proto)