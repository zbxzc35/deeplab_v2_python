# coding: utf-8

import sys
sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/deeplab-public-ver2/python/')

import caffe
from caffe import layers as L, params as P

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

"""
    proto_path: path for saving prototxt
    train: True if for training networks else for testing networks
    
    data_root: folder path for training images, e.g. "/home/wuhuikai/DataBase/VOC2012/VOCdevkit/VOC2012"
    source: file path for file containing a list of training images,
            e.g. "voc12/list/train_aug.txt"
    num_labels: class count to segment
    batch_size: training batch size

    prefix: ONLY for testing, folder for saving features,
            e.g. "voc12/features/deeplab_v2_vgg/val/fc8/"
    source_id: ONLY for testing, file containing a list of testing image ids,
               e.g. "voc12/list/val_id.txt"
"""
def deeplab_vgg16(proto_path, train, data_root, source, num_labels, batch_size=20, prefix=None, source_id=None):
    n = caffe.NetSpec()

    # Data Layer
    n.data, n.label, n.data_dim = L.ImageSegData(
        ntop=3,
        transform_param=dict(
            mirror=True if train else False,
            crop_size=321 if train else 513,
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
    
    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 1024, pad=12, dilation=12)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 1024, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score = L.Convolution(
        n.drop7, 
        num_output=num_labels, 
        kernel_size=1, 
        param=[
            dict(
                lr_mult=10, 
                decay_mult=1
            ), 
            dict(
                lr_mult=20, 
                decay_mult=0
            )
        ],
        weight_filler=dict(
            type="xavier"
        ),
        bias_filler=dict(
            type="constant",
            value=0
        )
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
            n.score, 
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
            n.score, 
            n.label_shrink,
            ignore_label=255
        )
        with open('/home/wuhuikai/Segmentation/Deeplab_v2/exper/train_proto_template') as f:
            template = f.read()
        proto = str(n.to_proto()) + template
    else:
        n.fc8_interp = L.Interp(
            n.score,
            zoom_factor=8
        )
        with open('/home/wuhuikai/Segmentation/Deeplab_v2/exper/test_proto_template') as f:
            template = f.read()
        proto = str(n.to_proto()) + template % (prefix, source_id)
    
    with open(proto_path, 'w') as f:
        f.write(proto)