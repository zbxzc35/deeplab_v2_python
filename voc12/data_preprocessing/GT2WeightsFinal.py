# coding: utf-8

import os.path
import glob
from PIL import Image
import numpy as np

orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug'
save_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/WeightsAugFinal'

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

annots = glob.glob(os.path.join(orig_folder, '*.png'))
total = len(annots)

import scipy.io
for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = np.array(Image.open(annot), dtype=np.uint8)
    
    # Result
    result = np.zeros(im.shape) 
    labels = np.unique(im)
    count = float(im.size) / len(labels)
    for label in labels:
        label_count = np.sum(im == label)
        result[im == label] = count/label_count
    r_min, r_max = np.min(result), np.max(result)
    result = (result - r_min) / (r_max - r_min) + 1.0
    result[im == 255] = 1.0
    
    # Row weight
    (W, H) = im.shape
    w_weight = np.zeros(im.shape)
    for w in xrange(W):
        start = 0
        for h in xrange(H+1):
            if h == H or im[w, h] != im[w, start]:
                w_weight[w, start:h] = h-start
                start = h   
    # Col weight
    h_weight = np.zeros(im.shape)
    for h in xrange(H):
        start = 0
        for w in xrange(W+1):
            if w == W or im[w, h] != im[start, h]:
                h_weight[start:w, h] = w-start
                start = w
    # Weight
    weight = np.minimum(w_weight, h_weight)
    weight[im == 255] = min(W, H)
    weight = 1.0 / weight + 1.0

    # Result
    result *= weight
    
    base_name = os.path.splitext(os.path.basename(annot))[0]
    scipy.io.savemat(os.path.join(save_folder, base_name+'.mat'), {'data': result})
