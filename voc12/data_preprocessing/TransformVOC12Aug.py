# coding: utf-8

import os
import scipy.io
import glob
import png
from PIL import Image
import numpy as np

orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/benchmark_RELEASE/dataset/cls'
save_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug_Visualization'

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
    
colormap = map(lambda x: map(lambda xx: int(255 * xx), x), scipy.io.loadmat('pascal_seg_colormap.mat')['colormap'])

annots = glob.glob(os.path.join(orig_folder, '*.mat'))
total = len(annots)

for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = scipy.io.loadmat(annot)['GTcls']['Segmentation'][0, 0]
    with open(os.path.join(save_folder, os.path.splitext(os.path.basename(annot))[0] + '.png'), 'w') as f:
        png.Writer(size=(im.shape[1], im.shape[0]), palette = colormap).write(f, im)  

######################################################################################################################
        
orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClass'
annots = glob.glob(os.path.join(orig_folder, '*.png'))
total = len(annots)

for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = np.array(Image.open(annot), dtype=np.uint8)
    with open(os.path.join(save_folder, os.path.basename(annot)), 'w') as f:
        png.Writer(size=(im.shape[1], im.shape[0]), palette = colormap).write(f, im)  