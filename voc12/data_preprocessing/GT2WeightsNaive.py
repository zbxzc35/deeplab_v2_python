# coding: utf-8

import os.path
import glob
from PIL import Image
import numpy as np

orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug'
save_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/WeightsAug'

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

annots = glob.glob(os.path.join(orig_folder, '*.png'))
total = len(annots)

import scipy.io
for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = np.array(Image.open(annot), dtype=np.uint8)
    
    result = np.ones_like(im)
    base_name = os.path.splitext(os.path.basename(annot))[0]
    scipy.io.savemat(os.path.join(save_folder, base_name+'.mat'), {'data': result})