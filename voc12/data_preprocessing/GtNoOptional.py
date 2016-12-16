# coding: utf-8

import os.path
import glob
from PIL import Image
import numpy as np

orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug'
save_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAugNoOptional'

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

annots = glob.glob(os.path.join(orig_folder, '*.png'))
total = len(annots)

for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = np.array(Image.open(annot), dtype=np.uint8)

    optinal_list = np.where(im == 255)
    for i, j in zip(*optinal_list):
        r = 1
        while True:
            label = np.unique(im[i-r:i+r, j-r:j+r])
            label = label[label != 255]
            label = label[label != 0]
            if len(label) > 0:
                im[i, j] = label[0]
                break
            r += 1
            
    Image.fromarray(im).save(os.path.join(save_folder, os.path.basename(annot)))