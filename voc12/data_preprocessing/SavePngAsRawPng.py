import os.path
import glob
from PIL import Image
import numpy as np

orig_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug_Visualization'
save_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012/SegmentationClassAug'

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

annots = glob.glob(os.path.join(orig_folder, '*.png'))
total = len(annots)

for idx, annot in enumerate(annots):
    print 'processing {} ({}) ...'.format(idx, total)
    im = np.array(Image.open(annot), dtype=np.uint8)
    Image.fromarray(im).save(os.path.join(save_folder, os.path.basename(annot)))
