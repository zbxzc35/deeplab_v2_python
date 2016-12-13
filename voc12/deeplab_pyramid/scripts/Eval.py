# coding: utf-8

"""
    python Eval.py [num_worker] [exp_name] [feature_folder]
"""

import os
from PIL import Image
import numpy as np
import sys

class_num = 21

VOC_root_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit'
seg_root = os.path.join(VOC_root_folder, 'VOC2012')
gt_dir = os.path.join(seg_root, 'SegmentationClass');

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper'
post_folder = 'post_none'

testset = 'val'
output_pkl_folder = os.path.join(EXP, 'voc12/features/{}/{}/{}'.format(str(sys.argv[2]), testset, str(sys.argv[3])))
save_root_folder = os.path.join(output_pkl_folder, post_folder)
print 'Saving to %s' % save_root_folder

seg_res_dir = os.path.join(save_root_folder, 'results/VOC2012')
save_result_folder = os.path.join(seg_res_dir, 'Segmentation', 'comp6_{}_cls'.format(testset), str(sys.argv[3]));
if not os.path.isdir(save_result_folder):
    os.makedirs(save_result_folder)

import scipy.io
colormap = scipy.io.loadmat('pascal_seg_colormap.mat')['colormap']
palette = map(lambda x: map(lambda xx: int(255 * xx), x), colormap)

import glob
import png
import cPickle

annots = glob.glob(os.path.join(output_pkl_folder, '*.pkl'))
total = len(annots)

def pkl2png(idx, annot):
    if idx % 100 == 0:
        print 'processing %d (%d)...' % (idx, total)
    
    with open(annot) as f:
        raw_result = np.transpose(cPickle.load(f), [1, 2, 0])
        
    img_fn = os.path.splitext(os.path.basename(annot))[0].replace('_blob_0', '')
    img = Image.open(os.path.join(seg_root, 'JPEGImages', img_fn+'.jpg'))
    img_col, img_row = img.size
    result = np.asarray(np.argmax(raw_result[0:img_row, 0:img_col, :], 2), dtype=np.uint8)
    
    with open(os.path.join(save_result_folder, img_fn+'.png'), 'w') as f:
        png.Writer(size=(result.shape[1], result.shape[0]), palette = palette).write(f, result)  


from joblib import Parallel, delayed
Parallel(n_jobs=int(sys.argv[1]))(delayed(pkl2png)(idx, annot) for idx, annot in enumerate(annots))

sys.path.insert(0, '/home/wuhuikai/Segmentation/Deeplab_v2/exper')

from EvalSegResult import VOCevalseg
VOCevalseg(
    class_num,
    os.path.join(seg_root, 'ImageSets/Segmentation/{}.txt'.format(testset)),
    gt_dir,
    save_result_folder 
)