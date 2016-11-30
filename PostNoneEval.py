# coding: utf-8

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

debug = False

class_num = 21

VOC_root_folder = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit'
seg_root = os.path.join(VOC_root_folder, 'VOC2012')
gt_dir = os.path.join(seg_root, 'SegmentationClass');

EXP = '/home/wuhuikai/Segmentation/Deeplab_v2/exper'
post_folder = 'post_none'

testset = 'val'
output_mat_folder = os.path.join(EXP, 'voc12/features/deeplab_v2_vgg16/{}/fc8'.format(testset))
save_root_folder = os.path.join(output_mat_folder, post_folder)
print 'Saving to %s' % save_root_folder

seg_res_dir = os.path.join(save_root_folder, 'results/VOC2012')
save_result_folder = os.path.join(seg_res_dir, 'Segmentation', 'comp6_{}_cls'.format(testset));
if not os.path.isdir(save_result_folder):
    os.makedirs(save_result_folder)

import scipy.io
colormap = scipy.io.loadmat('pascal_seg_colormap.mat')['colormap']
palette = map(lambda x: map(lambda xx: int(255 * xx), x), colormap)

import glob
import png

annots = glob.glob(os.path.join(output_mat_folder, '*.mat'))
total = len(annots)

for idx, annot in enumerate(annots):
    if idx % 100 == 0:
        print 'processing %d (%d)...' % (idx, total)
    
    raw_result = np.transpose(np.squeeze(scipy.io.loadmat(annot)['data']), [1, 0, 2])
    img_fn = os.path.splitext(os.path.basename(annot))[0].replace('_blob_0', '')
    img = Image.open(os.path.join(seg_root, 'JPEGImages', img_fn+'.jpg'))
    img_col, img_row = img.size
    result = np.asarray(np.argmax(raw_result[0:img_row, 0:img_col, :], 2), dtype=np.uint8)
    
    if debug:
        gt = np.array(Image.open(os.path.join(gt_dir, img_fn+'.png')))
        plt.figure()
        plt.subplot(221), plt.imshow(img), plt.title('img'), plt.axis('off')
        plt.subplot(222), plt.imshow(colormap[gt[...]]), plt.title('gt'), plt.axis('off')
        plt.subplot(224), plt.imshow(colormap[result[...]]), plt.title('predict'), plt.axis('off')
        
    with open(os.path.join(save_result_folder, img_fn+'.png'), 'w') as f:
        png.Writer(size=(result.shape[1], result.shape[0]), palette = palette).write(f, result)  

from EvalSegResult import VOCevalseg
VOCevalseg(
    class_num,
    os.path.join(seg_root, 'ImageSets/Segmentation/{}.txt'.format(testset)),
    gt_dir,
    save_result_folder
)
