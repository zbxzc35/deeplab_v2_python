# coding: utf-8

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

class_name = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
    ]

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def VOCevalseg(class_num, test_list, gt_path, res_path):
    with open(test_list) as f:
        lines = f.readlines()
    
    confcounts = np.zeros((class_num, class_num))
    count = 0
    num_missing_img = 0
    
    total = len(lines)
    for idx, line in enumerate(lines):
        line = line.strip()
        if idx % 100 == 0:
            print 'test confusion: %d/%d' % (idx, total)
        
        im = np.array(Image.open(os.path.join(gt_path, line + '.png')), dtype=np.uint8)
        resfile = os.path.join(res_path, line + '.png')
        try:
            resim = np.array(Image.open(resfile), dtype=np.uint8)
        except Exception as e:
            num_missing_img += 1
            print 'Fail to read {}'.format(resfile)
            continue
        
        maxlabel = np.max(resim)
        if maxlabel >= class_num:
            print 'Results image ''%s'' has out of range value %d (the value should be < %d)' % (line, maxlabel, class_num)
        
        imshape = im.shape
        resshape = resim.shape
        if imshape != resshape:
            print 'Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.' % (line, resshape[0], resshape[1], imshape[0], imshape[1])
        
        confcounts += fast_hist(im.flatten(), resim.flatten(), class_num)
        count += np.sum(im < 255)
            
    if num_missing_img > 0:
        print 'WARNING: There are %d missing results!' % num_missing_img
    
    # confusion matrix - first index is true label, second is inferred label
    conf = 100 * confcounts / (1e-20 + np.sum(confcounts,1))[:, np.newaxis]
    rawcounts = confcounts;

    # Pixel Accuracy
    overall_acc = 100 * np.sum(np.diag(confcounts)) / np.sum(confcounts);
    print 'Percentage of pixels correctly labelled overall: %6.3f%%' % overall_acc

    # Class Accuracy
    class_acc = np.zeros(class_num)
    class_count = 0
    print 'Accuracy for each class (pixel accuracy)'
    
    for i in xrange(class_num):
        denom = np.sum(confcounts[i, :])
        if denom == 0:
            denom = 1
        class_acc[i] = 100 * confcounts[i, i] / denom;
        class_count += 1;
        print '  %14s: %6.3f%%' % (class_name[i], class_acc[i])
    
    print '-------------------------'
    avg_class_acc = np.sum(class_acc) / class_count
    print 'Mean Class Accuracy: %6.3f%%' % avg_class_acc
    
    # Pixel IOU
    accuracies = np.zeros(class_num)
    print 'Accuracy for each class (intersection/union measure)'
    
    real_class_count = 0;
    for j in xrange(class_num):
        gtj = np.sum(confcounts[j, :])
        resj = np.sum(confcounts[:, j])
        gtjresj = confcounts[j, j]
        # The accuracy is: true positive / (true positive + false positive + false negative) 
        # which is equivalent to the following percentage:
        denom = gtj + resj - gtjresj
        if denom == 0:
            denom = 1
        accuracies[j] = 100*gtjresj/denom
        real_class_count += 1
        print '  %14s: %6.3f%%' % (class_name[j], accuracies[j])

    avacc = np.sum(accuracies) / real_class_count
    print '-------------------------'
    print 'Average accuracy: %6.3f%%' % avacc


################################################################################################################
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

VOCevalseg(
    class_num,
    os.path.join(seg_root, 'ImageSets/Segmentation/{}.txt'.format(testset)),
    gt_dir,
    save_result_folder
)
