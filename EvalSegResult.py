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
