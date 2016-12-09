# coding: utf-8

"""
    python CRFPost_Eval.py [num_worker] [exp_folder] [feature_folder] [save_folder]
"""

import sys

TESTSET = 'val'
FEATURE_DIR = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/voc12/features/{}/{}/{}'.format(str(sys.argv[2]), TESTSET, str(sys.argv[3]))
SEG_ROOT = '/home/wuhuikai/Segmentation/Benchmark/Pascal/VOCdevkit/VOC2012'
IMAGE_DIR = SEG_ROOT + '/JPEGImages'
NUM_LABELS = 21

# specify the parameters
MAX_ITER=10

Bi_W=4
Bi_X_STD=65
Bi_Y_STD=65
Bi_R_STD=3
Bi_G_STD=3 
Bi_B_STD=3

POS_W=2
POS_X_STD=2
POS_Y_STD=2


# #######################################
# # MODIFY THE PATY FOR YOUR SETTING
# #######################################
import os
SAVE_DIR = os.path.join(
    FEATURE_DIR, 
    'post_densecrf_W{}_XStd{}_RStd{}_PosW{}_PosXStd{}_{}'.format(
        Bi_W,
        Bi_X_STD,
        Bi_R_STD,
        POS_W,
        POS_X_STD,
        str(sys.argv[4])
    )
)

print 'SAVE TO {}'.format(SAVE_DIR)

if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
import glob
import scipy.io
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
import png

annots = glob.glob(os.path.join(FEATURE_DIR, '*.mat'))
total = len(annots)
palette = map(lambda x: map(lambda xx: int(255 * xx), x), scipy.io.loadmat('pascal_seg_colormap.mat')['colormap'])

def crf_post(idx, annot):
    print 'processing %d (%d)...' % (idx, total)
    
    raw_result = np.transpose(np.squeeze(scipy.io.loadmat(annot)['data']), [2, 1, 0])
    img_fn = os.path.splitext(os.path.basename(annot))[0].replace('_blob_0', '')
    img = np.array(Image.open(os.path.join(IMAGE_DIR, img_fn+'.jpg')), dtype=np.uint8)
    img_row = img.shape[0]; img_col = img.shape[1]
    
    result = - np.reshape(raw_result[:, 0:img_row, 0:img_col].astype(np.float32), (NUM_LABELS, -1))
    
    d = dcrf.DenseCRF2D(img_col, img_row, NUM_LABELS)
    d.setUnaryEnergy(result)
    
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(POS_X_STD, POS_Y_STD), 
                          compat=POS_W)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(Bi_X_STD, Bi_Y_STD),
                           srgb=(Bi_R_STD, Bi_G_STD, Bi_B_STD), 
                           rgbim=img,
                           compat=Bi_W)
    
    Q = d.inference(MAX_ITER)
    MAP = np.argmax(Q, axis=0).reshape((img_row, img_col)).astype(np.uint8)
    
    with open(os.path.join(SAVE_DIR, img_fn+'.png'), 'w') as f:
        png.Writer(size=(MAP.shape[1], MAP.shape[0]), palette = palette).write(f, MAP)  

from joblib import Parallel, delayed
Parallel(n_jobs=int(sys.argv[1]))(delayed(crf_post)(idx, annot) for idx, annot in enumerate(annots))

from EvalSegResult import VOCevalseg
VOCevalseg(
    NUM_LABELS,
    os.path.join(SEG_ROOT, 'ImageSets/Segmentation/{}.txt'.format(TESTSET)),
    os.path.join(SEG_ROOT, 'SegmentationClass'),
    SAVE_DIR
)
