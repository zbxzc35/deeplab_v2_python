import numpy as np

def split(im, crop_size, step):
    (height, width, _) = im.shape
    return [ im[h:h+crop_size, w:w+crop_size, :] for h in xrange(0, height, step) for w in xrange(0, width, step) ]

def concate(im_group, crop_size, step, im):
    (height, width, _) = im.shape
    (channel, _, _) = im_group[0].shape
    out = np.zeros([channel, height, width])

    id = 0
    for h in xrange(0, height, step):
        for w in xrange(0, width, step):
            out[:, h:h+crop_size, w:w+crop_size] += im_group[id]
            id += 1
    return out

def padding(im, crop_size):
    (height, width, _) = im.shape
    h_diff, w_diff = crop_size-height, crop_size-width
    h_pad, w_pad = h_diff/2, w_diff/2
    return np.pad(im, ((h_pad, h_diff-h_pad), (w_pad, w_diff-w_pad), (0, 0)), 'constant', constant_values=0)

def unpadding(im, out, crop_size):
    (height, width, _) = im.shape
    h_diff, w_diff = crop_size-height, crop_size-width
    h_pad, w_pad = h_diff/2, w_diff/2
    return np.copy(out[:, h_pad:h_pad+height, w_pad:w_pad+width])