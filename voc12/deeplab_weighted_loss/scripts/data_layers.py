import caffe

import numpy as np
from PIL import Image

import scipy.io

import random

class VOCSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - root_folder: path to the root of the dataset
        - source: file containing image path & label path
        - mean: tuple of mean values to subtract
        - shuffle: load in random order (default: True)
        """
        # config
        params = eval(self.param_str)
        self.root = params['root_folder']
        self.source = params['source']
        self.mean = np.array(params['mean'])
        self.shuffle = params.get('shuffle', True)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = open(self.source, 'r').read().splitlines()
        self.idx = 0

        # randomization: seed and pick
        if self.shuffle:
            random.seed(None)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        (img, label) = self.indices[self.idx].split(' ')
        self.data = self.load_image(img)
        self.label = self.load_label(label)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.shuffle:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.root + idx)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.root + idx)
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

class VOCSegDataWeightsLayer(caffe.Layer):
    """
    Load (input image, label image, loss weights) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - root_folder: path to the root of the dataset
        - source: file containing image path & label path
        - mean: tuple of mean values to subtract
        - shuffle: load in random order (default: True)
        """
        # config
        params = eval(self.param_str)
        self.root = params['root_folder']
        self.source = params['source']
        self.mean = np.array(params['mean'])
        self.shuffle = params.get('shuffle', True)
        self.beta = params.get('beta', 0.99)

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define 3 tops: data ,label & weight.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.blobs.add_blob()

        # load indices for images and labels
        self.indices = open(self.source, 'r').read().splitlines()
        self.idx = 0

        # randomization: seed and pick
        if self.shuffle:
            random.seed(None)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image + weight pair
        (img, label, self.weight_path) = self.indices[self.idx].split(' ')
        self.data = self.load_image(img)
        self.label = self.load_label(label)
        self.weight = self.load_weight(self.weight_path)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

        # pick next input
        if self.shuffle:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        diff = np.abs(top[2].diff)
        min_diff, max_diff = np.min(diff), np.max(diff)
        diff = (diff - min_diff) / (max_diff - min_diff + 1e-5)
        self.weight = self.beta*self.weight + (1-self.beta)*diff
        scipy.io.savemat(self.root+self.weight_path, {'data': self.weight})

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.root + idx)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.root + idx)
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

    def load_weight(self, idx):
        """
        Load weight as 1 x height x width array of loss weights.
        The leading singleton dimension is required by the loss.
        """
        return np.squeeze(scipy.io.loadmat(self.root + idx)['data'])