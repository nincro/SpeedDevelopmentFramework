# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:20:54 2019

@author: Administrator
"""
from .dataset import Dataset
import tensorflow.examples.tutorials.mnist.input_data as input_data
    
class MnistDataset(Dataset):
    def __init__(self, path_to_read="./dataset/data/MNIST_data/"):
        super().__init__(path_to_read=path_to_read)
        #此部分需要自己定义
        self.width = 28
        self.height = 28
        self.num_classes = 10
        self.num_channels = 1
        self.xshape = (None, self.width, self.height, self.num_channels)
        self.yshape = (None, self.num_classes)
        
        return

    def loadDataset(self, one_hot = False):
        self.dataset = input_data.read_data_sets(self.path_to_read, one_hot=one_hot)
        self.xtrain = self.dataset.train.images
        self.ytrain = self.dataset.train.labels
        self.size_train = self.dataset.train.num_examples
        
        self.xtest = self.dataset.test.images
        self.ytest = self.dataset.test.labels
        self.size_test = self.dataset.test.num_examples
        
        
    