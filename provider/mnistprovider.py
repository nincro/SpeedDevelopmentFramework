# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:30:51 2019

@author: Administrator
"""
#import tensorflow.examples.tutorials.mnist.input_data as input_data
        
from .provider import Provider

class MnistProvider(Provider):
    def __init__(self, dataset, batch_size=100, shuffle=False):
#        self.dataset = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
#        
#        
#        self.xtrain = self.dataset.train.images
#        self.ytrain = self.dataset.train.labels
#        self.size_train = self.dataset.train.num_examples
#        
#        self.xtest = self.dataset.test.images
#        self.ytest = self.dataset.test.labels
#        self.size_test = self.dataset.test.num_examples
        super().__init__(dataset)
        self.st = 0
        self.batch_size = batch_size
        
        #此部分需要自己定义
        self.width = dataset.width
        self.height = dataset.height
        self.num_classes = dataset.num_classes
        self.num_channels = dataset.num_channels
        self.xshape = [None, dataset.width, dataset.height, dataset.num_channels]
        self.yshape = [None, dataset.num_classes]
        return
    
    def getNumClasses(self):
        return self.num_classes
    
    def getXShape(self):
        return self.xshape
    
    def getYShape(self):
        return self.yshape
    
    def loadTrainBatch(self):
        st = self.st
        ed = st+self.batch_size
        
        #左闭右开
        ed = ed if ed <= self.dataset.size_train else self.dataset.size_train 
        
        xtrain = self.dataset.xtrain[st:ed]
        ytrain = self.dataset.ytrain[st:ed]
#        print(xtrain.shape)
        xtrain = xtrain.reshape((-1,self.dataset.width, self.dataset.height, 1))
        ytrain = ytrain.reshape((-1,self.dataset.num_classes))
#        print(xtrain.shape)
        
        self.st=ed if ed!=self.dataset.size_train else 0
        flag_done = True if self.st==0 else False 
        
        return xtrain, ytrain, flag_done
        
    def loadTestBatch(self):
        
        
        return
    
    