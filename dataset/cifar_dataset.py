# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:30:07 2019

@author: Administrator
"""

from .dataset import Dataset
from .downloader import Downloader
import pickle
import numpy as np
class CifarDataset(Dataset):
    def __init__(self, 
#                 downloader,
                 path_to_read="./dataset/cifar/"):
        super().__init__(path_to_read=path_to_read)
#        self.downloader = downloader
#        self.num_classes = 10
#        self.url_to_download = 'http://www.cs.toronto.edu/~kriz/cifar-{}-python.tar.gz'.format(self.num_classes)
        
    
        
        return
    
    def labels_to_one_hot(self, labels):
        """Convert 1D array of labels to one hot representation
        
        Args:
            labels: 1D numpy array
        """
        new_labels = np.zeros((labels.shape[0], self.num_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def labels_from_one_hot(self, labels):
        """Convert 2D array of labels to 1D class based representation
        
        Args:
            labels: 2D numpy array
        """
        return np.argmax(labels, axis=1)
    
    def loadDataset(self, one_hot=True):
        self.dataset = self.downloader.download_data_url()
    
    def read_cifar(self, filenames,one_hot):
        if self.num_classes == 10:
            labels_key = b'labels'
        elif self.num_classes == 100:
            labels_key = b'fine_labels'

        images_res = []
        labels_res = []
        for fname in filenames:
            with open(fname, 'rb') as f:
                images_and_labels = pickle.load(f, encoding='bytes')
            images = images_and_labels[b'data']
            images = images.reshape(-1, 3, 32, 32)
            images = images.swapaxes(1, 3).swapaxes(1, 2)#(-1,32,32,3)
            images_res.append(images)
            labels_res.append(images_and_labels[labels_key])
        images_res = np.vstack(images_res)
        labels_res = np.hstack(labels_res)
        if one_hot:
            labels_res = self.labels_to_one_hot(labels_res)
        return images_res, labels_res

import os
    
class Cifar10Dataset(CifarDataset):
    def __init__(self, path_to_read="./dataset/cifar/cifar10/"):
        super().__init__(path_to_read=path_to_read)
        self.downloader = Downloader(url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                                     path_to_save='./dataset/cifar/cifar10/')
        
        self.width=32
        self.height=32
        self.num_classes = 10
        self.num_channels = 3
        
        self.data_augmentation = False
        
        
        
        
    
    
    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path, 'cifar-10-batches-py')
        train_filenames = [
            os.path.join(
                sub_save_path,
                'data_batch_%d' % i) for i in range(1, 6)]
        test_filenames = [os.path.join(sub_save_path, 'test_batch')]
        return train_filenames, test_filenames
    
    def loadDataset(self, one_hot=True):
        super().loadDataset()
        train_filenames, test_filenames = self.get_filenames(save_path=self.path_to_read)
        self.xtrain, self.ytrain =self.read_cifar(filenames=train_filenames,one_hot=one_hot)
        self.xtest, self.ytest = self.read_cifar(filenames=test_filenames,one_hot=one_hot)
        self.xshape=(None,32,32,3)
        if one_hot:
            self.yshape=(None,self.num_classes)
        else:
            self.yshape=(None,1)
            
        self.size_train = self.xtrain.shape[0]
        self.size_test = self.xtest.shape[0]
        return
        
        

class Cifar100Dataset(CifarDataset):
    _n_classes = 100
    data_augmentation = False

    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path, 'cifar-100-python')
        train_filenames = [os.path.join(sub_save_path, 'train')]
        test_filenames = [os.path.join(sub_save_path, 'test')]
        return train_filenames, test_filenames
    
    
    