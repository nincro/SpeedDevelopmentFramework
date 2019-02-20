# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:17:27 2019

@author: Administrator
"""
import numpy as np
class Dataset(object):
    def __init__(self, path_to_read, downloader=None):
        self.path_to_read = path_to_read
        return
    
    def labelsToOneHot(self, labels):
        """Convert 1D array of labels to one hot representation
        
        Args:
            labels: 1D numpy array
        """
        new_labels = np.zeros((labels.shape[0], self.n_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def labelsFromOneHot(self, labels):
        """Convert 2D array of labels to 1D class based representation
        
        Args:
            labels: 2D numpy array
        """
        return np.argmax(labels, axis=1)
    
    def loadDataset(self):
        return NotImplementedError