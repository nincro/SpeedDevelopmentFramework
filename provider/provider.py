# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:21:53 2019

@author: Administrator
"""
import numpy as np
class Provider(object):
    def __init__(self, dataset):
        #dataset 一律以ndarray格式存储
        self.dataset = dataset
        return
    
    def loadTrainBatch(self):
        
        return NotImplementedError
    
    def loadTestBatch(self):
        return NotImplementedError
        
    
    
    