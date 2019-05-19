# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:51:27 2019

@author: Administrator
"""
from model.model import Model
from provider.mnistprovider import MnistProvider
from dataset.mnist_dataset import MnistDataset
def main():
    dataset = MnistDataset()
    dataset.loadDataset(one_hot=True)
    batch_size = 100
    provider = MnistProvider(dataset, batch_size=batch_size)
    
    model = Model(data_provider=provider)
    model.trainOneEpoch()
    
    return



if __name__ == "__main__":
    main()