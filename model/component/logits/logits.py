# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:03:31 2019

@author: Administrator
"""
from tensorflow.contrib import slim



def myLogits(x, num_classes):
    
    y = slim.conv2d(inputs=x,
                    num_outputs=32,
                    kernel_size=[3, 3],
                    stride=[1, 1],
                    scope="conv1")
    y = slim.max_pool2d(inputs=y,
                        kernel_size=[2, 2],
                        scope="pool1")
    y = slim.flatten(inputs=y,
                     scope="flatten1")
    y = slim.fully_connected(inputs=y,
                             num_outputs=32,
                             scope="fc1")
    y = slim.fully_connected(inputs=y,
                             num_outputs=num_classes,
                             scope="fc2")
    return y

   

