# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:03:31 2019

@author: Administrator
"""
from .net import Net
from tensorflow.contrib import slim
import tensorflow as tf


class EasyNet(Net):
    
    
    
#    @property
#    def logits(self):
#        y = slim.conv2d(inputs=self.x_holder,
#                        num_outputs=32,
#                        kernel_size=[3,3],
#                        stride=[1,1],
#                        scope="conv1")
#        y = slim.max_pool2d(inputs=y,
#                            kernel_size=[2,2],
#                            scope="pool1")
#        y = slim.flatten(inputs=y,
#                         scope="flatten1")
#        y = slim.fully_connected(inputs=y,
#                                 num_outputs=32,
#                                 scope="fc1")
#        y = slim.fully_connected(inputs=y,
#                                 num_outputs=self.provider.num_classes,
#                                 scope="fc2")
#        
#        return y
#    
#    @property
#    def total_loss(self):
#        
#        loss = slim.losses.softmax_cross_entropy(self.logits, self.y_holder)
#        total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
#        return total_loss
#    
#    @property
#    def accuracy(self):
#        accuracy = tf.reduce_mean(
#                tf.cast(
#                    tf.equal(tf.argmax(self.logits,axis=1),tf.argmax(self.y_holder,axis=1)),
#                    tf.float32
#                )
#            )
#                
#        return accuracy
#    
#    @property
#    def train_op(self):
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
#        train_op = slim.learning.create_train_op(self.total_loss,optimizer=optimizer)
#        return train_op
#    
#    
    
    
    def get_logits(self):
        if hasattr(self, 'logits'):
            return self.logits
        y = slim.conv2d(inputs=self.x_holder,
                        num_outputs=32,
                        kernel_size=[3,3],
                        stride=[1,1],
                        scope="conv1")
        y = slim.max_pool2d(inputs=y,
                            kernel_size=[2,2],
                            scope="pool1")
        y = slim.flatten(inputs=y,
                         scope="flatten1")
        y = slim.fully_connected(inputs=y,
                                 num_outputs=32,
                                 scope="fc1")
        y = slim.fully_connected(inputs=y,
                                 num_outputs=self.provider.num_classes,
                                 scope="fc2")
        self.logits = y
        return y
    
    def get_total_loss(self):
        if hasattr(self, 'total_loss'):
            return self.total_loss
#        assert self.logits is not 0
        loss = slim.losses.softmax_cross_entropy(self.get_logits(), self.y_holder)
        self.total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
        return self.total_loss
    
    def get_accuracy(self):
        if hasattr(self, 'accuracy'):
            return self.accuracy
#        assert self.logits is not 0
        accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.get_logits(),axis=1),tf.argmax(self.y_holder,axis=1)),
                    tf.float32
                )
            )
                
        return accuracy
    
    
    def get_train_op(self):
        if hasattr(self, 'train_op'):
            return self.train_op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        train_op = slim.learning.create_train_op(self.get_total_loss(),optimizer=optimizer)
        return train_op
    
    
    