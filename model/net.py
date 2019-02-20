# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:33:30 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.contrib import slim
class Net(object):
    def __init__(self, data_provider):
        self.provider = data_provider
        self.x_holder = tf.placeholder(dtype=tf.float32, shape=data_provider.xshape)
        self.y_holder = tf.placeholder(dtype=tf.float32, shape=data_provider.yshape)
        self.frequency = 100
        
#        self.logits
#        self.train_op
#        self.total_loss
#        self.accuracy
        
        self.logits = self.get_logits()
        self.train_op =  self.get_train_op()
        self.total_loss = self.get_total_loss()
        self.accuracy = self.get_accuracy()
        
        return
    
#    @property
#    def logits(self):
#        return NotImplementedError
#    @property
#    def train_op(self):
#        return NotImplementedError
#    @property
#    def total_loss(self):
#        return NotImplementedError
#    @property
#    def accuracy(self):
#        return NotImplementedError
    
    
    
    def get_logits(self):
        return NotImplementedError
    
    def get_train_op(self):
        if hasattr(self, 'train_op'):
            return self.train_op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        train_op = slim.learning.create_train_op(self.get_total_loss(),optimizer=optimizer)
        return train_op
    
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
    
    def trainEpoch(self):
        flag = False
#        self.get_logits()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            cnt=0
            while flag is False:
                cnt+=1
                x,y,flag = self.provider.loadTrainBatch()
                mydict = {self.x_holder:x, self.y_holder:y}
                
                sess.run(self.get_train_op(), feed_dict=mydict)
                if cnt % self.frequency == 0:
                    cnt=0
                    print("loss:{}".format(sess.run(self.get_total_loss(),feed_dict=mydict)))
                    print("accuracy:{}".format(sess.run(self.get_accuracy(),feed_dict=mydict)))
#            
        return
    