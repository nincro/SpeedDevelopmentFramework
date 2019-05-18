# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:33:30 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.contrib import slim
class Net(object):
    def __init__(self, 
                 data_provider,
                 depth, 
                 should_save_log=False,
                 should_save_model=False,
                 renew_logs=False,
                 log_path="./saved_log",
                 model_path="./saved_model",
                 **kwargs):
        self.provider = data_provider
        self.x_holder = tf.placeholder(dtype=tf.float32, shape=data_provider.xshape)
        self.y_holder = tf.placeholder(dtype=tf.float32, shape=data_provider.yshape)
        self.frequency = 100
        
        self.depth              = depth
        self.should_save_log   = should_save_log
        self.should_save_model  = should_save_model
        self.log_path = log_path
        self.model_path = model_path
        

        self.logits = self.getLogits()
        self.train_op =  self.getTrainOperation()
        self.total_loss = self.getTotalLoss()
        self.accuracy = self.getAccuracy()
        
        return
    
    def initialize(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.log_path)


    def getLogits(self):
        return NotImplementedError
    
    def getTrainOperation(self):
        if hasattr(self, 'train_op'):
            return self.train_op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        train_op = slim.learning.create_train_op(self.getTotalLoss(), optimizer=optimizer)
        return train_op
    
    def getTotalLoss(self):
        if hasattr(self, 'total_loss'):
            return self.total_loss
#        assert self.logits is not 0
        loss = slim.losses.softmax_cross_entropy(self.getLogits(), self.y_holder)
        self.total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
        return self.total_loss
    
    def getAccuracy(self):
        if hasattr(self, 'accuracy'):
            return self.accuracy
#        assert self.logits is not 0
        accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.getLogits(), axis=1), tf.argmax(self.y_holder, axis=1)),
                    tf.float32
                )
            )
                
        return accuracy
    
    def trainOneEpoch(self):
        flag = False
#        self.get_logits()
#         with tf.Session() as sess:
        sess = self.sess
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnt=0
        while flag is False:
            cnt+=1
            x,y,flag = self.provider.loadTrainBatch()
            mydict = {self.x_holder:x, self.y_holder:y}

            sess.run(self.getTrainOperation(), feed_dict=mydict)
            if cnt % self.frequency == 0:
                cnt=0
                print("loss:{}".format(sess.run(self.getTotalLoss(), feed_dict=mydict)))
                print("accuracy:{}".format(sess.run(self.getAccuracy(), feed_dict=mydict)))
#            
        return
    