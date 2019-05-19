# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:33:30 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.contrib import slim
from .component.logits.logits import myLogits
class Model(object):
    def __init__(self, 
                 data_provider,
                 depth=0, 
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
        

        self.logits = myLogits(self.x_holder, self.provider.getNumClasses())
        
#        self.total_loss = self.getTotalLoss()
        self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.getLogits(), axis=1), tf.argmax(self.y_holder, axis=1)),
                    tf.float32
                )
            )
        
        self.total_loss = slim.losses.softmax_cross_entropy(self.getLogits(), self.y_holder)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        self.train_op =  self.getTrainOperation()
        
        self.initialize()
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


    def setDataProvider(self, data_provider):
        self.provider = data_provider
        return

    def setLogits(self, logits):
        self.logits = logits
        return

    def getLogits(self):
        return self.logits
    
    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        return
    
    def getOptimizer(self):
        return self.optimizer
    
    def getTrainOperation(self):
        if hasattr(self, 'train_op'):
            return self.train_op
        
        train_op = slim.learning.create_train_op (total_loss=self.getTotalLoss(), optimizer=self.getOptimizer())
        return train_op
    
    
    def setTotalloss(self, loss):
        self.loss = loss(self.getLogits(),self.y_holder);
        return
    
    def getTotalLoss(self):
        return self.total_loss
    
    def setAccuracy(self, accuracy):
        self.accuracy = accuracy
        return
    
    def getAccuracy(self):
        
                
        return self.accuracy
    
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
    