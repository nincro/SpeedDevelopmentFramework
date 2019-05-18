# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:12:56 2019

@author: Administrator
"""
from .net import Net
from tensorflow.contrib import slim
import tensorflow as tf
class Xception(Net):
    

    def __init__(self,provider):
        super().__init__(provider=provider)
        self.depth = 130
    
    def getLogits(self):
        if hasattr(self, 'logits'):
            return self.logits
        with slim.arg_scope(self.arg_scope()):
            # ===========ENTRY FLOW==============
            with tf.variable_scope("ENTRY_FLOW"):
                # Block 1
                net = slim.conv2d(self.x_holder, 32, [3, 3], padding='same')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 32, [3, 3], padding='same')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                residual = slim.conv2d(net, 64, [1, 1], stride=2)
                residual = slim.batch_norm(residual,)

                # Block 2
                net = slim.separable_conv2d(net, 64, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 64, [3, 3])
                net = slim.batch_norm(net)
                net = slim.max_pool2d(net, [3, 3], stride=2, padding="same")
                net = tf.add(net, residual)

            # residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
            # residual = slim.batch_norm(residual, scope='block2_res_bn')

            #
            # # Block 3
            # net = tf.nn.relu(net, name='block3_relu1')
            # net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv1')
            # net = slim.batch_norm(net, scope='block3_bn1')
            # net = tf.nn.relu(net, name='block3_relu2')
            # net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv2')
            # net = slim.batch_norm(net, scope='block3_bn2')
            # net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
            # net = tf.add(net, residual, name='block3_add')
            # residual = slim.conv2d(net, 728, [1, 1], stride=2, scope='block3_res_conv')
            # residual = slim.batch_norm(residual, scope='block3_res_bn')
            #
            # # Block 4
            # net = tf.nn.relu(net, name='block4_relu1')
            # net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv1')
            # net = slim.batch_norm(net, scope='block4_bn1')
            # net = tf.nn.relu(net, name='block4_relu2')
            # net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv2')
            # net = slim.batch_norm(net, scope='block4_bn2')
            # net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block4_max_pool')
            # net = tf.add(net, residual, name='block4_add')

            # ===========MIDDLE FLOW===============
            n = (self.depth-10)//3
            with tf.name_scope("MIDDLE_FLOW"):
                for i in range(n):
                    residual = net
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.add(net, residual)

            # ========EXIT FLOW============
            with tf.variable_scope("EXIT_FLOW"):
                residual = slim.conv2d(net, 128, [1, 1], stride=2)
                residual = slim.batch_norm(residual)

                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)

                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)

                net = slim.max_pool2d(net, [3, 3], stride=2, padding='same')
                net = tf.add(net, residual)

                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)

                net = tf.reduce_mean(net, axis=[1, 2])

            logits = slim.fully_connected(net, self.provider.num_classes)
            self.logits = logits
        return logits
    
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
    
    
    def getTrainOperation(self):
        if hasattr(self, 'train_op'):
            return self.train_op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        train_op = slim.learning.create_train_op(self.getTotalLoss(), optimizer=optimizer)
        return train_op
    