#coding: utf-8

import tensorflow as tf
#import model.mynet
from dataset.mnist_dataset import MnistDataset
from dataset.cifar_dataset import CifarDataset
from provider.mnistprovider import MnistProvider
from utils import downloader
from model.easynet import EasyNet
#from tensorflow.contrib import slim



def main(_):
    
    dataset = MnistDataset()
    dataset.loadDataset(one_hot=True)
    batch_size = 100
    provider = MnistProvider(dataset, batch_size=batch_size)

    model = EasyNet(data_provider=provider)
    model.trainEpoch()

#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        sess.run(tf.local_variables_initializer())
#        for epoch in range(10):
#            flag = False
#            cnt = 0
#            frequency = 100
#            while flag is False:
#                cnt+=1
##                print("running epoch {}".format(epoch))
#                x,y,flag = mnist_provider.loadTrainBatch()
##                x = tf.convert_to_tensor(x)
##                y = tf.convert_to_tensor(y)
#                mydict = {x_holder:x, y_holder:y}
#                sess.run(train_op, feed_dict=mydict)
#                if cnt % frequency == 0:
#                    cnt=0
#                    print("loss:{}".format(sess.run(loss,feed_dict=mydict)))
#                    print("accuracy:{}".format(sess.run(accuracy,feed_dict=mydict)))
##            
    return

if __name__ == "__main__":
    tf.app.run()