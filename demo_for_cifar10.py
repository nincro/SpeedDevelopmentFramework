#coding: utf-8

import tensorflow as tf
from dataset.mnist_dataset import MnistDataset
from dataset.cifar_dataset import Cifar10Dataset
from provider.ImageProvider import ImageProvider
from model.easynet import EasyNet



def main(_):
    dataset = Cifar10Dataset()
    dataset.loadDataset(one_hot=True)
    batch_size = 100
    provider = ImageProvider(dataset, batch_size=batch_size)

    model = EasyNet(data_provider=provider)
    model.trainEpoch()
    return

if __name__ == "__main__":
    tf.app.run()