# Dataset 数据集抽象
## 传入参数
* path_to_read 数据源存放的路径

## 需要override的函数
* loadDataset 载入数据函数

## 需要指定的参数
* x_shape 特征矩阵的shape

* y_shape 类别矩阵的shape

* num_classes 类别数目

其余参数请根据实际需要进行补充

## 数据缓存格式
* numpy.ndarray
