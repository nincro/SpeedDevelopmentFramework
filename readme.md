# framework for speed development of deep-learning network
# 深度学习网络快速开发框架

# 目录结构
```
--code
  --dataset(数据抽象接口)
    --data(数据源)
  --provider(数据集接口)
  --model(深度学习模型接口)

```
# 模块说明
## dataset 数据集抽象（目前只针对image数据集）
将数据集抽象为ndarray格式
提供数据集height,width等参数


## dataprovider 数据集接口
针对模型提供特定大小的数据集，格式为ndarray

## model 模型定义
定义模型的各个模块，包括网络结构，损失函数，优化器等

# 使用说明
```py
#实例化一个dataset对象
dataset = MnistDataset()
#载入数据
dataset.loadDataset(one_hot=True)
#声明需要的batch_size
batch_size = 100
#实例化一个provider对象，并将具体的数据集对象传入
provider = MnistProvider(dataset, batch_size=batch_size)
#实例化一个模型，传入provider对象
model = EasyNet(data_provider=provider)
#调用模型自带的训练函数
model.trainEpoch()
```
