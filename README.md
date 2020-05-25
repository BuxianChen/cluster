# README

## 0. 环境配置与使用

```
python3.7
tensorflow2.0
numpy
scipy # 用于计算聚类指标
scikit-learn
h5py # 读取.h5格式的usps数据集
scikit-fuzzy # 非必须, 因为实际上没有使用模糊c-means聚类
```

启动方法（更多参数及含义参见`main.py`）：

```shell
# 默认情况下代码的输出会显示在屏幕上，并保存在results/log/文件夹下
python main.py -d mnist # 或者usps, fashion_mnist
```

## 1. 代码架构:

```
--data  # 数据处理模块
  --datasets.py  # 读入数据
  --data_transforms.py  # 数据增强
--datasets # 数据集文件夹
--models  # 模型模块
  --autoencoders  # 自编码网络
  --baselines  # kmeans, fuzzy-cmeans等基准算法
  --layers  # 一些层的定义与封装
  --model.py  # 论文中的算法
--results  # 实验结果
  --log
    --{model_name}_{data_name}_{timestamp}.txt
--utils
  --logger.py  # 对logging的封装
  --losses.py  # 损失函数
  --metrics.py  # 评价指标
  --optimizers.py  # 优化器
main.py  # 启动文件
```
## 2. 数据集与模型的实验结果

MNIST数据集与Fashion-MNIST数据集代码会自动进行下载，USP数据集可从[kaggle](https://www.kaggle.com/bistaumanga/usps-dataset)上下载`usps.h5`文件，并放在datasets/usps目录下。

MNIST数据集的结果需要调参才能达到下面的数值（不刻意调参大约为97.5/94.0），USPS数据集的结果超过下面的数值是正常现象。

| 模型/数据集/(ACC/NMI) | MNIST | Fashion_MNIST | USPS   |
| ----------- | ---------- | ------------- | ---------- |
| model    | 98.3/95.1 | 58.9/61.2 | 85.5/90.4 |
