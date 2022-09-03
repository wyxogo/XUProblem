## 问题描述
图像分类是一个输入图像，输出该图像所属类别的CV核心问题之一。图像分类的传统方法是特征描述及检测，其对简单数据有效，但却不适用于如今复杂的数据。如今深度学习是处理图像分类的主流方法。面对复杂数据，我们通常通过更改网络结构、损失函数、或数据预处理等操作实现提高准确率的效果。但是不同的处理其优化方式、适用范围不同。（请在pytorch平台上进行实验，代码中需注释表明关键模块）

1. 使用pytorch平台提供的预训练的VGG19作为backbone，交叉熵为损失函数，对cifar100中10类(maple, bed, bus, plain, dolphin, bottles, cloud, bridge, baby, rocket)训练一个10分类网络。记录网络在测试集上对不同类别的识别准确率，并保存网络模型。

2. 将问题1中的十个类别更换为：maple, bed, bus, plain, streetcar, oak, palm, pine, willow, forest。使用相同的实验配置训练一个10分类网络，与问题1中的分类准确率比较，并分析它们的差异。如果新的准确率更低，请从损失函数的角度提高分类精度。
3. 对问题1，2的网络读取卷积层最后一层的特征，将这两组特征分别通过T-SNE可视化成二维的特征分布图。比较这两个特征分布图，结合问题1、2试着分析不同的原因。

## 文件目录

```
.
├── README.md
├── colab.ipynb
├── dataset
│   ├── cifar-100-python
│   │   ├── file.txt~
│   │   ├── meta
│   │   ├── test
│   │   └── train
│   └── cifar-100-python.tar.gz
├── datasets.py
├── main.py
├── outputs1
│   ├── T-SNE1.jpg
│   ├── best_model
│   │   └── model.pth
│   ├── test_log.txt
│   └── train-xxxxxxxx-12-44-18
│       └── log.txt
├── outputs2
│   ├── T-SNE2.jpg
│   ├── best_model
│   │   └── model.pth
│   ├── test_log.txt
│   └── train-xxxxxxxx-12-53-37
│       └── log.txt
├── tsne.py
├── utils.py
└── vgg_model.py
```
`dataset`为数据集路径，第一次执行后会自动创建下载

`outputs` 中保存输出文件，包括`T-SNE`图，训练日志，训练模型保存，测试日志，其中`best_model`中保存的是所有训练保存模型中损失函数最低的模型，测试时需要加载该模型

## 使用

### 在Colab上使用

<a href="https://colab.research.google.com/github/wyxogo/XUProblem/blob/master/colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 打开链接后逐步执行即可（需科学用网）

### 在本地使用

```bash
# clone repo:
git clone https://github.com/wyxogo/XUProblem.git
cd XUProblem
# 问题一训练
python main.py --mode=train --batch_size=64 --epochs=50 --output="./outputs1/" --problem=1
# 问题一测试
python main.py --mode=test --batch_size=32 --output="./outputs1/" --problem=1
# 问题二训练
python main.py --mode=train --batch_size=64 --epochs=50 --output="./outputs2/" --problem=2
# 问题二测试
python main.py --mode=test --batch_size=32 --output="./outputs2/" --problem=2
# T-SNE图分析
python main.py --tsne --mode=test --batch_size=32 --output="./outputs1/" --problem=1   
python main.py --tsne --mode=test --batch_size=32 --output="./outputs2/" --problem=2

```