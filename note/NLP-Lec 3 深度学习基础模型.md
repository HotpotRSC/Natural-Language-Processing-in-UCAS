# NLP-Lec 3 深度学习基础模型

![image-20260116113150362](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116113150362.png)

## 3.1 概述

#### 深度学习

深度学习是机器学习的一个分支，其具有四个主要脉络：

![image-20260116113341571](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116113341571.png)

#### 自然语言处理中常用的神经网络模型

- **全连接前馈神经网络DNN**

- **卷积神经网络CNN**
- **图卷积神经网络GNN**
- **循环神经网络RNN**

## 3.2 全连接前馈神经网络DNN

### 3.2.1 人工神经元模型

#### 生物神经元

单个神经细胞只有两种状态——兴奋和抑制

#### 人工神经元模型

![image-20260116114243823](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114243823.png)

![image-20260116114007286](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114007286.png)

#### 激活函数

为了增强网络的表达能力，引入连续的非线性激活函数。

![image-20260116114205969](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114205969.png)

#### 人工神经网络

由多个神经元组成的具有并行分布结构的神经网络模型：

![image-20260116114444461](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114444461.png)

### 3.2.2 前馈神经网络DNN

#### 前馈神经网络DNN

在前馈神经网络中，各神经元分别属于不同的层，整个网络无反馈，信号从输入层向输出层单向传播，可用一个有向无环图表示：

![image-20260116114859434](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114859434.png)

### 3.3.3 梯度下降法

通过**有监督训练**来学习模型参数：通过训练集的实例数据$(x^i, y^i)$学习参数。

#### 迭代调参方法

通过调整参数，让模型输出递归性地逼近标准输出，即如何调节参数$\Theta$，使模型输出$\hat{y}^i$与标准输出$y^i$差距最小，我们可以通过定义损失函数$L(\Theta)$，并求$minL(\Theta)$。

#### 梯度下降法

给定训练数据集$\{(x^1, y^1),...,(x^r, y^r),...,(x^R, y^R)\}，调参方法：$$\theta^i=\theta^{i-1}-\mu\nabla L(\theta^{i-1})$。

- **梯度下降法（Gradient Descent）**

  $L(\theta)=\frac{1}{R}\sum_r||f(x^r;\theta)-y^r||$，$\theta^i=\theta^{i-1}-\mu\nabla L(\theta^{i-1})$，$\nabla L(\theta^{i-1})=\frac{1}{R}\sum_r\nabla L^r(\theta^{i-1})$

- **随机梯度下降法（Stochastic Gradient Descent）**

- **Mini-batch 梯度下降法（Mini-batch Stochastic Gradient Descent）**

### 3.3.4 反向传播算法BP

将输出误差以某种形式反传给各层的所有单元，各层按本层误差修正各单元的连接权值。

![image-20260116134421937](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116134421937.png)

前馈神经网络的**训练过程**可以分为：

1. 先前馈计算每一层的状态和激活值，直到输出层；
2. 反向传播计算每一层的误差；
3. 计算每一层参数的偏导数，并更新参数。

#### 梯度消失问题

在神经网络中误差反向传播的迭代公式为：$\delta^l = \sigma'(z^l) \cdot (W^{l+1})^T \delta^{l+1}$

其中需要用到激活函数$\sigma(z^L)$的导数误差从输出层反向传播时，每层都要乘激活函数的导数，这样当激活函数导数值小于1时，误差经过每一层的传递都会不断衰减，当**网络很深时甚至消失**。

#### 解决梯度消失问题方法

- 选择合适的激活函数
- 用复杂的门结构代替激活函数
- 残差结构

#### 解决过拟合问题方法

- 选择合适的正则方法
- 损失函数加入适当的正则项
- Dropout