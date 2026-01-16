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

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114444461.png" alt="image-20260116114444461" style="zoom: 50%;" />

### 3.2.2 前馈神经网络DNN

#### 前馈神经网络DNN

在前馈神经网络中，各神经元分别属于不同的层，整个网络无反馈，信号从输入层向输出层单向传播，可用一个有向无环图表示：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116114859434.png" alt="image-20260116114859434" style="zoom:50%;" />

### 3.2.3 梯度下降法

通过**有监督训练**来学习模型参数：通过训练集的实例数据$(x^i, y^i)$学习参数。

#### 迭代调参方法

通过调整参数，让模型输出递归性地逼近标准输出，即如何调节参数$\Theta$，使模型输出$\hat{y}^i$与标准输出$y^i$差距最小，我们可以通过定义损失函数$L(\Theta)$，并求$minL(\Theta)$。

#### 梯度下降法

给定训练数据集$\{(x^1, y^1),...,(x^r, y^r),...,(x^R, y^R)\}，调参方法：$$\theta^i=\theta^{i-1}-\mu\nabla L(\theta^{i-1})$。

- **梯度下降法（Gradient Descent）**

  $L(\theta)=\frac{1}{R}\sum_r||f(x^r;\theta)-y^r||$，$\theta^i=\theta^{i-1}-\mu\nabla L(\theta^{i-1})$，$\nabla L(\theta^{i-1})=\frac{1}{R}\sum_r\nabla L^r(\theta^{i-1})$

- **随机梯度下降法（Stochastic Gradient Descent）**

- **Mini-batch 梯度下降法（Mini-batch Stochastic Gradient Descent）**

### 3.2.4 反向传播算法BP

将输出误差以某种形式反传给各层的所有单元，各层按本层误差修正各单元的连接权值。

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116134421937.png" alt="image-20260116134421937" style="zoom: 50%;" />

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

## 3.3 卷积神经网络CNN

### 3.3.1 概述

当处理图像数据时，使用全连接前馈神经网络会带来巨大的参数量，导致训练效率低下。因此，我们通过改变网络结构，使其在**达到同样的效果下，参数量更少**：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116140350043.png" alt="image-20260116140350043" style="zoom:67%;" />

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116140417139.png" alt="image-20260116140417139" style="zoom:67%;" />

**多卷积核连接：**

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116140653320.png" alt="image-20260116140653320" style="zoom:67%;" />

一个$3\times 3$的卷积核共有9个参数，M个卷积核共有$M\times 9$个参数。

#### 示例：全连接与卷积连接参数对比

![image-20260116141037721](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116141037721.png)

如一副$10\times 10$图像，设隐藏层神经元为1024:

- **全连接：**第一层参数为$1024\times100=102400$
- **卷积连接：**
  - 用16个$3\times3$滤波器进行卷积操作后得到$16\times 8\times 8=1024$个卷积层神经元，第一层参数为$16\times9=144$
  - 用12个$2\times2$滤波器进行卷积操作后得到$12\times 9\times 9=1053$个卷积层神经元，第一层参数为$12\times4=52$

### 3.3.2 卷积神经网络结构

#### 池化层（Pooling）

卷积连接虽然减少了网络的参数量，单网络规模（节点数）并没有减少。考虑到我们对图像进行采样并不会改变图像对分类结果：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116141740423.png" alt="image-20260116141740423" style="zoom:50%;" />

我们采用池化操作对图像进行采样缩小网络规模，同时进一步减少参数量：

![image-20260116141856775](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116141856775.png)

**常用的池化操作：**

- **Max Pooling**
- **Mean Pooling**

#### 全连接层

将最后池化层的单元“平化”然后组成全连接输入网：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116142228115.png" alt="image-20260116142228115" style="zoom: 67%;" />

#### CNN网络结构

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116142309997.png" alt="image-20260116142309997" style="zoom: 50%;" />

卷积神经网络具有以下特性：

- **局部连接**
- **权重共享**
- **空间或时间上的次采样**

**这些特性使得卷积神经网络具有一定程度的平移、缩放和扭曲不变性。**

### 3.3.3 卷积神经网络学习

#### 参数学习

CNN在训练时，卷积层和池化层作为一个整体：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116142551687.png" alt="image-20260116142551687" style="zoom:50%;" />

### 3.3.4 卷积神经网络应用

#### CNN在NLP中应用

- CNN可用于NLP中的各种分类任务：文本分析、情感分析、实体关系抽取等
- 用于其他任务的特征提取

#### 情感分类

![image-20260116142812307](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116142812307.png)

## 3.4 图卷积神经网络GNN

### 3.4.1 概述

#### 图卷积神经网络：

- **基于谱域的图卷积神经网络：**

  在谱域中定义卷积，卷积是通过图傅立叶变换和卷积定理定义的

- **基于空间的图卷积神经网络：**

  定义定点域中的卷积，卷积定义为目标定点的邻域中所有顶点的加权平均值函数

### 3.3.2 基于空间的GNN

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116143241461.png" alt="image-20260116143241461" style="zoom:67%;" />

#### GNN模型结构

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116143400104.png" alt="image-20260116143400104" style="zoom: 67%;" />

#### GNN卷积步骤

**（1）Aggregation；（2）Transformation**

- Basic GNN：

  $h_v^k=\sigma\left(W_k\sum\limits_{u\in N(v)}\frac{h_u^{k-1}}{|N(v)|}+B_kh_v^{k-1}\right)$

- GCN：

  $h_v^k=\sigma\left(W_k\sum\limits_{u\in N(v)\cup v}\frac{h_u^{k-1}}{\sqrt{|N(v)|\cdot|N(v)|}}\right)$

#### GNN模型训练

在最后一层（K层）得到每个节点的表示后，可以根据其任务将其带入损失函数，然后利用梯度下降法训练参数。

### 3.4.3 GNN变形

#### GNN变形

根据聚集和层级连接方式的不同派生出大量不同形式的GNN。

- **GNN Models based on Propagation Step:**

![image-20260116145625097](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116145625097.png)

- **GNN Models based on Connection:**

![image-20260116145903289](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116145903289.png)

## 3.5 循环神经网络RNN

### 3.5.1 概述

单一DNN、CNN无法处理时许相关序列问题。

#### 循环神经网络核心思想

将处理问题在时序上分解为一系列相同的“单元”，单元的神经网络可以在时许上展开，且能将上一时刻的**结果传递给下一时刻**，整个网络按时间轴展开。

### 3.5.2 RNN单元结构

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116152929519.png" alt="image-20260116152929519" style="zoom:50%;" />

#### RNN单元按时序展开

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116153046848.png" alt="image-20260116153046848" style="zoom:50%;" />

#### RNN输入输出结构

RNN的输入和输出结构可以等长或不等长：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116153132718.png" alt="image-20260116153132718" style="zoom:50%;" />

### 3.5.3 循环神经网络训练

#### 参数学习

**有监督训练：**训练集数据$\{(x^1, y^1),...,(x^r, y^r),...,(x^R, y^R)\}$，模型参数$\theta$，用$y$与$\hat{y}$的误差单一损失函数$L(\theta)$，然后使用梯度下降法学习参数。

#### 学习算法——BPTT（Backpropagation Through Time）

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116154329568.png" alt="image-20260116154329568" style="zoom: 25%;" />

在 RNN 中，前向传播阶段通过计算**隐藏状态和输入**来生成预测结果:$h_t=f(Ux_t+Wh_{t-1})$，$\hat{h}_t=g(Vh_t)$。

我们以一个长度为3的时间序列为例，展示对于参数U、V和W的偏导数的计算过程：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116154751877.png" alt="image-20260116154751877" style="zoom: 25%;" />

将上面的RNN用数学表达式来表示就是：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116155240523.png" alt="image-20260116155240523" style="zoom: 50%;" />

针对$t=3$时刻，求$U，V，W$的梯度（偏导），使用链式法则得到：

![image-20260116155336180](/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116155336180.png)

所以，我们可以根据$t=3$时刻的偏导，来计算任意时刻对$U，V，W$的偏导:

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116155657580.png" alt="image-20260116155657580" style="zoom:50%;" />

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116155719502.png" alt="image-20260116155719502" style="zoom:50%;" />

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116155732843.png" alt="image-20260116155732843" style="zoom:50%;" />

#### 梯度消失/爆炸问题

根据上述推到的偏到公式，我们可以看到在求偏导过程中涉及到连乘，这使得RNN在训练时更容易出现梯度消失或爆炸问题，使得训练困难。

### 3.5.4 循环神经网络改进及变形

由于距离当前节点越远的节点对当前节点处理的影响越小，无法建模长时间的依赖关系（循环神经网络的长期依赖问题）

#### 长短时记忆神经网络LSTM

LSTM单元不仅接受$x_t$和$h_{t-1}$，还建立量一个机制（维持一个状态$C_t$），能保留前面远处节点信息在长距离传播中不会丢失：

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116160314166.png" alt="image-20260116160314166" style="zoom:50%;" />

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116160445015.png" alt="image-20260116160445015" style="zoom:50%;" />

#### GRU（Gated Recurrent Unit）

**GRU是简化的LSTM**：输入门和遗忘门合并为更新门（更新门决定隐状态保留放弃部分）

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116161026444.png" alt="image-20260116161026444" style="zoom:50%;" />

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116161049036.png" alt="image-20260116161049036" style="zoom:50%;" />

### 3.5.5 循环神经网络的应用

<img src="/Users/fangyiqin/Library/Application Support/typora-user-images/image-20260116161201204.png" alt="image-20260116161201204" style="zoom:50%;" />