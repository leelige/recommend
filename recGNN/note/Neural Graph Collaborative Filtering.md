# Neural Graph Collaborative Filtering

**作者：Xiang Wang，Xiangnan He(corresponding author)，etc**

**单位：NSU，USTC**

**会议：SIGIR 2019**

## Introduction

We argue that an inherent drawback of such methods（传统的矩阵分解和MLP方法） is that, the **collaborative signal**, ==which is latent in user-item interactions,====is not encoded in the embedding process==（指明隐藏的用户-物品交互无法较好地用embedding表示）. As such, the resultant embeddings may **not** be sufficient to capture the collaborative filtering effect（导致协同过滤的效果不佳）.

**NGCF**是一个**框架**

This leads to the expressive modeling of **high-order connectivity**（**高阶连通性**） in user-item graph, effectively injecting the collaborative signal into the embedding process in an explicit manner.**[高阶连通性的目的就是为了获取隐式交互]**

[GitHub](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

![](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112164824687.png)

**从u1角度来看，多条路径：**

u1←i2←u2                 **表明u1和u2之间的行为相似性**

u1 ← i2 ← u2 ← i4    **表明u1可能采用i4（i5同理）**

u1 ← i3 ← u3 ← i4    **因为i4有两条路径，所以i4相比于i5可能更会引起u1的兴趣** 

## 论文改进点

Specifically, ==we devise an **embedding propagation** layer==, which refines a user’s (or an item’s) embedding by aggregating the embeddings of the interacted items (or users).

## 方法

### 初始化

**模型的input，以实现 end-to-end，embedding layer**



![image-20220112171056054](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112171056054.png)

### embedding propagation layers

使用GNNs的架构——GCN,2017

***message construction* and *message aggregation***

message construction（信息的构建）

![image-20220112193708857](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112193708857.png)

![image-20220112193720121](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112193720121.png)

根据GCN论文中的提示，前面的系数为归一化系数

message aggregation（信息的融合）

**1-hop的aggregation**

![image-20220112195355302](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112195355302.png)

**l-hop的agregation**

![image-20220112200110718](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112200110718.png)

![image-20220112200237517](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112200237517.png)

![image-20220112200622879](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112200622879.png)

**写成矩阵的形式：**

![image-20220112201639235](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112201639235.png)

![image-20220112202026331](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112202026331.png)

![image-20220112202148850](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112202148850.png)

## model prediction

![image-20220112202226602](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112202226602.png)

![image-20220112202631595](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112202631595.png)

**实际这里采用了类似 18 ICML *Representation Learning on Graphs with Jumping Knowledge Networks* 的做法来防止 GNN 中的过平滑问题。GNN 的过平滑问题是指，随着 GNN 层数增加，GNN 所学习的 Embedding 变得没有区分度。过平滑问题与本文要捕获的高阶连接性有一定的冲突，所以这里需要在克服过平滑问题。** 来源:[zhihu](https://zhuanlan.zhihu.com/p/84313274)

## optimization

BPR-pariwise loss function



![image-20220112203807489](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112203807489.png)

![image-20220112204052455](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112204052455.png)

L通常设置大小<5

> 例如，在我们实验的**Gowalla**数据集(20K用户和40K项目)上，当嵌入大小为64，我们使用3个大小为64×64的传播层时，MF有450万参数，而我们的NGCF只使用2.4万附加参数。总之，NGCF使用了很少的额外模型参数来实现高阶连通性建模。

### overfittiing problem

***message dropout* and *node dropout***（只在训练中使用）

message dropout：随机删除传出的消息 ，论文中以概率p1去删除

node dropout：随机删除拉普拉斯矩阵中的节点，论文中以p2*(M+N)

## SVD

SVD++可以看作是没有高阶传播层的NGCF的一种特情况

> 特别地，我们把L设为1。在传播层内，我们禁用了变换矩阵和非线性激活函数。然后，eu和ei分别视为用户u和项i的最终表示。我们将这个简化的模型命名为NGCF-SVD，它可以表述为：

![image-20220112211758568](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112211758568.png)

### time complexity

![image-20220112212203186](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112212203186.png)

## 实验

![image-20220112213403777](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112213403777.png)

作者对每个数据集的user-item做了处理，**确保每个user和每个item都至少有10个交互**（10-core setting）

![image-20220112213847032](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112213847032.png)

所有的方法都用BPR的loss function

### parameter setting

- platform：TensorFlow
- embedding size：64
- batch size：1024
- L ：3

### sparsity issue

> 稀疏性问题通常限制了推荐系统的表达性，因为非活跃用户之间的很少有交互作用不足以生成高质量的表示。我们将研究利用**连接信息**是否有助于缓解这个问题。

**以Gowalla数据集为例，每个用户的交互数分别小于24、50、117和1014。图4说明了结果w.r.t.在Gowalla、Yelp2018和亚马逊图书的不同用户组中，以recall@20为例：**

![image-20220112215027423](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112215027423.png)

**验证了嵌入传播对相对不活跃的用户有利**



**Effect of Embedding Propagation Layer and Layer Aggregation Mechanism**

![image-20220112223512525](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112223512525.png)

**为了研究嵌入传播（即图卷积）层如何影响性能，我们考虑了使用不同层的NGCF-1的变体。特别地，我们从消息传递函数中删除了节点与其邻居之间的表示交互**（下图所示）

![image-20220112223619503](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112223619503.png)

**Effect of Dropout**

![image-20220112223725928](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112223725928.png)

### **Effect of High-order Connectivity**

node可视化

![image-20220112224010896](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220112224010896.png)

## future work

有许多其他形式的结构信息可以有助于理解用户行为，比如上下文感知和语义丰富的推荐中的交叉特征、项目知识图谱和社交网络。例如，通过将项目知识图谱与用户-项目图集成，我们可以在用户与项目之间建立知识感知的连接，这有助于揭示用户在选择项目时的决策过程。**我们希望NGCF的发展将有利于用户在线行为的推理，以获得更有效和可解释的推荐。**
