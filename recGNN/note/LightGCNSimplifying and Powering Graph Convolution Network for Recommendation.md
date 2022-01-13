# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation 

**作者：Xiangnan He(第一作者), etc**

**单位：USTC**

**会议：SIGIR 2020**

## GCNs的问题

**the two most common designs in GCNs ：**

- feature transformation （特征转换）
- nonlinear activation （非线性激活）

Even worse, includi ng them adds to the difficulty of training and degrades recommendation performance.【主要问题：这两部分增加了训练难度，并降低了推荐效率】。

**论文工作**：In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation.【简化GCN的设计，使其更简洁和适合推荐】

LightGCN ——==neighborhood aggregation==

具体来说，LightGCN通过在用户-项目交互图上线性传播它们来学习用户和项目嵌入，并使用在所有层上学习到的嵌入的加权和作为最终的嵌入。这种简单、线性、整洁的模型更容易实现和训练，在完全相同的实验设置下，比神经图协同过滤的基于GCN的推荐模型**(NGCF)**表现出实质性的改进（**平均相对提高16.0%**）。

- [github_tensorflow](https://github.com/kuandeng/LightGCN)
- [github_pytorch](https://github.com/gusye1234/pytorch-light-gcn)

## Model

![image-20220113141324600](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113141324600.png)

###  ablation studies

![image-20220113150832901](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113150832901.png)

![image-20220113151718052](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113151718052.png)

对于这三个变量，我们保留了所有的超参数（如学习率、正则化系数、辍学率等）。与NGCF的最佳设置相同。我们在表1中报告了Gowalla和Amazon-Book数据集的两层设置的结果。可以看出，**删除特征转换(即NGCF-f)会导致在所有三个数据集上对NGCF的一致改进**。相比之下，去除非线性激活对精度影响不大。然而，**如果我们在去除特征变换(即NGCF-fn)的基础上去除非线性激活，性能就会显著提高**。从这些观察结果中，我们得出了以下结论：

- 添加特征变换对NGCF产生负面影响，在NGCF和NGCF-n模型中去除，可显著提高性能；
- 当包含特征变换时，添加非线性激活会影响轻微，但当禁用特征变换时，它会产生负面影响。
- 总的来说，**特征转换和非线性激活对NGCF产生了相当负面的影响**，因为通过同时去除它们，NGCF-fn比NGCF有了很大的改善（召回率相对提高了9.57%）。

由图1可以看出，在整个训练过程中，NGCF-fn实现的训练损失比NGCF、NGCF-f和NGCF-n要低得多。与测试回忆的曲线对齐，我们发现这种较低的训练损失成功地转移到**更好的推荐精度**。NGCF与NGCF-f的比较显示出相似的趋势，只是改进幅度较小。

**从这些证据中，我们可以得出结论——NGCF的恶化源于训练困难，而不是过拟合**

从理论上讲，NGCF比NGCF-f具有更高的表示能力，因为将权值矩阵W1和W2设置为单位矩阵，I可以完全还原NGCF-f。然而，在实际应用中，NGCF比NGCF-f具有**更高的训练损失和更差的泛化性能**。而非线性激活的加入进一步加剧了表示能力与泛化性能之间的差异。

## method

### LightGCN

![image-20220113154434478](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113154434478.png)

lightgcn放弃使用特征变换和非线性激活

![image-20220113154849605](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113154849605.png)

**在LGC在(Light Graph Convoution)中，我们只聚合已连接的邻居，而不集成目标节点本身（即自连接）**。这与大多数现有的图卷积操作不同，后者通常会聚合扩展的邻居，并且需要专门处理自连接。

在LightGCN中，唯一可训练的模型参数是第0层的嵌入，即所有用户的e0u和所有项的e0i

![image-20220113155751190](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113155751190.png)

**其中，αk≥0表示第k层嵌入在构成最终嵌入中的重要性(本质上就是注意力机制)**。它可以被视为需要手动调整的超参数，也可以作为需要自动优化的模型参数。在我们的实验中，我们发现将αk均匀设置为**1/(K+1)**可以获得良好的性能

我们执行**图层组合**来得到最终表示的原因有三方面：

- （1）随着层数的增加，嵌入将过度平滑。因此，简单地使用最后一层是有问题的。
- （2）不同层上的嵌入捕获不同的语义。例如，第一层强制用户和有交互的项目平滑，第二层平滑有交互项目（用户）重叠的用户（项目），更高的层捕获高阶接近。因此，结合它们将使表示更加全面。
- （3）将不同层的嵌入与加权和结合起来，捕获了图卷积与自连接的影响，**这是GCNs中的一个重要技巧**。

**prediction layer：**

![image-20220113160927807](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113160927807.png)

用户-项目交互矩阵为R∈RM×N，其中M和N分别表示用户和项目的数量，如果你与项目i交互，则每个条目Rui为1，否则为0。然后得到用户项图的邻接矩阵为

![image-20220113162037087](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113162037087.png)

设第0层嵌入矩阵为E（0）∈R(M+N)×T，其中T为嵌入大小。然后我们可以得到LGC的矩阵等价形式为：

![image-20220113162714503](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113162714503.png)![image-20220113162758711](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113162758711.png)

### analysis

**通过进行层组合，LightGCN包含了自连接的影响，因此LightGCN不需要在邻接矩阵中添加自连接**

**SGCN: **

![image-20220113163426672](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113163426672.png)

其中，I∈R(M+N)×(M+N)是一个**单位矩阵**，它被添加在a上以包含自连接。在下面的分析中，为了简单起见，我们省略了(D+I)−1/2，因为它们只重新缩放嵌入。在SGCN中，在最后一层获得的嵌入用于下游预测任务，可以表示为：

![image-20220113163626080](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113163626080.png)

**APPNP(将GCN与个性化的PageRank联系起来，提出了一种名为APPNP的GCN变体，它可以远程传播而没有过度平滑的风险)**

**始终考虑根节点**

![image-20220113164258064](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113164258064.png)

**最后一层：**

![image-20220113164428616](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113164428616.png)

**通过相应地设置αk，LightGCN可以完全恢复APPNP所使用的预测嵌入。**

==因此，LightGCN在对抗过平滑方面具有APPNP的优势——通过正确地设置α，我们允许使用一个大的K来进行可控的过平滑的长距离建模。==

> APPNP在邻接矩阵中添加了自连接。然而，正如我们之前所示的，由于不同层的加权和，这是多余的

**second-order embedding smoothness**

![image-20220113170133569](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113170133569.png)

**推理过程：**

<img src="https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113171719438.png" alt="image-20220113171719438" style="zoom:50%;" />

![image-20220113170818418](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113170818418.png)

这个系数是可解释的：**二阶邻居v对u的影响是由**

- 1)相互作用的项目的数量越大，coefficient越大；

- 2)相互作用项目的受欢迎程度，受欢迎程度越低（即，更表明用户的个性化偏好），coefficient越大；

- 3)v的活动越少，coefficient越大。

  **这种可解释性很好地满足了CF在测量用户相似性时的假设，并证明了LightGCN的合理性。**

### model training

**LightGCN的可训练参数只有第0层的嵌入，即Θ={E(0)}**；也就是说，模型的复杂度与标准矩阵分解(MF)相同

![image-20220113173248712](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113173248712.png)

dropout策略同ngcf：node dropout and message dropout 

重点：αK的探索，论文作者通过验证集上的实验去探索

> 然而，我们发现在训练数据上学习α并不能导致改进。这可能是因为训练数据没有包含足够的信号来学习好的α，可以推广到未知的数据。我们还试图从验证数据中学习α，正如受到[5]学习验证数据上的超参数的启发一样。性能略有提高（小于1%）。我们将对α的最佳设置的探索（例如，为不同的用户和项目个性化它）作为未来的工作

## 实验

**为了减少实验工作量，保持比较的公平，我们密切遵循NGCF工作的设置。我们采取ngcf作者使用的实验数据集（包括训练/测试 分割比例）**

![image-20220113174032315](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113174032315.png)

- embedding_size:64
- initialization: Xavier method
- optimization: adam
- batchsize: 1024-2048
- αk: 1/(1+K)
- K:1~4(层数)     **当k=3时取得较好效果**

### result

![image-20220113175507773](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113175507773.png)

![image-20220113141324600](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113141324600.png)

**可以看到LightGCN比NGCF-fn表现得更好，这是NGCF的变体，可以消除特征转换和非线性激活。由于NGCF-fn仍然比LightGCN包含更多的操作(例如，自连接，用户嵌入和图卷积中项目之间的交互嵌入），这表明这些操作对于NGCF-fn也可能是无用的。**

![image-20220113180146774](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113180146774.png)

**增加层数可以提高性能，但好处会减少**。一般的观察结果是，将层数从0（即矩阵分解模型）增加到1可以获得最大的性能增益，并且在大多数情况下，使用**层数为3**可以获得令人满意的性能。这一观察结果与NGCF的发现相一致。

在训练过程中，==LightGCN始终获得较低的训练损失，说明LightGCN对训练数据的拟合性优于NGCF。此外，较低的训练损失成功地转移到较高的测试精度，表明LightGCN具有较强的泛化能力。相比之下，NGCF较高的训练损失和较低的测试精度反映了训练如此重的模型的实际难度==。注意，在图中，我们展示了两种方法在最优超参数设置下的训练过程。**虽然增加NGCF的学习率可以减少其训练损失(甚至低于LightGCN)，但测试召回率并不能提高，因为用这种方式降低训练损失只能为NGCF找到简单的解决方案。**

![image-20220113180737629](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113180737629.png)

base中mult-VAE最好（居然好于NGCF？），不同数据集中还是有差别的

**GRMF 添加norm后在yelp和amazon上并没有改善**

LightGCN可以通过调参α获取更好的效果，这里α取的是1/(1+k)

### 消融实验分析

 ***Impact of Layer Combination（层数组合的影响）***

- 随着层数从第1个到第4个。在大多数情况下，峰值点在第2层，而之后它迅速下降到第4层的最差点。这表明，用一阶和二阶邻居平滑节点的嵌入对于CF非常有用，但在使用高阶邻居时，会出现过平滑问题。
- 针对LightGCN，我们发现其性能随着各层数的增加而逐渐提高。即使使用4层，LightGCN的性能也不会下降。这证明了层组合解决过平滑的有效性。

![image-20220113181337727](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113181337727.png)

 ***Impact of Symmetric Sqrt Normalization（归一化系数的影响）***

***Analysis of Embedding Smoothness（embedding平滑度分析）***

![image-20220113181641293](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113181641293.png)

![image-20220113181813198](https://raw.githubusercontent.com/leelige/upic/main/picgo/image-20220113181813198.png)

越小越好，但不能过小，会造成过平滑

## future work

最近的一个趋势是利用项目知识图谱、社交网络和多媒体内容等**辅助信息**进行推荐，其中gcn已经建立了新的最先进的技术。然而，这些模型也可能存在类似的NGCF问题，因为用户-项目交互图也是由可能**不必要的相同的神经操作建模的**。

我们计划在这些模型中探索LightGCN的想法。**另一个未来的方向**<u>是个性化层组合权重αk，以便为不同的用户实现自适应顺序平滑（例如，稀疏用户可能需要来自高阶邻居的更多信号，而活动用户需要的更少）</u>。最后，我们将进一步探索LightGCN的简单性的优势，研究是否存在针对**非抽样回归损失(non-sampling regression loss)**^[1]^的快速解决方案，并对在线工业场景优化。



参考文献：

[1]Xiangnan He, Jinhui Tang, Xiaoyu Du, Richang Hong, Tongwei Ren, and Tat-Seng Chua. 2019. Fast Matrix Factorization with Nonuniform Weights on Missing Data. *TNNLS* (2019).