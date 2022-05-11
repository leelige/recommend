# article

* [GNN](#GNN)

* [RecGNN](#recGNN)

* [Multi interest](#Multi-interest)

* [Group](#Group)

* [Collaborative Filtering](#Collaborative-Filtering)

* [Learning to Rank](#learning-to-rank)

##  GNN


|      | title |author |organization|conference  | note |
| :----: |  :----------------------------------------------------------: | :--------: |:--------: |:-------: | :-------: |
| 1    |       [node2vec: Scable Featureds Learning for Networks](https://github.com/leelige/recommend/blob/master/GNN/01Node2vec.pdf)    | Jure Leskovec  | Stanford University | KDD 2016  |  基于随机游走的graph embedding,对BFS和DFS进行trade-off,是对Deepwalk的改进 |
| 2    |       [LINE：Large-scale Information Network Embedding](https://github.com/leelige/recommend/blob/master/GNN/02LINE.pdf)   |   Jian Tang  | MSRA | WWW 2015  | 基于随机游走的graph embedding，本质上是BFS，实现1阶和2阶的相似，后来node2vec性能优于它，并拓展了DFS|
| 3    |              [Structural Deep Network Embedding](https://github.com/leelige/recommend/blob/master/GNN/03SDNE.pdf)            | Daixin Wang| Tsinghua  |  KDD 2016  |          使用auto-encoder优化一阶和二阶相似性       |
| 4    | [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://github.com/leelige/recommend/blob/master/GNN/04metapath2vec.pdf)| Yuxiao Dong | Microsoft Rearch|  KDD 2017   |  基于随机游走的graph embedding，实现异构图上skip-gram和负采样|
| 5    |  [Translating Embeddings for Modeling Multi-relational Data](https://github.com/leelige/recommend/blob/master/GNN/05TransE.pdf)  |Antoine Bordes |CNRS| NIPS 2013  |   知识图谱经典方法，将实体和关系抽象成矩阵计算相似性，并构造知识图谱中的三元组  |
| 6    | [Semi-supervised Classification with Graph Convolutional Networks](https://github.com/leelige/recommend/blob/master/GNN/08GCN.pdf)|  Thomas N.Kipf  |University of Amsterdam| ICLR 2017  |           new，GNN思想的里程碑式文章，严格证明了拉普拉斯矩阵的构造，联系了基于频域和空域的图神经网络    |
| 7    |      [Inductive Representation Learning on Large Graphs](https://github.com/leelige/recommend/blob/master/GNN/07GraphSage.pdf)    |    William & Jure Leskovec  |Stanford| NIPS 2017           |   GraphSage，改进GCN在归纳式学习(transductive Learning)上的缺陷        |
| 8    |                   [Graph Attention Network](https://github.com/leelige/recommend/blob/master/GNN/06GAT.pdf)              |    Petar Velickovic & Yoshua Bengio| University of Cambridge | ICLR 2018  |                 在直推式和归纳式学习上都取得较好的效果    |
| 9    |             [Gated Graph Sequence Neural Networks](https://github.com/leelige/recommend/blob/master/GNN/09GGNN.pdf)       |   Yujia Li| University of Toronto   | ICLR 2016  |                  将GRU思想应用于GNN，实现GNN上的序列模型应用  |
| 10   |         [Neural Message Passing for Quantum Chemistry](https://github.com/leelige/recommend/blob/master/GNN/10MPNN.pdf)   |    Justin Gilmer| Google Brain  | ICML 2017  |               GNN框架，集成了多种GNN模型，在分子化学上实现较好的效果     |



##  RecGNN

|      | title | author| organization | conference | note |
| :----: |  :----------------------------------------------------------: | :--------: | :-------: |:-------: | :-------: |
| 1    | [Neural Graph Collaborative Filtering](https://github.com/leelige/recommend/blob/master/recGNN/Neural%20Graph%20Collaborative%20Filtering.pdf) | Xiangnan He |   USTC|  SIGIR 2019  | [reading note](https://github.com/leelige/recommend/blob/master/recGNN/note/Neural%20Graph%20Collaborative%20Filtering.md) |
| 2 | [LightGCN: Simplifying and Powering Graph Convolution](https://github.com/leelige/recommend/blob/master/recGNN/LightGCN.pdf) | Xiangnan He | USTC | SIGIR 2020 |   [reading note](https://github.com/leelige/recommend/blob/master/recGNN/note/LightGCNSimplifying%20and%20Powering%20Graph%20Convolution%20Network%20for%20Recommendation.md) |


## Multi interest

|      |                            title                             |    author    |     organization      | conference  |                             note                             |
| :--: | :----------------------------------------------------------: | :----------: | :-------------------: | :---------: | :----------------------------------------------------------: |
|  1   | [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://github.com/leelige/recommend/blob/master/multi/Multi-Interest%20Network%20with%20Dynamic%20Routing%20for%20Recommendation%20at%20Tmall.pdf) |   Li Chao    |        Alibaba        |  KDD 2019   | capsule network first used in diversity recommendation(作者对dynamic routine 做了改进)   **sequential rec** |
|  2   | [Controllable Multi-Interest Framework for Recommendation](https://github.com/leelige/recommend/blob/master/multi/Controllable%20Multi-Interest%20Framework%20for%20Recommendation.pdf) |  Yukuo Cen   |        Alibaba        |  KDD 2020   | [capsule network](https://github.com/leelige/recommend/blob/master/extend/Dynamic%20Routine%20between%20capsules.pdf)(Dynamic Routine)   [pytorch_code](https://github.com/leelige/recommend/tree/master/code/pytorch_ComiRec)       **sequential rec** |
|  3   | [A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks]( https://github.com/leelige/recommend/blob/master/multi/A%20Framework%20for%20Recommending%20Accurate%20and%20Diverse%20Items%20Using%20Bayesian%20Graph%20Convolutional%20Neural%20Networks.pdf) | Jianing Sun  | Huawei Noah’s Ark Lab |  KDD 2020   |                Bayesian method(node copying)                 |
|  4   | [Dynamic Graph Construction for Improving Diversity of Recommendation](https://github.com/leelige/recommend/blob/master/multi/Dynamic%20Graph%20Construction%20for%20Improving%20Diversity%20of%20Recommendation.pdf) |    Rui Ye    |        Meituan        | RecSys 2021 |                         graph extend                         |
|  5   | [A Hybrid Bandit Framework for Diversifified Recommendation](https://github.com/leelige/recommend/blob/master/multi/A%20Hybrid%20Bandit%20Framework%20for%20Diversified%20Recommendation.pdf) |  Qinxu Ding  | Alibaba-NTU[南洋理工] |  AAAI 2021  |                                                              |
|  6   | [Enhancing Domain-Level and User-Level Adaptivity in Diversified Recommendation](https://github.com/leelige/recommend/blob/master/multi/Enhancing%20Domain-Level%20and%20User-Level%20Adaptivity%20in%20Diversified%20Recommendation.pdf) |  Yile Liang  |  university of Wuhan  | SIGIR 2021  |                                                              |
|  7   | [PD-GAN: Adversarial Learning for Personalized Diversity-Promoting Recommendation](https://github.com/leelige/recommend/blob/master/multi/PD-GAN-Adversarial%20Learning%20for%20Personalized%20Diversity-Promoting%20Recommendation.pdf) |   Qiong Wu   |      Alibaba-NTU      | IJCAI 2019  |                                                              |
|  8   | [Future-Aware Diverse Trends Framework for Recommendation](https://github.com/leelige/recommend/blob/master/multi/Future-Aware%20Diverse%20Trends%20Framework%20for%20Recommendation.pdf) |   Yujie Lu   |      Tecent-ZJU       |  WWW 2021   |                                                              |
|  9   | [Sliding Spectrum Decomposition for Diversified](https://github.com/leelige/recommend/blob/master/multi/Sliding%20Spectrum%20Decomposition%20for%20Diversified%20Recommendation.pdf) | Yanhua Huang |        小红书         |  KDD 2021   |                                                              |
|  10  | [On the Diversity and Explainability of Recommender Systems:A Practical Framework for Enterprise App Recommendation](https://github.com/leelige/recommend/blob/master/multi/On%20the%20Diversity%20and%20Explainability%20of%20Recommender%20Systems_A%20Practical%20Framework%20for%20Enterprise%20App%20Recommendation.pdf) | Wenzhuo Yang |  SalesForce Research  |  CIKM 2021  |                     application(special)                     |

## Group

|      |                            title                             |     author     |                organization                |   conference    |             note              |
| :--: | :----------------------------------------------------------: | :------------: | :----------------------------------------: | :-------------: | :---------------------------: |
|  1   | [Attentive Group Recommendation](https://github.com/leelige/recommend/blob/master/group/attentive%20group%20recommendation.pdf) |     Da Cao     |            university of Hunan             |   SIGIR 2018    | group recommendation baseline |
|  2   | [GAME: Learning Graphical and Attentive Multi-view Embeddings for Occasional Group Recommendation](https://github.com/leelige/recommend/blob/master/group/GAME%20Learning%20Graphical%20and%20Attentive%20Multi-view.pdf) |  Zhixiang He   |        City University of Hong Kong        |   SIGIR 2020    |                               |
|  3   | [GroupIM: A Mutual Information Maximization Framework for Neural Group Recommendation](https://github.com/leelige/recommend/blob/master/group/A%20Mutual%20Information%20Maximizing%20Framework%20for%20Neural%20Group%20Recommendation.pdf) | Aravind Sankar | University of Illinois at Urbana-Champaign |   SIGIR 2020    |         [reading note](https://github.com/leelige/recommend/blob/master/group/note/GroupIM%20A%20Mutual%20Information%20Maximization%20Framework%20for%20Neural%20Group%20Recommendation.md)                      |
|  4   | [Hierarchical Hyperedge Embedding-Based Representation Learning for Group Recommendation](https://github.com/leelige/recommend/blob/master/group/Hierarchical%20Hyperedge%20Embedding-Based%20Representation.pdf) |    Lei Guo     |     Shandong Normal University, China      | ACM Transaction 2021 |                               |
|  5   | [Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation](https://github.com/leelige/recommend/blob/master/group/Double-Scale%20Self-Supervised%20Hypergraph%20Learning%20for%20Group%20Recommendation.pdf) |   Junwei Zhang     |     Chong Qing University, China      | CIKM 2021 |                               |


## Collaborative-Filtering

|      |                            title                             |     author     |                organization                |   conference    |             note              |
| :--: | :----------------------------------------------------------: | :------------: | :----------------------------------------: | :-------------: | :---------------------------: |
| 1 |                                [one-class collaborative filtering](https://github.com/leelige/recommend/blob/master/Collaborative%20Filtering/one-class%20collaborative%20filtering.pdf) |    潘嵘    |     中山大学(现), HP LAB    |    ICDM 2008   |    讨论协同过滤**负采样**技术的开山之作，难点在于如何从共现矩阵中提取负样本，论文在这一点上并没有清晰说明，但论文对于missing value的讨论具有借鉴意义 |


## Learning to Rank
|      |                            title                             |     author     |                organization                |   conference    |             note              |
| :--: | :----------------------------------------------------------: | :------------: | :----------------------------------------: | :-------------: | :---------------------------: |
| 1 |                                [A Short Introduction to Learning to Rank](https://github.com/leelige/recommend/blob/master/ranking/a%20short%20introduction%20to%20learning%20to%20rank.pdf) |    李航    |     MRSA    |    IEICE TRANSACTIONS on Information and Systems 2011   |   LTR（Learning torank）学习排序是一种监督学习（SupervisedLearning）的排序方法。LTR已经被广泛应用到文本挖掘的很多领域，比如IR中排序返回的文档，推荐系统中的候选产品、用户排序，机器翻译中排序候选翻译结果等等 |
| 2 |                                [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://github.com/leelige/recommend/blob/master/ranking/Bayesian%20Personalized%20Ranking.pdf) |    Steffen Rendle   |     University of Hildesheim  |    UAI 2009 (CCF B 会议)  |  对级排序算法(pair-wise ranking)  |

