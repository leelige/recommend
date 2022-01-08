# article

* GNN
* Rec GNN

## GNN


|      | title |conference | note |
| :----: |  :----------------------------------------------------------: | :--------: | :-------: |
| 1    |       [node2vec: Scable Featureds Learning for Networks](https://github.com/leelige/recommend/blob/master/GNN/01Node2vec.pdf)       |  KDD 2016  | 基于随机游走的graph embedding,对BFS和DFS进行trade-off,是对Deepwalk的改进 |
| 2    |       [LINE：Large-scale Information Network Embedding](https://github.com/leelige/recommend/blob/master/GNN/02LINE.pdf)        |  WWW 2015  | 基于随机游走的graph embedding，本质上是BFS，实现1阶和2阶的相似，后来node2vec性能优于它，并拓展了DFS|
| 3    |              [Structural Deep Network Embedding](https://github.com/leelige/recommend/blob/master/GNN/03SDNE.pdf)               |  KDD 2016  |    使用auto-encoder优化一阶和二阶相似性       |
| 4    | [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://github.com/leelige/recommend/blob/master/GNN/04metapath2vec.pdf) |  KDD 2017  |  基于随机游走的graph embedding，实现异构图上skip-gram和负采样|
| 5    |  [Translating Embeddings for Modeling Multi-relational Data](https://github.com/leelige/recommend/blob/master/GNN/05TransE.pdf)   | NIPS 2013  | 知识图谱经典方法，将实体和关系抽象成矩阵计算相似性，并构造知识图谱中的三元组  |
| 6    | [Semi-supervised Classification with Graph Convolutional Networks](https://github.com/leelige/recommend/blob/master/GNN/08GCN.pdf) | ICLR 2017  |    new，GNN思想的里程碑式文章，严格证明了拉普拉斯矩阵的构造，联系了基于频域和空域的图神经网络    |
| 7    |      [Inductive Representation Learning on Large Graphs](https://github.com/leelige/recommend/blob/master/GNN/07GraphSage.pdf)       | NIPS 2017  |   GraphSage，改进GCN在归纳式学习(transductive Learning)上的缺陷        |
| 8    |                   [Graph Attention Network](https://github.com/leelige/recommend/blob/master/GNN/06GAT.pdf)                    | ICLR 2018  |       在直推式和归纳式学习上都取得较好的效果    |
| 9    |             [Gated Graph Sequence Neural Networks](https://github.com/leelige/recommend/blob/master/GNN/09GGNN.pdf)             | ICLR 2016  |         将GRU思想应用于GNN，实现GNN上的序列模型应用  |
| 10   |         [Neural Message Passing for Quantum Chemistry](https://github.com/leelige/recommend/blob/master/GNN/10MPNN.pdf)         | ICML 2017  |      GNN框架，集成了多种GNN模型，在分子化学上实现较好的效果     |



## Rec GNN

|      | title | author| conference | note |
| :----: |  :----------------------------------------------------------: | :--------: | :-------: | :-------: |
| 1    |       [Neural Graph Collaborative Filtering]()    | 何向南 |  SIGIR 2019  | |
