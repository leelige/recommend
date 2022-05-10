# one-class collaborative filtering-负采样

**author: 潘嵘(中山大学)**

**organization:  HP Lab**

**conference:  ICDM 2008**

[TOC]

**单类协同过滤(OCCF)**

```apl
什么是单类协同过滤(OCCF)?

单类反馈数据是指用户点击的情况，比如我们看新闻，听音乐等，用户浏览过或者点击过的内容代表用户读了或听了，是用户喜欢的正反馈数据，但还有大量的数据是用户未点击的。
```

许多协同过滤(CF)的应用，如新闻项目推荐和书签推荐，最自然地被认为是**单类协同过滤(OCCF)问题**。在这些问题中，训练数据通常只是由反映用户的**交互或不交互的二元数据组成**。通常这类数据是**非常稀疏的**(一小部分是正样本)，因此在负样本的例子的解释中出现了歧义。 **负样本和无标签的正样本混合在一起，我们通常无法区分它们**。

==我们不能将无标签的正样本当作负样本==。本文研究了协同过滤下的单类问题 。我们提出了两个框架来解决OCCF的问题。

* **基于加权低秩近似 (weighted low rank approxiamtion)**

* **基于负采样 (negative example sampling)**

## Introduction

从提供搜索结果到产品推荐，个性化服务在网络上正变得越来越不可或缺。这些系统中使用的核心技术是**协同过滤(CF)**，它旨在根据所有用户之前评估的项目来预测特定用户的项目偏好。用户给出不同的分数表示评分(比如netflix中的1-5分)。然而，在许多其他情况下，它也可以通过用户的行为隐式地表示出来，**如点击或不点击，添加书签或不添加书签**。这些形式的**隐式评级**更常见，也更容易获取。

虽然优点很明显，但隐式评级的一个缺点：**特别是在数据稀疏的情况下，是很难识别可表征的负样本**。<u>所有的负样本和未交互的正样本混合在一起，无法区分。</u>我们将只给出的正样本的协同过滤称为单类协同过滤(OCCF)。OCCF发生在不同的场景中，下面有两个例子。

* Social Bookmarks：在这样的系统中，每个用户 **收藏(bookmark)** 一组网页，这些网页可以被视为**用户兴趣的正样本**。 但是，对于用户没有收藏一个网页的行为，可以有两种可能的解释。==第一个是，这个页面是用户感兴趣的，但之前没有看到这个页面；第二个是用户看到了这个页面，但不感兴趣。==我们不能假设所有不在他书签中的页面都是**负样本**。
* Clickthrough History：点击数据被广泛用于个性化搜索和搜索结果的改进。通常一个三元组**<u，q，p>**表示一个 **用户(user): u** 提交了一个**查询(query): q** 并点击了一个 **页面(page): p**。通常不会收集未被点击的页面。与书签示例类似，我们无法判断页面是否因为其内容无关或冗余而不被点击。

有几种直观的策略可以解决这个问题。一种方法是标记负样本，将数据转换为一个经典的CF问题。但这样代价过大，特别是对于用户来说。事实上，用户很少提供传统学习算法所需的评分，特别是负样本。此外，根据一些用户研究表明，如果一个用户被要求在系统表现良好之前提供许多正反馈和负反馈，她将不愿再使用该系统。

另一个常见的解决方案是**将所有缺失的数据视为负样本**。根据经验，这个解决方案效果很好（见第4.6节）。缺点是它会使推荐结果产生偏差，因为一些缺失的数据可能是正样本。

另一方面，如果我们将缺失视为未知，即忽略所有缺失的例子，只利用正的例子，然后将其输入只建模非缺失数据的CF算法，由这种方法产生的一个简单的解决方案是，所有对**缺失值的预测都是正样本**。

==因此，**All Missing as Negative (AMAN)** 和 **All Missing as Unknown (AMAU)** 是OCCF中的两种极端策略。==

在本文中，我们考虑如何平衡将缺失值作为负样本的程度。我们提出了两种可能的OCCF解决方法。这些方法允许我们在输入中调整权衡 改进所谓的负样本，得到更好的CF算法。

1. 第一种方法是基于加权低秩近似

2. 第二个方法是基于负采样

它们都利用未知数据中包含的信息，纠正将它们作为负样本的偏见。不仅基于加权的方法解决了这个问题，而且基于采样的方法对大规模稀疏数据集以更低的计算成本近似精确解。

在OCCF问题中，我们提出的解决方案显著优于两个极端情况(AMAN和AMAU)，在我们的实验中，它比最佳基线方法至少提高了8%。此外，我们经验表明，这两种提出的OCCF解决方案框架（基于加权和采样）具有**几乎相同的性能**。

## 相关工作

### 协同过滤(collaborative filtering)

在过去，许多研究人员从不同的方面探索了协同过滤(CF)，从提高算法的性能到从异构资源中整合更多的数据源。然而，之前关于协同过滤的研究仍然假设我们有积极的（高评分）和消极的（低评分）的例子。在非二元结构数据的情况下，项目(item)采用评分方案进行评分(上面Netflix评分机制)。以前的大多数工作都集中在这个问题的设置上。**在所有的CF问题中，都有很多缺少评级的例子**。在 `Spectral analysis of data,STOC 2001`和`Collaborative fifiltering and the missing at random assumption, UAI 2007`中，作者讨论了对协同过滤问题中缺失值的分布进行建模。它们都不能处理没有负样本的情况。

在二元结构的情况下，每个样本要么是正的，要么是负的。Das等人[^ Google news personalization: scalable online collaborative fifiltering , WWW 2007]研究了新闻推荐，点击新闻报道是正的，不点击表示负的。作者比较了关于这个大规模二元CF问题的一些实际方法。2007年的KDD杯举办了一个“Who rated What”的推荐任务，训练数据与Netflix的奖励数据集相同（带有评分）。获胜者团队提出了一种利用二元训练数据结合SVD和流行度的混合方法。

### 单分类(one-class classification)

针对二元分类问题，已经有提出从正样本中学习的算法。一些研究解决了同时使用无标签样本和正样本的问题。对于单类SVM，该模型描述了单个类，仅从**正样本学习**。这种方法类似于密度估计。当无标签的样本可用时，解决单类分类问题的一个策略是使用EM类算法迭代地预测负样本并学习分类器。也有一部分研究[^PAC learning from positive statistical queries. ALT 1998]将正样本以一定概率变为无标签样本，通过统计概率模型学习正样本和无标签的样本

### 类不平衡问题(class imbalance problem)

我们的工作还与类不平衡问题有关，这种问题通常发生在分类任务中。单类问题可以看作是类不平衡问题的一个极端情况。采用了两种策略来解决类不平衡问题。一个是在数据级别。其想法是使用采样来重新平衡数据。另一个 一个是在算法层面上，使用cost-sensitive。这两种策略的比较可以在[^Extreme re-balancing for svms: a case study. SIGKDD 2004 ]中找到。

## 基于加权和采样的方法

如上所述，AMAN和AMAU是协同过滤的两种通用策略，它们可以被认为是两个极端。我们认为，在OCCF问题中，可能有一些方法可以优于这两种策略；例子包括“所有缺失为弱负”或“一些缺失为负”。其思想是对目标函数中正样本和负样本的误差项给予不同的权重；第二个是将一些缺失的值作为负样本进行采样，本文介绍了一些采样策略。

### 问题定义

假设我们有m个用户和n个项目，并且之前的查看信息存储在一个矩阵 **R** 中。**R** 的元素取值 1，这代表一个正样本，**‘ ？‘**表示一个未知的（缺失的）正样本或负样本。我们的任务是从基于 **R** 的缺失数据中识别出潜在的正样本，我们将其称为单类协同过滤(OCCF)。请注意，在本文中，我们假设除了 **R** 之外，我们没有关于用户和项的额外信息。在本文中，我们用粗体的大写字母来表示一个矩阵。

给定一个矩阵 $A$，$A_{ij}$ 表示它的元素，$A_i$ 表示 $A$ 的第 $i$ 行， $A_j$ 表示 $A$ 的第 $j$ 列，$A^T$ 表示 $A$ 的转置 

### wALS(weighted alternating least squares, 加权交替最小二乘)

我们解决单类协同过滤问题的第一种方法是基于加权低秩近似 (**wLRA**, `Weighted low-rank approximations, ICML 2003`) 技术。加权低秩近似 **(wLRA)** 应用于CF问题，其朴素加权方案将“1”分配给观察到的例子，“0”分配给缺失（未观察到的）值，这是对应的到AMAU。另一种方式是将所有无标签值视为负样本。然而，由于缺失值有正样本，这种处理方法可能会造成误差。我们通过在误差项上使用较低的权重来解决这个问题。

* **给定一个矩阵** $\boldsymbol{R}=\left(R_{i j}\right)_{m \times n} \in\{0,1\}^{m \times n}$**，表示有m个users，n个items**

* **给定一个非负权值矩阵** $\boldsymbol{W}=\left(W_{i j}\right)_{m \times n} \in \mathfrak{R}_+^{m \times n}$
* **用低秩矩阵$X=(X_{ij})_{m×n}$ 近似 $\boldsymbol{R}$**

**目标函数：**

![image-20220504161038186](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504161038_image-20220504161038186.png)

在OCCF中，我们为正样本设置了$R_{ij}=1$；==**对于缺失的值，我们假设它们中的大多数都是负样本**，**并设置**$R_{ij}=0$==。

由于我们对 $R_{ij}=1$ 的正样本有很高的置信度，我们设置其对应的$W_{ij}=1$，在 $R_{ij}=0$ 的位置设置对应的$W_{ij}\in[0,1]$

在讨论“负”样本的加权方案之前，我们展示了如何有效地解决优化问题 $argmin_X\mathcal{L}(X)$。

**考虑矩阵分解：**$X=UV^T$    $U\in \mathfrak{R}^{m\times d} \  \ and \ \ V\in \mathfrak{R}^{n\times d}$，$d$ 表示特征个数 **(numbers of features)**，$d《 \ r \ , \ r \approx min(m,n)是矩阵R的秩$

![image-20220504175955933](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504175956_image-20220504175955933.png)

为了防止过拟合(overfitting)，添加正则项：

![image-20220504180658434](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504180658_image-20220504180658434.png)

![image-20220504180643641](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504180643_image-20220504180643641.png)

λ是一个正则化参数，在实际问题中，它是通过**交叉验证**来确定的。**Eq(4)** 包含了中正则化低秩近似的特殊情况。

**Eq(4)梯度下降求偏导：**

![image-20220504181630297](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504181630_image-20220504181630297.png)

![image-20220504181816035](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220504181816_image-20220504181816035.png)

$\widetilde{W_{i\cdot}} \in \mathfrak{R}^{n \times n}$，是对角矩阵，对角线上的值是权值矩阵 $W$ 第 $i$ 行的值，即 $W_{i\cdot}$ 

$I$ 是单位矩阵

**对 U 来说：**

![image-20220509175813950](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220509175814_image-20220509175813950.png)

**对 V 来说：**

![image-20220505134659525](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220505134659_image-20220505134659525.png)

$\widetilde{W_{\cdot j}} \in \mathfrak{R}^{n \times n}$，是对角矩阵，对角线上的值是权值矩阵 $W$ 第 $j$ 列的值，即 $W_{\cdot j}$ 

#### Weighting Schemes: Uniform, User Oriented, and Item Oriented (加权方案)

**一共有三种加权方案：**

* **所有missing data** 都以概率 $\delta \in [0,1]$ 成为负样本
* **(user oriented)** 如果一个用户有很多正交互，那么missing data为负样本的概率很大
* **(item oriented)** 如果一个项目有很少的正交互，那么missing data为负样本的概率很大

![image-20220505164306868](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220505164307_image-20220505164306868.png)

### OCCF的基于采样的ALS

正如我们上面所述，对于单类协同过滤，一个简单的策略是假设所有缺失的值都是负的。这种关于大多数缺失值为负的隐含假设在大多数cases中大致成立。然而，这里的主要缺点是，当评级矩阵R的大小**较大时**，**计算成本非常高**。**wALS** 也有同样的问题。

采样策略分为两个步骤：

* 采样负样本

* 重构R

![image-20220506082326004](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220506082326_image-20220506082326004.png)



#### 采样负样本 ( 重要! ）

其实和 **wALS** 类似，只不过权值赋值改为采样

* Uniform Random Sampling (均匀分布随机采样)：$\widehat{P}_{i j} \propto 1$，所有missing data以同一概率变为负样本
* User-Oriented Sampling：$\widehat{P}_{i j} \propto \sum_i I[R_{ij}=1]$，如果一个用户有很多正交互，那么missing data为负样本的概率很大
* Item-Oriented Sampling：$\widehat{P}_{i j} \propto 1/\sum_j I[R_{ij}=1]$，如果一个项目有很少的正交互，那么missing data为负样本的概率很大



**Matlab code** （negative sample）

```matlab
function [m] = sample_trainer_new(R, dim_of_R, repeats, scheme, options)

if nargin < 3
	fprintf('m = sample_trainer_new(R, dim_of_R, repeats [,scheme="%s", option="%s"])\n',...
			'user', '-s 1 -t 20 -l 0.1 -k 8 -b 1');
	fprintf(' schemes: user/item_f/item_w/item_s/uniform/');
	fprintf('Use sample_trainer(R, dim_of_R, 0, "", option="-s solvers -b 0 other_options") to perform other pu solvers\n');
	return 
end
if nargin < 4
	scheme = 'user';
end
if nargin < 5
	options = '-s 1 -t 20 -l 0.1 -k 8 -b 1';
end

users = dim_of_R(1);
items = dim_of_R(2);

% 注意：MATLAB下标从1开始
if issparse(R)       									% 判断是否为稀疏矩阵
	[I J V] = find(R); 									% [行下标,列下标,值]=find(R), 返回非0值, I,J,K都是列向量
	idx = find(I<=users & J <= items);  % I 中小于users的索引，J 中小于items的索引，都非0
	R = [I(idx) J(idx) V(idx)];         
end

%%%
[I J V] / R
---------
1   1   1
2   1   1
4   1   1
2   2   1
3   2   1
6   2   1
3   3   1
1   4   1
3   4   1
---------
%%%
nr_samples = repeats * size(R,1);     % 负采样个数——repeats倍的正样本
if strcmp(scheme,'user')
	newI = repmat(R(:,1),  repeats, 1); % 只选择正样本，repmat(A,a,b) 矩阵A先按行扩展a倍，再按列扩展b倍, newI_shape=(repeats*R(:,1),1)=(nr_samples,1)
	population = [1:items]'; 
	weights = ones(size(population))/length(population);  % weight矩阵是常数
	newJ = randsample([1:items]', nr_samples, true); 
elseif strcmp(scheme,'item_f')
	newI = repmat(R(:,1),  repeats, 1);
	population = [1:items]'; 
	weights = histc(R(:,2), population); weights = weights / sum(weights); % 另一种理解：如果一个交互丰富的user和热门item没有交互，那么大概率为负
	newJ = randsample(population, nr_samples, true, weights);
elseif strcmp(scheme,'item_w')
	newI = randsample([1:users]', nr_samples, true);
	population = [1:items]'; 
	weights = histc(R(:,2), population); weights = users - weights; weights = weights / sum(weights); %item-oriented
	newJ = randsample(population, nr_samples, true, weights);
elseif strcmp(scheme,'item_s')
	newI = randsample([1:users]', nr_samples, true);
	population = [1:items]'; 
	weights = histc(R(:,2), population); weights(weights==0)=-1;weights = 1./weights; weights(weights<0) = 0;  
	weights = weights / sum(weights);  % 排除交互为0的item的影响，即采样时不需要考虑它，即’稀疏‘R的影响
  newJ = randsample(population, nr_samples, true, weights);
elseif strcmp(scheme, 'uniform')
	population = [1:items]';
	weights = ones(size(population))/length(population);
	newI = randsample([1:users]', nr_samples, true);
	newJ = randsample([1:items]', nr_samples, true);
else
	population = [1:items]';
	weights = ones(size(population))/length(population);
	newI = [];
	newJ = [];
	nr_samples = 0;
end

newV = zeros(nr_samples,1);
newR = [R; [newI newJ newV]];

m = pmf_train(newR, [], options); % []表示处于训练中，如果不是[],则表示为测试集
m.population = population;
m.weights = weights;
m.scheme = scheme;
end

```

**ALS C++ code**

```c++
int run_als(mxArray *plhs[], int nrhs, const mxArray *prhs[], pmf_parameter_t &param) { // {{{
	mxArray *mxW, *mxH;
	smat_t training_set, test_set;

	size_t tmp_rows = nrhs>3? mxGetM(prhs[2]): 0;  //mxGetM获得数组行数
	size_t tmp_cols = nrhs>3? mxGetM(prhs[3]): 0;  //mxGetM获得数组行数
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(prhs[0], training_set, tmp_rows, tmp_cols);
	mxArray_to_smat(prhs[1], test_set, training_set.rows, training_set.cols);

	// fix random seed to have same results for each run
	// (for random initialization)
	long seed = 0L;
	// ALS requires rowmajor model
	pmf_model_t model(training_set.rows, training_set.cols, param.k, pmf_model_t::ROWMAJOR);
	mat_t& W = model.W, &H = model.H;

	// Initialization of W and H
	if(nrhs >= 4) {
		mxDense_to_matRow(prhs[2], W);
		mxDense_to_matRow(prhs[3], H);
	} else {
		model.rand_init(seed);
	}

	if(param.remove_bias) {
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
	}

	// Random permutation for rows and cols of training_set for better load balancing
	std::vector<unsigned> row_perm, inverse_row_perm;
	std::vector<unsigned> col_perm, inverse_col_perm;
	if(do_shuffle) {
		gen_permutation_pair(training_set.rows, row_perm, inverse_row_perm);
		gen_permutation_pair(training_set.cols, col_perm, inverse_col_perm);

		training_set.apply_permutation(row_perm, col_perm);
		test_set.apply_permutation(row_perm, col_perm);
		if(nrhs >= 4)
			model.apply_permutation(inverse_row_perm, inverse_col_perm);
	}

	// Execute the program
	double time = omp_get_wtime();
	if(param.solver_type == ALS)
		als(training_set, test_set, param, model);
	else if(param.solver_type == PU_ALS)
		als_pu(training_set, test_set, param, model);
	double walltime = omp_get_wtime() - time;

	if(do_shuffle) // recover the permutation for the model
		model.apply_permutation(row_perm, col_perm);

	// Write back the result
	plhs[0] = pmf_model_to_mxStruture(model);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*(mxGetPr(plhs[1])) = walltime;

	return 0;
} // }}}
```



**pmf_matlab.hpp**

```c
#include "mex.h"
#include "../pmf.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

typedef entry_t<val_type> ENTRY_T;
#define entry_t ENTRY_T

// Conversion Utility {{{
int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}
// convert matlab sparse matrix to C smat fmt
class mxSparse_iterator_t: public entry_iterator_t<val_type> {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols);  // nnz：非零项总个数
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		entry_t next() {
			while (cur_idx >= jc_t[cur_row+1])
				++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			entry_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt); 
		}

};           

// convert matlab Coo matrix to C smat fmt
class mxCoo_iterator_t: public entry_iterator_t<val_type> {
	private:
		double *row_idx, *col_idx, *val;
		size_t cur_idx;
		bool good_entry(size_t row, size_t col) {
			return 1 <= row && row <= rows && 1 <= col && col <= cols;
		}
		void init(double *_row_idx, double *_col_idx, double *_val, size_t _nnz, size_t _rows, size_t _cols) { // {{{
			row_idx = _row_idx; col_idx = _col_idx; val = _val; nnz = _nnz; rows = _rows; cols = _cols;
			cur_idx = 0;
			if(_rows == 0 && _cols == 0) {
				for(size_t idx = 0; idx < nnz; idx++) {
					if((size_t)row_idx[idx] > rows) rows = (size_t) row_idx[idx];
					if((size_t)col_idx[idx] > cols) cols = (size_t) col_idx[idx];
				}
			} else { // filter entries with out-of-range row/col indices
				for(size_t idx = 0; idx < nnz; idx++) {
					size_t row = (size_t) row_idx[idx], col = (size_t) col_idx[idx];
					if(!good_entry(row,col))
						nnz--;
				}
			}
		} // }}}

	public:
		size_t rows, cols;
		mxCoo_iterator_t(const mxArray *M, size_t _rows, size_t _cols) {
			double *data = mxGetPr(M);
			size_t _nnz = mxGetM(M);
			init(&data[0], &data[_nnz], &data[2*_nnz], _nnz, _rows, _cols);
		}
		mxCoo_iterator_t(double *_row_idx, double *_col_idx, double *_val, size_t _nnz, size_t _rows, size_t _cols) {
			init(_row_idx, _col_idx, _val, _nnz, _rows, _cols);
		}
		entry_t next() {
			size_t row = 0, col = 0;
			while(1) {
				row = (size_t) row_idx[cur_idx];
				col = (size_t) col_idx[cur_idx];
				if(good_entry(row, col))
					break;
				cur_idx++;
			}
			entry_t ret(row-1, col-1, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

// convert matlab Dense column-major matrix to C smat fmt
class mxDense_iterator_t: public entry_iterator_t<val_type> {
	private:
		size_t cur_idx;
		double *val;
	public:
		size_t rows, cols;
		mxDense_iterator_t(const mxArray *mxM): rows(mxGetM(mxM)), cols(mxGetN(mxM)), val(mxGetPr(mxM)){
			cur_idx = 0; nnz = rows*cols;
		}
		entry_t next() {
			entry_t ret(cur_idx%cols, cur_idx/cols, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

template<class T>
void mxSparse_to_smat(const mxArray *M, T &R) {
	size_t rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols);
	mxSparse_iterator_t entry_it(M);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
}

template<class T>
void mxCoo_to_smat(const mxArray *mxM, T &R, size_t rows=0, size_t cols=0) {
	mxCoo_iterator_t entry_it(mxM, rows, cols);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxCoo_to_smat(double *row_idx, double *col_idx, double *val, size_t nnz, T &R, size_t rows=0, size_t cols=0) {
	mxCoo_iterator_t entry_it(row_idx, col_idx, val, nnz, rows, cols);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxDense_to_smat(const mxArray *mxM, T &R) {
	mxDense_iterator_t entry_it(mxM);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxArray_to_smat(const mxArray *mxM, T &R, size_t rows=0, size_t cols=0) {
	if(mxIsDouble(mxM) && mxIsSparse(mxM))
		mxSparse_to_smat(mxM, R);
	else if(mxIsDouble(mxM) && !mxIsSparse(mxM)) {
		mxCoo_to_smat(mxM, R, rows, cols);
	}
}
// }}} end-of-conversion

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) { // {{{
	size_t rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M.resize(cols, vec_t(rows,0));
	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			M[c][r] = val[idx++];
	return 0;
} // }}}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {// {{{
	size_t cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			val[idx++] = M[c][r];
	return 0;
} // }}}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) { // {{{
	size_t rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M.resize(rows, vec_t(cols,0));
	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			M[r][c] = val[idx++];
	return 0;
} // }}}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) { // {{{
	size_t rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(size_t c = 0, idx = 0; c < cols; ++c)
		for(size_t r = 0; r < rows; r++)
			val[idx++] = M[r][c];
	return 0;
} // }}}

mxArray* pmf_model_to_mxStruture(pmf_model_t& model) { // {{{
	static const char *field_names[] = {"W", "H", "global_bias"};
	static const int nr_fields = 3;
	mxArray *ret = mxCreateStructMatrix(1, 1, nr_fields, field_names);
	mxArray *mxW = mxCreateDoubleMatrix(model.rows, model.k, mxREAL);
	mxArray *mxH = mxCreateDoubleMatrix(model.cols, model.k, mxREAL);
	mxArray *mxglobal_bias = mxCreateDoubleMatrix(1, 1, mxREAL);
	if(model.major_type == pmf_model_t::COLMAJOR) {
		matCol_to_mxDense(model.W, mxW);
		matCol_to_mxDense(model.H, mxH);
	} else { // pmf_model_t::ROWMAJOR
		matRow_to_mxDense(model.W, mxW);
		matRow_to_mxDense(model.H, mxH);
	}
	*mxGetPr(mxglobal_bias) = model.global_bias;
	mxSetField(ret, 0, field_names[0], mxW);
	mxSetField(ret, 0, field_names[1], mxH);
	mxSetField(ret, 0, field_names[2], mxglobal_bias);
	return ret;
} // }}}

pmf_model_t gen_pmf_model(const mxArray *mxW, const mxArray *mxH,
		double global_bias=0, pmf_model_t::major_t major_type = pmf_model_t::ROWMAJOR) { // {{{
	size_t rows = mxGetM(mxW), cols = mxGetM(mxH), k = mxGetN(mxW);
	pmf_model_t model = pmf_model_t(rows, cols, k, major_type, false, global_bias);
	if(model.major_type == pmf_model_t::COLMAJOR) {
		mxDense_to_matCol(mxW, model.W);
		mxDense_to_matCol(mxH, model.H);
	} else { // pmf_model_t::ROWMAJOR
		mxDense_to_matRow(mxW, model.W);
		mxDense_to_matRow(mxH, model.H);
	}
	return model;
} // }}}

pmf_model_t mxStruture_to_pmf_model(const mxArray *mx_model, pmf_model_t::major_t major_type = pmf_model_t::ROWMAJOR) { // {{{
	static const char *field_names[] = {"W", "H", "global_bias"};
	static const int nr_fields = 3;
	mxArray *mxW = mxGetField(mx_model, 0, "W");
	mxArray *mxH = mxGetField(mx_model, 0, "H");
	mxArray *mxglobal_bias = mxGetField(mx_model, 0, "global_bias");
	double global_bias = *(mxGetPr(mxglobal_bias));
	return gen_pmf_model(mxW, mxH, global_bias, major_type);

} // }}}

```







因为 $\widetilde{\boldsymbol{R}}$ 是随机的，导致其不稳定，使用 **bagging** 技巧 ( **其实就是多次采样取均值** )

![image-20220506085105953](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220506085106_image-20220506085105953.png)

接下来使用**ALS**重构 **R**

### 计算复杂度

#### wALS

![image-20220506130605899](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220506130605_image-20220506130605899.png)

#### sALS-ENS

![image-20220506130550225](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220506130550_image-20220506130550225.png)



$n_r:正样本个数$

$\alpha*n_r:负样本个数$

## 实验

### 数据集

* 第一个数据集，雅虎新闻数据集，是通过记录流点击新闻。每个记录都是一个**用户-新闻**对，它由用户id和雅虎新闻文章的URL组成。预处理后 为了确保相同的新闻总是得到相同的文章id，我们有3158个独立用户和1536个相同的新闻故事。

* 第二个数据集来自一个社交书签网站。它是爬取 http://del.icio.us.该数据包含246,436篇帖子，有3000个用户和2000个标签

### Method

作为机器学习和数据挖掘中最常用的方法，我们使用交叉验证来估计不同算法的性能。验证数据集被随机分为训练集和测试集，分割比例为80/20。训练集包含80%的已知正样本，矩阵的其他元素被视为未知。该测试集包括其他20%已知的正样本和所有未知的样本。

**请注意，在训练集中已知的正样本在测试过程中被排除在外。一种方法的良好性能的直觉是，该方法有很高的概率来排序已知的正样本，而大多数未知样本通常是负样本**。我们使用MAP和half-life utility来评估测试集上的性能，这将在下面进行讨论。我们重复上述步骤 20次，并报告实验结果的平均值和标准差。我们的方法的参数和基线是由交叉验证确定的

### 评价指标

* **MAP ( Mean Average Precision )**

首先了解 **precision** 和 **recall**

理解推荐系统中的MAP指标，首先我们需要明确Precision这个指标，即精确度。Precision和Recall的公式如下所示，Precision代表了预测为正的样本中，有多少是真正的正样本

![image-20220509213231251](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220509213231_image-20220509213231251.png)

在推荐系统场景下，我们可以定义正样本为**相关的商品**，因此**Precision**就代表了，推荐的n个商品中，有多少个相关商品(正样本)。而**Recall**就代表了数据库中一共有m个相关商品，推荐系统选出了多少个**相关商品**(正样本)。

举例说明：

例如下面的理财产品推荐场景，用户在未来购买了四款产品，而一个推荐系统在当前推荐了三款产品，用户只购买了一款产品。那么此时，推荐系统的Recall为1/4，Precision为1/3

![b094a9a1-9c0d-44ff-9443-ac1cb9b2dcbe_](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510122745_b094a9a1-9c0d-44ff-9443-ac1cb9b2dcbe_.jpg)

值得注意的是，由于屏幕大小限制，推荐系统只能展示前N个商品，因此一般推荐系统中的Percision计算会采用Cut-off形式进行计算。如下图所示，尽管我们的推荐系统可以推荐m个商品，但是在Cutoff-Precision的计算过程中，只会考虑前k个商品的Percision。

![f26025e1-85c1-44c4-829a-7f2681d95421_](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510124905_20220510124519_f26025e1-85c1-44c4-829a-7f2681d95421_.jpg)

根据上面的概念，我们就可以定义Average Precision。从公式中可以看出，AP@N可以直观理解为枚举Precision@k之后取平均值

![preview](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510124917_v2-b45e7061caf19ec9d871865dbec420e4_r.jpg)

**m=N, 如果是正样本(标签为正), rel(k)=1, 否则rel(k)=0**

以上面左图为例：$AP@3=\frac{1}{3}\{P(1)\cdot rel(1)+P(2)\cdot rel(2)+P(3)\cdot rel(3) \}$

$P(1)=1;\ P(2)=1/2;\ P(3)=1/3$

$rel(1)=1;\ rel(2)=0;\ rel(3)=0$

所以  $AP@3=\frac{1}{3}(1\cdot1+\frac{1}{2}\cdot0+\frac{1}{3}\cdot0)=0.33$

在推荐系统场景下，使用AP最大的好处在于AP不仅仅考虑了商品推荐的准确率，还考虑了推荐顺序上的差异。考虑下面这样一个表格，从整体来考虑的话，三种推荐方案都只推荐了一个相关商品，但是第一种推荐方案明显是更好的，而AP指标可以体现这种差异。

![preview](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510140152_v2-80191aa96afb29c947c3319b2179e699_r.jpg)

**上表为正样本在ranklist上的不同位置得到的AP指标值**

介绍了AP@N指标，我们就可以定义MAP@N指标了。其实MAP@N指标就是将所有用户 U 的AP@N指标进行平均。

![image-20220510160315668](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510160315_image-20220510160315668.png)

总的来说，MAP指标同时考虑了预测精准度和相对顺序，从而避免了传统Precision指标无法刻画推荐商品相对位置差异的弊端。因此。在很多推荐系统场景下，MAP指标是一个非常值得尝试的推荐系统评估指标。

[参考来源](如何理解推荐系统中的MAP评估指标？ - 震灵的回答 - 知乎 https://www.zhihu.com/question/491231560/answer/2160321311)

* **Half-life Utility (HLU)** 

![image-20220510161024150](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510161024_image-20220510161024150.png)

![image-20220510161033706](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510161033_image-20220510161033706.png)

$R_u:\delta(j)表示在rank \  list上的第j个位置是否为正样本，如果是就为1，反之为0，\beta是一个half-life参数，论文设置为5$[^Empirical analysis of predictive algorithms for collaborative fifiltering. UAI 1998]

$R_u^{max}: 正样本在rank \ list的top位置上(第一位)$

### Baseline

#### AMAN

在AMAN设置中，大多数传统的协同过滤算法都可以直接应用。在本文中，我们使用了几种著名的协同过滤算法，并结合了AMAN 作为我们的baseline，其中包括交替的最小二乘作为负假设（ALS-AMAN），奇异值分解(SVD)，以及基于邻域的方法**用户-用户相似度**和**项目-项目相似度**算法

#### AMAU

按照AMAU策略，很难采用传统的协同过滤算法来获得非平凡解，正如我们在第1节中所讨论的。在这种情况下，**根据项目的总体受欢迎程度**进行排序是一种简单但广泛使用的推荐方法。==**另一种可能的方法是将单类协同过滤问题转化为单类分类问题。**==本文使用单类SVM方法。其想法是为每个项目创建一个单类SVM分类器，它将用户对其余项目的评分作为输入特征，并预测用户对目标项目的评分是正还是负。SVM分类器的训练实例集由那些对目标项目进行评级的用户的评级概况组成，应该只包含单类协同过滤中的正样本，**这可以用于为每个目标项训练一个单类SVM分类器。**

#### 结果

![image-20220510173452778](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510173452_image-20220510173452778.png)

参数α控制负样本的比例。随着α→0，方法接近AMAU策略，随着α→1，方法接近AMAU策略。我们可以清楚地看到最好的结果介于两者之间。也就是说，加权方法和抽样方法都优于baseline。加权方法略优于抽样方法。但如所示 在表（3）中，当α相对较小时，抽样方法更有效。

![image-20220510175752272](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510175752_image-20220510175752272.png)

**我们还可以看到，与AMAU策略相比，AMAN更有效。**这是因为，虽然未标记的样本的标签信息是未知的，但我们仍然预先知道它们中的大多数都是负样本。无视这些信息并不会导致有竞争力的建议。这与类不平衡分类问题中的结论有些不同，在这些问题中，丢弃主导类的例子通常会产生更好的结果。

图3中alpha为：

![image-20220510181611246](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510181611_image-20220510181611246.png)

![image-20220510181503742](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220510181503_image-20220510181503742.png)

$\alpha不影响baseline(item-item, SVD,ALS-AMAN)的流行度$

## 对于sALS-ENS的困惑

![图片2](https://cdn.jsdelivr.net/gh/leelige/upic@main/uPic/20220509175643_%E5%9B%BE%E7%89%872.png)



### 



