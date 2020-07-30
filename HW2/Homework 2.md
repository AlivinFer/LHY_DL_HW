## Homework 2

### Purpose: Binary Classification

从给定的个人信息，预测此人的年收入是否大于 50k

### Data 简介

数据集为：ADULT(https://archive.ics.uci.edu/ml/datasets/Adult)

该数据来源于美国1994年人口普查数据库抽取而来，可以用来预测居民收入是否超过 50k/year。

数据集各属性说明：0~13 为属性，14为类别

|   序号    |     字段名     |      含义      |  类型  |
| :-------: | :------------: | :------------: | :----: |
|     0     |      age       |      年龄      | double |
|     1     |   workclass    |    工作类型    | string |
|     2     |     fnlwgt     |      序号      | string |
|     3     |   education    |    教育程度    | string |
|     4     | education_num  |   受教育时间   | double |
|     5     | marital_status |    婚姻状况    | sting  |
|     6     |   occupation   |      职业      | string |
|     7     |  relationship  |      关系      | string |
|     8     |      race      |      种族      | string |
|     9     |      sex       |      性别      | string |
|    10     |  capital_gain  |    资本收益    | string |
|    11     |  capital_loss  |    资本损失    | string |
|    12     | hours_per_week | 每周工作小时数 | double |
|    13     | native_country |      原籍      | string |
| 14(label) |     income     |      收入      | string |

总共有 32561 条训练资料，16281 条测试资料，其中资料维度为 106

### Summary 总结

使用 生成模型（generative model）和 判别模型（discriminative model）

- generative model 的难点在于要假设资料的分配和变数之间的关系，如果资料越符合假设分配，效果也就越好
- discriminative model 的难点则是在于如何选择超参数

#### Logistic Regression

一般对于二分类最常用的方法为逻辑回归（logistic regression），其背后有一些统计的推导过程，简单说逻辑回归跟一般线性回归差别只在于计算线性回归之后再利用 sigmoid 函数将数值转换到 0-1 之间，另外将转换过的数值透过门槛值来区分类别，而门槛值得设置可以根据资料的不同来做设计，常用门槛值为 0.5。

此次作业将所有的训练资料中的 20% 当成验证集，并由另外的 80% 的资料集来训练参数，并使用 Mini-batch Gradient Descent 算法来训练逻辑回归的参数 W 和 b。门槛值则用最一般的方式设置 0.5。

#### Probabilistic Generative Model

由于我们的目标是将资料进行二元分类，可以假设年收入大于 50k（y=1）为 <img src="https://latex.codecogs.com/gif.latex?C_{1}" title="C_{1}" />类别和年收入小于 50k 为<img src="https://latex.codecogs.com/gif.latex?C_{2}" title="C_{2}" />类别且各为 106维的正太分布（高斯分布），且每个特征是独立的，其中协方差矩阵共用，最后由最大估计法直接计算参数<img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1},&space;\mu&space;_{2},&space;\Sigma" title="\mu _{1}, \mu _{2}, \Sigma" />的最佳解。

有了模型的参数，我们可由概率的方式来决定资料是属于哪个类别，也就是说，分别计算资料来自于第一类的概率<img src="https://latex.codecogs.com/gif.latex?P(C_{1})" title="P(C_{1})" />和第二类的概率<img src="https://latex.codecogs.com/gif.latex?P\left&space;(C_{2}&space;\right&space;)" title="P\left (C_{2} \right )" />以及资料在第一类的概率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{1})" title="P(x\mid C_{1})" />和第二类的概率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{2})" title="P(x\mid C_{2})" />，最后由上述这些概率去计算资料属于第一类的概率：
$$
P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}
$$
和第二类的概率：<img src="https://latex.codecogs.com/gif.latex?1-P(x\mid&space;C_{1})" title="1-P(x\mid C_{1})" />,最后由此概率决定资料类别。

这个作业假设资料符合正太分布，主要的原因是因为数学推导相对而言比较简单加上正太分布相对而言比较直观，当然也可以假设其他概率分布，如像是 0 和 1的离别资料，假设伯努利分布相对于正态分布就会比较合理，另外假设每个特征是独立的也就是使用 Naive Bayes Classifier

