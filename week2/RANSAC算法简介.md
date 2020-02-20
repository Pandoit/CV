# 1.RANSAC算法简介

&ensp;&ensp;当我们从估计模型参数时，用$p$表示一些迭代过程中从数据集内随机选取出的点均为局内点的概率；此时，结果模型很可能有用，因此$p$也表征了算法产生有用结果的概率。用w表示每次从数据集中选取一个局内点的概率，如下式所示：  
&ensp;&ensp;$w$ = 局内点的数目 / 数据集的数目  
&ensp;&ensp;通常情况下，我们事先并不知道$w$的值，但是可以给出一些鲁棒的值。假设估计模型需要选定$n$个点，$w^n$是所有$n$个点均为局内点的概率；$1-w^n$是$n$个点中至少有一个点为局外点的概率，此时表明我们从数据集中估计出了一个不好的模型。如果进行了$k$次迭代，$(1-w^n)^k$表示算法永远都不会选择到n个点均为局内点的概率，它和$1-p$相同。因此，$1−p=(1-w^n)^k$。我们对上式的两边取对数，得出:$k=log(1−p)/log(1−w^n)$  ，由此式可知要$p$大也就是要$k$大，注意该式子分母为负。
&ensp;&ensp;值得注意的是，这个结果假设$n$个点都是独立选择的；也就是说，某个点被选定之后，它可能会被后续的迭代过程重复选定到。这种方法通常都不合理，由此推导出的$k$值被看作是选取不重复点的上限。例如，要寻找适合的直线，RANSAC算法通常在每次迭代时选取2个点，计算通过这两点的直线可能模型，要求这两点必须唯一。

# 2.RANSAC算法流程

1.在数据集里任取两个点，从而确定$y=ax+b$中的$a$和$b$值

2.计算数据集里${(\hat	y-y)}/{\hat y}$是否在**满足阈值1**，如果是将所有满足条件的点添加到内点里

3.计算所有内点占总共数据集的百分比，该百分比**满足阈值2**就结束算法，返回$a$和$b$值

4.如果不满足3中的阈值2的条件则在内点集里再次随机抽取两个点，确定新的$a'$和$b'$然后再按照第2步扩充内点

并继续按照3判断，满足就终止，不满足继续按照步骤123进行循环，当该循环次数**满足阈值3**的时候亦可以终循环次数输出$a$和$b$值。(具体讲解可以看视频第三节的结尾部分)

5.**其终止条件有两个**，一个是内点数目满足阈值2，还有一个是循环次数满足阈值3.

#3.伪代码如下： 
input:
    data - a set of observations 一组观测数据
    model - a model that can be fitted to data 适用于数据的模型
    n - the minimum number of data required to fit the model 适用于模型的最少数据个数
    k - the number of iterations performed by the algorithm 算法的迭代次数
    t - a threshold value for determining when a datum fits a model 决定数据是否适用于模型的阈值
    d - the number of close data values required to assert that a model fits well to data 决定模型是否适用于数  &emsp;据集的数据数目

output:
    best_model - model parameters which best fit the data (or null if no good model is found)和数据最匹配的  &emsp;模型参数（如果没有找到好的模型，返回null）
    best_consensus_set - data point from which this model has been estimated 估计出模型的数据点
    best_error - the error of this model relative to the data 与数据相关的模型误差

iterations = 0

best_model = null

best_consensus_set = null

best_error = infinity

while iterations < k
    maybe_inliers = n randomly selected values from data  从数据集中随机选择n个内点
    maybe_model = model parameters fitted to maybe_inliers 适合这n个内点的模型参数
    consensus_set = maybe_inliers

    for every point in data not in maybe_inliers  对于每一个不在maybe_inliers 中的数据点
       if point fits maybe_model with an error smaller than t 如果该点适合maybe_model，并且误差小于阈值
           add point to consensus_set  就将该点加入到consensus_set中
    
    if the number of elements in consensus_set is > d 如果consensus_set中的数据点个数大于d，这暗示着已经找到了好的模型，现在测试该模型到底多好
       (this implies that we may have found a good model,now test how good it is)
       better_model = model parameters fitted to all points in consensus_set 适合consensus_set中所有点的模型参数
       this_error = a measure of how well better_model fits these points 衡量适合这些点的better_model有多好
       if this_error < best_error 比最好的误差还小，说明发现了比之前好的模型，保存该模型直到有更好的模型
          (we have found a model which is better than any of the previous ones,keep it until a better one is found)
           best_model = this_model
           best_consensus_set = consensus_set
           best_error = this_error
    
    increment iterations迭代次数增加

return best_model, best_consensus_set, best_error


```python

```
