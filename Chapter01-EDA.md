

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# 第1章 探索性数据分析

统计学是一门应用科学关注的是数据的分析和建模。探索数据是所有数据科学项目的第一步。探索性数据分析（EDA）是统计学中一个相对新的领域。经典统计学几乎只注重推断，即从小样本得出关于整体数据的结论。约翰图基于1962年在论文“The Future of Data Analysis”中提出了“数据分析”的学科，并将统计推断包括在其中。于1977年出版了“Exploratory Data Analysis”一书，提出了“探索性数据分析”的研究领域。

## 1.1 结构化数据的组成
如何将大量的原始数据转换为可操作的信息，这是数据科学所面对的主要挑战。使用统计学的概念，需要将非结构化的原始数据结构化，或者出于研究目的采集数据。

*术语*
- 连续型数据：可以在一个区间内取任何值。同义词：区间数据，浮点型数据，数值数据。
- 离散型数据：数据只能取整数，例如计数。同义词：整数型数据，计数型数据。
- 分类型数据：数值只能从特定的集合中取值，表示一系列可能的分类。同义词：枚举数据，列举数据，因子数据，标称数据，多分支数据。
- 二元数据：一种特殊的分类数据，数值只能从两个值中取一个。同义词：二分数据，逻辑型数据，指示性数据，布尔型数据。
- 有序数据：具有明确排序的分类数据。同义词：有序因子数据。

对于数据分析和预测建模来说，数据建模对于确定可视化类型，数据分析或者统计模型是非常重要的。使用数据类型可以改善计算性能。变量的数据类型决定了软件处理变量的计算方法。

## 1.2 矩形数据

矩形数据对象是数据科学分析中典型引用结构，举行数据对象包括电子表格，数据库表格等。举行数据本质是一个二维矩阵。通常一行表示一个记录（事例），列表示特征（变量）。数据通常并非一开始就是矩形形式的，先经过处理，才能转换为相应形式。

### 1.2.1 数据框和索引

传统的数据库表会指定一列或者多列作为索引，索引可以极大提高某些SQL查询的效率。pandas数据分析库中基本的举矩形数据结构是DataFrame对象，默认会创建一个整型索引，支持设置多级或者层次索引，以提高特定操作的效率。

*术语差异*
统计学家在模型中使用预测变量去预测一个响应或因变量，而数据科学家使用特征来预测目标。对于一行数据，计算机科学家使用样本这一术语；而统计学家使用样本表示一个行的集合。

### 1.2.2 非矩形数据结构

时序数据记录了对同一变量的连续测量值。时序数据是统计预测方法的原始输入数据，也是物联网设备所产生对 数据的管家组成部分。

空间数据结构用于地图和定位分析。在对象标识中，空间数据关注的是对象及空间坐标。字段视图关注空间中的小单元及相关的度量值。

图形（或网络）数据结构用于表示物理上，社交网络上的抽象关系。图形结构对于网络优化和推荐系统等问题十分重要。

*统计学中图*
计算机学学中，图通常表示对实体之间的关联情况的描述，是底层的数据结构。统计学中，图用于指代各种绘图和可视化结构，而不仅仅是指实体间的关联情况，通常用来指代可视化，而非数据结构。

### 1.2.3 扩展阅读

- Python中关于数据框的文档：[Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)


## 1.3 位置估计

变量表示了测量数据或者计数数据。探索数据的一个基本步骤就是获取每个特征（变量）的“典型值”。典型值是对数据最常出现位置的估计，即数据的集中趋势。

*术语*

| 术语       | 定义                                                       | 同义词       |
| ---------- | ---------------------------------------------------------- | ------------ |
| 均值       | 所有数据之和除以数值的个数                                 | 平均数       |
| 加权均值   | 各数值乘以相应的权重值，相加求和，再除以权重总和。         | 加权平均值   |
| 中位数     | 使得数据集中有一半数据位于该值之上和之下                   | 第50百分位数 |
| 加权中位数 | 使得排序数据集中，分别有一半的权重之和位于该值之上和之下。 |              |
| 切尾均值   | 从数据集中剔除一定数量的极值后，再求均值。                 | 截尾均值     |
| 稳健       | 对极值不敏感                                               | 耐抗性       |
| 离群值     | 与大部分数值差异很大的数据值。                             | 极值         |
|            | (*注*：形式化的定义和运算在后续章节中。)                     |              |





估计量（estimate）表示从手头已有数据中计算得到的值，用于描述观察值和确切的（理论上为真的）状态之间的差异，数据科学家跟倾向于称这些计算得到的值为度量（metric）。统计学的核心是如何解释不确定度，而数据科学家关注如何解决一个具体的目标。因此，统计学家使用估计量，而数据科学家使用度量。



### 1.3.1 均值

均值，又成为平均值。均值等于所有值的和除以值的个数。给定n个数据值：$x_1, x_2, \dots, x_n$，均值计算公式：

$$
均值=\overline{x}=\frac{\sum_{i=1}^{n}x_i }{n}
$$



```python
x=[3,5,1,2]
x_mean=sum(x)/len(x)
x_mean
```




    2.75




```python
def simple_average(x):
    return np.sum(x)/len(x)

x=[3,5,1,2]
simple_average(x)
```




    2.75



**切尾均值**：是均值的一种变体。在有序数据集的咯昂段去除一定数量的值，再计算剩余部分的均值。对于有序数据集$x_{(1)},x_{(2)},\dots, x_{(n)}$，其中$x_{(1)}$表示最小值，$x_{(n)}$表示最大值，从序列中去除p个最大值和p个最小值后的切尾均值的计算公式：
$$
切尾均值=\overline{x}=\frac{\sum_{i=p+1}^{n-p}x_{(i)} }{n-2p}
$$


```python
# def tail_average(x, p):
#     return np.sum(sorted(x)[p:-p])/(len(x)-2*p)

# x=np.array([4, 5, 3, 10, 0, 2])
# print(tail_average(x, p=2))

def tail_average(x, trim):
    n=len(x)
    p=int(n*trim)
    #print(sorted(x)[p:n-p])
    return np.sum(sorted(x)[p:n-p])/(n-2*p)

x=np.array([4, 5, 3, 10, 0, 2])
print(tail_average(x, trim=0.2))
```

    3.5
    

**加权均值**：将每个值$x_i$乘以一个权重值$w_i$，并将加权值的总和除以权重的总和，计算公式为：
$$
加权均值=\overline{x_w}=\frac{\sum_{i=1}^{n} w_ix_i}{\sum_{i=1}^{n}w_i}
$$


```python
def weighted_average(x,w):
    return np.sum(x*w)/np.sum(w)

x=np.array([1,2,3])
w=np.array([4,0,-1])
weighted_average(x,w)
```




    0.33333333333333331



**中位数**是位于有序数据集中间位置的数值。如果数值的个数为偶数，那么中位数实际上是位于中间位置的两个值的均值。


```python
def simple_median(x):
    n=len(x)
    x=np.sort(x)
    return (x[int(n/2)-1]+x[int(n/2)])/2.0 if n%2==0 else x[int(n/2)]

x=np.array([1,3,2,0,100])
print(simple_median(x))
x=np.array([1,3,2,0])
print(simple_median(x))
```

    2
    1.5
    

**加权中位数**取可以使有序数据集上下两部分的权重总和相同的值。

加权中位数和中位数对离群值不敏感。



```python
def weighted_median(x, w):
    t=[]
    for x_, w_ in zip(x, w):
        t.append([x_, w_]) 
    t=np.array(sorted(t, key=lambda xx:xx[0]))
    tw=t[:,1]
    tw_accum=np.cumsum(tw)
    idx=np.where(tw_accum>tw_accum[-1]/2)[0][0]
    return t[idx][0]

x=np.array([1,3,2,0,100])
w=np.array([0.1,0.5,0.25,0.0,0.15])
weighted_median(x,w)
```




    3.0



**离群值**是距离数据集中其他所有值都很远的值。离群值本身并不一定是无效或者错误的数据，但往往是由数据的错误所导致的。


```python
muder={'state':['阿拉巴马州','阿拉斯加州','亚利桑那州','阿肯色州','加利福尼亚州','科罗拉多州','康涅狄格州','特拉华州'],
      'population':[4779736, 710231, 6392017, 2915918, 37253956, 5029196, 3574097, 897934],
      'muder_rate':[5.7, 5.6, 4.7, 5.6, 4.4, 2.8, 2.4, 5.8]}

muder_df=pd.DataFrame(muder)
muder_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>muder_rate</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.7</td>
      <td>4779736</td>
      <td>阿拉巴马州</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.6</td>
      <td>710231</td>
      <td>阿拉斯加州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>6392017</td>
      <td>亚利桑那州</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6</td>
      <td>2915918</td>
      <td>阿肯色州</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.4</td>
      <td>37253956</td>
      <td>加利福尼亚州</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.8</td>
      <td>5029196</td>
      <td>科罗拉多州</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.4</td>
      <td>3574097</td>
      <td>康涅狄格州</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.8</td>
      <td>897934</td>
      <td>特拉华州</td>
    </tr>
  </tbody>
</table>
</div>




```python
muder_df['population']
```




    0     4779736
    1      710231
    2     6392017
    3     2915918
    4    37253956
    5     5029196
    6     3574097
    7      897934
    Name: population, dtype: int64




```python
simple_average(muder_df['population'])
```




    7694135.625




```python
len(muder_df['population'])
```




    8




```python
sum(muder_df['population'])
```




    61553085




```python
simple_median(muder_df['population'])
```




    4176916.5




```python
muder_us=pd.read_csv('data/state.csv')
muder_us
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Population</th>
      <th>Murder.Rate</th>
      <th>Abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>4779736</td>
      <td>5.7</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>710231</td>
      <td>5.6</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>6392017</td>
      <td>4.7</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>2915918</td>
      <td>5.6</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>37253956</td>
      <td>4.4</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>5029196</td>
      <td>2.8</td>
      <td>CO</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>3574097</td>
      <td>2.4</td>
      <td>CT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>897934</td>
      <td>5.8</td>
      <td>DE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Florida</td>
      <td>18801310</td>
      <td>5.8</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Georgia</td>
      <td>9687653</td>
      <td>5.7</td>
      <td>GA</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hawaii</td>
      <td>1360301</td>
      <td>1.8</td>
      <td>HI</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Idaho</td>
      <td>1567582</td>
      <td>2.0</td>
      <td>ID</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Illinois</td>
      <td>12830632</td>
      <td>5.3</td>
      <td>IL</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Indiana</td>
      <td>6483802</td>
      <td>5.0</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iowa</td>
      <td>3046355</td>
      <td>1.9</td>
      <td>IA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Kansas</td>
      <td>2853118</td>
      <td>3.1</td>
      <td>KS</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kentucky</td>
      <td>4339367</td>
      <td>3.6</td>
      <td>KY</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Louisiana</td>
      <td>4533372</td>
      <td>10.3</td>
      <td>LA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Maine</td>
      <td>1328361</td>
      <td>1.6</td>
      <td>ME</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maryland</td>
      <td>5773552</td>
      <td>6.1</td>
      <td>MD</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Massachusetts</td>
      <td>6547629</td>
      <td>2.0</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Michigan</td>
      <td>9883640</td>
      <td>5.4</td>
      <td>MI</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Minnesota</td>
      <td>5303925</td>
      <td>1.6</td>
      <td>MN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Mississippi</td>
      <td>2967297</td>
      <td>8.6</td>
      <td>MS</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Missouri</td>
      <td>5988927</td>
      <td>6.6</td>
      <td>MO</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Montana</td>
      <td>989415</td>
      <td>3.6</td>
      <td>MT</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Nebraska</td>
      <td>1826341</td>
      <td>2.9</td>
      <td>NE</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nevada</td>
      <td>2700551</td>
      <td>6.0</td>
      <td>NV</td>
    </tr>
    <tr>
      <th>28</th>
      <td>New Hampshire</td>
      <td>1316470</td>
      <td>0.9</td>
      <td>NH</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Jersey</td>
      <td>8791894</td>
      <td>3.9</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Mexico</td>
      <td>2059179</td>
      <td>4.8</td>
      <td>NM</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New York</td>
      <td>19378102</td>
      <td>3.1</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>32</th>
      <td>North Carolina</td>
      <td>9535483</td>
      <td>5.1</td>
      <td>NC</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Dakota</td>
      <td>672591</td>
      <td>3.0</td>
      <td>ND</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Ohio</td>
      <td>11536504</td>
      <td>4.0</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Oklahoma</td>
      <td>3751351</td>
      <td>4.5</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oregon</td>
      <td>3831074</td>
      <td>2.0</td>
      <td>OR</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Pennsylvania</td>
      <td>12702379</td>
      <td>4.8</td>
      <td>PA</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Rhode Island</td>
      <td>1052567</td>
      <td>2.4</td>
      <td>RI</td>
    </tr>
    <tr>
      <th>39</th>
      <td>South Carolina</td>
      <td>4625364</td>
      <td>6.4</td>
      <td>SC</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Dakota</td>
      <td>814180</td>
      <td>2.3</td>
      <td>SD</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Tennessee</td>
      <td>6346105</td>
      <td>5.7</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Texas</td>
      <td>25145561</td>
      <td>4.4</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Utah</td>
      <td>2763885</td>
      <td>2.3</td>
      <td>UT</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Vermont</td>
      <td>625741</td>
      <td>1.6</td>
      <td>VT</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Virginia</td>
      <td>8001024</td>
      <td>4.1</td>
      <td>VA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Washington</td>
      <td>6724540</td>
      <td>2.5</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>47</th>
      <td>West Virginia</td>
      <td>1852994</td>
      <td>4.0</td>
      <td>WV</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Wisconsin</td>
      <td>5686986</td>
      <td>2.9</td>
      <td>WI</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wyoming</td>
      <td>563626</td>
      <td>2.7</td>
      <td>WY</td>
    </tr>
  </tbody>
</table>
</div>




```python
simple_average(muder_us['Population'])
```




    6162876.3




```python
simple_median(muder_us['Population'])
```




    4436369.5




```python
tail_average(muder_us['Population'], trim=0.1)
```




    4783697.125




```python
muder_us['Murder.Rate']
```




    0      5.7
    1      5.6
    2      4.7
    3      5.6
    4      4.4
    5      2.8
    6      2.4
    7      5.8
    8      5.8
    9      5.7
    10     1.8
    11     2.0
    12     5.3
    13     5.0
    14     1.9
    15     3.1
    16     3.6
    17    10.3
    18     1.6
    19     6.1
    20     2.0
    21     5.4
    22     1.6
    23     8.6
    24     6.6
    25     3.6
    26     2.9
    27     6.0
    28     0.9
    29     3.9
    30     4.8
    31     3.1
    32     5.1
    33     3.0
    34     4.0
    35     4.5
    36     2.0
    37     4.8
    38     2.4
    39     6.4
    40     2.3
    41     5.7
    42     4.4
    43     2.3
    44     1.6
    45     4.1
    46     2.5
    47     4.0
    48     2.9
    49     2.7
    Name: Murder.Rate, dtype: float64




```python
weighted_average(x=muder_us['Murder.Rate'], w=muder_us['Population'])
```




    4.445833981123394




```python
weighted_median(x=muder_us['Murder.Rate'], w=muder_us['Population'])
```




    4.4000000000000004


