# 概率论基础

[author]: # "Vonng (fengruohang@outlook.com)"
[tags]: # "统计，数学，概率论"
[mtime]: #	"2017-03-27 14:00"
《统计推断》第一章：概率论笔记

---



## 1. 集合论

样本空间和样本点是概率论中无定义的基本概念，如同几何中的点和直线的概念一般。

##### 定义：事件

事件：事件是样本点的**集合**。

* $A=0$表示事件A不含任何样本点，即A是不可能事件。

  $A=0$是一个代数表达式而不是算术表达式，0在这里是一个符号。

* 样本空间中一切不属于事件A的点所构成的事件称为A的补事件。或称非事件。并以$A^C$记之，$S^C=0$

* 事件A、B、C的交，用$A\cap B \cap C$表示，并用$A \cup B \cup C$表示事件的并

* $A\subset B$ 称为A蕴涵B，意味着A的每一个点都在B中。



## 2 概率论基础

这里采用公理化的方法来**定义**概率。至于如何**解释**概率，例如“事件的出现频率”（频率学派），或者是“对事件出现的信念”（贝叶斯学派），这里我们并不关心。

### 2.1 公理化基础

对于样本空间S的每一个事件A，我们希望给A赋一个0到1之间的数值P(A)，称之为A的概率。

##### 定义：σ代数/Borel域

S的一族子集如果满足下列三个性质，就称为一个**σ代数**或一个**Borel域**，记作$\mathcal{B}$：

* $\varnothing \in \mathcal{B}$
* $A \in \mathcal{B} \Rightarrow A^C \in \mathcal{B} $
* $\displaystyle A_1,A_2,\cdots \in \mathcal{B} \Rightarrow \bigcup_{i=1}^{\infty}{A_i} \in \mathcal{B}$

满足这样三条性质（空集存在，对补运算与并运算封闭）的σ代数有很多，这里讨论的是包含S中全体开集的最小σ代数。对于可数样本空间，通常是$\mathcal{B}=\{S的全体子集，包括S本身\}$。对于不可数的样本空间，例如$S=(-\infty,\infty)$为实数轴，则可以取$\mathcal{B} $为包含所有形如$[a,b],(a,b],[a,b),(a,b)$的集合，其中$a,b \in \mathbb{R}$。

##### 定义：概率函数

已知样本空间S和σ代数$\mathcal{B}$，定义在$\mathcal{B}$上且满足下列条件的函数P称为一个**概率函数(probability fucntion)**

* $\forall A \in \mathcal{B}, P(A) \ge 0$
* $P(S) = 1$
* 若$A_1,A_2,\cdots \in \mathcal{B}$两两不相交，则$\displaystyle P(\bigcup_{i=1}^{\infty}{A_i}) = \sum_{i=1}^{\infty}{P(A_i)}$

概率非负性，概率归一化，概率可数可加。这三条性质称为概率公理，或Kolmogorov公理。只要满足这三条公理，函数P就可以称为一个概率函数。

（PS：统计学家通常不接受可数可加公理，只接受其推论：有限可加性公理$P(A\cup B)=P(A)+P(B)$）



### 2.2 概率演算

定理：设P是一个概率函数，$A,B \in \mathcal{B}$ 则

* $P(\varnothing) = 0$
* $P(A) \le 1$
* $P(A^C) = 1- P(A)$
* $P(B \cap A^C) = P(B)- P(A \cap B)$
* $P(A \cup B) = P(A) + P(B)- P(A \cap B)$
* $A \subset B \Rightarrow P(A) \le P(B)$
* $P(A \cap B) \ge P(A) + P(B) - 1$ ，Bonferroni不等式，用单个事件概率估算并发概率
* 对于任意划分$C_1,C_2,\cdots$，都有$\displaystyle P(A)= \sum_{i=1}^{\infty}{A \cap C_i}$
* 对于任意集合$A_1,A_2,\cdots$都有$\displaystyle P(\bigcup_{i=1}^{\infty}{A \cap C_i})  \le \sum_{i=1}^{\infty}{P(A\cap C_i)}$，Boole不等式。




### 2.3 计数

计数涉及到很多组合分析的知识，这些分析都基于这样一条定理：

##### 定理：计数基本定理

如果一项工作由k个相互独立的子任务组成，其中第i个任务可以使用$n_i$种方式完成，则正向工作可以用$n_1 \times n_2 \times \cdots \times n_k$种方式组成。

该定理的证明可以由笛卡尔积运算的定义与性质得出。



计数的两个基本问题包括：

* 样本是否有序？
* 抽样是否放回？



##### 定义：总体/子总体/有序样本

* 总体：我们用大小为n的总体表示一个由n个元素构成的集合。

  因为总体是集合，所以总体是无序的，总体相同当且仅当两个总体含有相同的元素。

* 子总体：从大小为n的总体中选取r个元素，就构成了一个大小为r的子总体。

* 对子总体中的元素进行编号，可以得到大小为r的**有序样本**。总共有$n!$种。



##### 从n个对象中选取r个的全体可能方式的数目

|       | 无放回抽样                                    | 有放回抽样              |
| ----- | ---------------------------------------- | ------------------ |
| 有序样本  | $\frac {n!} {(n-r)! } = \binom n r A_r^r $ | $n^r$              |
| 无序子总体 | $\binom n r = \frac {n!}{(n-r)!r!}$      | $\binom {n+r-1} r$ |

* 有序有放回最简单，每次n种可能，进行r次抽样，所以是$n^r$
* 有序无放回从n个总体中选择出大小为r的有序样本，所以$\binom n r A_r^r = \binom n r r! = \frac {n!}{(n-r)!}$
* 无序无放回和有序无放回类似，只不过抽出的是一个大小为r的子总体而不是有序样本。
* 有放回的无序抽样最复杂。可以理解为在n个元素上放入r个标记。把元素的边界当成一个元素考虑，那么n个盒子共有n+1个边界，共有r个标记。现在除去两侧的边界，一共有n-1+r个空位。从这些空位中选出r个来放置标记。所以是$\binom {n-1+r} r$



##### 常见组合问题

* 大小为n的总体，有放回抽样出大小为r的有序样本：

  $\displaystyle n^r$

* 大小为n的总体，无放回抽样出大小为r的有序样本：

  $\displaystyle (n)_r=n(n-1)\cdots(n-r+1)=\frac{n!}{(n-r)!} = C_n^r A_n^r = \binom n r r !$

* 大小为n的总体，有放回抽样出大小为r的子总体：

  $\displaystyle \binom n r = \frac{(n)_r}{r!} = C_n^r = \frac{n!}{(n-r)!r!}$

* 大小为n的总体，无放回抽样出大小为r的子总体：

  $\displaystyle \binom {n-1 +r} r$

* 大小为n的总体划分为k组，每组个数为$r_1,\cdots, r_k$：

  $\displaystyle \frac{n!} {r_1!r_2!\cdots r_k!}$

* 大小为n的总体里有m个阳性样本，无放回抽样出大小为r的子总体，其中出现k个阳性样本的概率：

  $\displaystyle \frac{\binom{m}{k} \binom{n-m}{r-k}}{\binom{n}{r}}$



## 3. 条件概率与独立性

##### 定义：条件概率

设A,B为S重的时间，且$P(B) > 0$ ，则在事件B发生的条件下事件A发生的条件概率记作$P(A |B)$表示为：
$$
\displaystyle
P(A|B) = \frac {P(A \cap B) } {P(B)}
$$
直觉上很好理解，AB共同发生的概率等于B发生的概率 乘以B发生条件下A发生的概率：$P(AB) = P(A|B)P(B)$

自然而然，A在B条件下的发生概率为：AB共同发生概率 除以 B的发生概率。这里事件B的样本点构成了新的样本空间，而P(A|B)也一定满足概率三公理，构成新样本空间上的一个概率函数。



##### 定理：Bayes公式

设$A_1,A_2,\cdots$为样本空间的一个划分，B为任意集合，则对$i=1,2,\cdots$，有：
$$
\displaystyle
P(A_i | B) = \frac
{P(B|A_i)P(A_i)}
{\sum_{j=1}^{\infty}{P(B|A_j)P(A_j)}}
$$


##### 定义：统计独立

称事件A，B统计独立(statistically independent)，如果$P(A \cap B) = P(A)P(B)$

称一系列事件$A_1,\cdots, A_n$相互独立(mutually independent)，如果对于任意$A_{i_1},\cdots,A_{i_k}$都有：
$$
\displaystyle
P( \bigcap_{j=1}^{k}{A_{i_j}}) = \prod_{j=1}^{k}P(A_{i_j})
$$


## 4. 随机变量

许多试验中存在一个具有概括作用的变量，它处理起来比原概率模型要简单的多。

例如：50个人表决的结果，样本空间为$2^{50}$。其实我们感兴趣的只不过是有多少人赞成，那么定义变量X=赞成个数，样本空间就变成了整数集合：$\{s| 0 \le s \le 50 \wedge s \in \mathbb{Z} \}$

##### 定义：随机变量

从样本空间映射到实数的函数称为**随机变量(random variable)**

定义了随机变量，也就定义了一个新的样本空间（随机变量的值域）。但更重要的是，我们要通过原来样本空间上定义的**概率函数**，定义出这个**随机变量的概率函数**：诱导概率函数$P_X$。

假设有样本空间$S=\{s_1,\cdots, s_n\}$以及概率函数P，定义随机变量X的值域为：$\mathcal{X} = \{x_1,\cdots, x_n\}$。我们可以如下定义$\mathcal{X}$上的概率函数$P_X$：观测到事件$X=x_i$发生当且仅当随机试验的结果$s_j \in S$满足$X(s_j)=x_i$，即：
$$
\displaystyle
P_x (X=x_i) = P(\{s_j \in S : X(S_j) =x_i\})
$$
因为$P_X$是通过已知的概率函数P得到的，所以称之为$\mathcal{X}$上的**诱导概率函数**，易证该函数也满足概率公理。

对于连续的样本空间S，情况类似：
$$
\displaystyle
P_x (X \in A) = P(\{s_j \in S : X(S_j) \in A\})
$$

## 5. 分布函数

对于任意随机变量，我们都可以构造一个函数：**累积分布函数(cumulative distribution function)**，简称CDF。

##### 定义：累积分布函数

随机变量X的累积分布函数，记作$F_X(x)$，表示：$F_X(x) = P_X(X \le x)$

X的分布为$F_X$，可以简记作：$X \sim  F_X(x)$，其中“~”读作分布如。



##### 例：掷硬币

同时投掷三枚硬币，令X=正面朝上的硬币数，则X的累积分布函数是一个阶梯函数：
$$
\displaystyle
F_X(x) = \left\{
\begin{aligned}
0     & &  -\infty < x < 0 \\
1/8 & & 0 \le x < 1 \\
1/2 & &  1 \le x < 2\\
7/8 & &  2 \le x < 3\\
1 & &  3 \le x < \infty\\
\end{aligned}
\right.
$$
由累积分布函数的定义可知，$F_X(x)$是**右连续**的。



##### 性质：累积分布函数

函数$F(x)$是一个累积分布函数，当且仅当它同时满足下列三个条件。

* $\displaystyle \lim_{x\rightarrow -\infty}{F(x)} = 0$ 且 $\displaystyle \lim_{x\rightarrow \infty}{F(x)} = 1$
* $F(x)$是$x$的单调递增函数
* $F(x)$右连续：$\displaystyle  \forall x_0 ( \lim_{x\rightarrow x_0^+}{F(x) } = F(x_0) )$



##### 定义：离散/连续随机变量

设X为一随机变量，如果$F_X(x)$是x的连续函数，则称X是**连续的(continuous)**；如果$F_X(x)$是x的阶梯函数，则称X是**离散(discrete)**的。



累积分布函数$F_X$能够完全确定随机变量X的概率分布。所以引出了随机变量同分布的概念。



##### 定义：随机变量同分布

称随机变量X和Y**同分布(identically distributed)**，如果对任意集合$A \in \mathcal{B}^1$，都有$P(X\in A)=P(Y\in A)$

注意两个同分布的随机变量并不表示 $X=Y$，比如令XY分别为连掷三次硬币正反面朝上的次数。



##### 定理：同分布随机变量的性质

随机变量X与Y同分布，当且仅当 $\forall x ( F_X(x) = F_Y(x))$





## 6. 概率密度函数与概率质量函数

与随机变量X，累积分布函数$F_X$相关的还有一个函数：若X是连续随机变量，该函数称作概率密度函数；若X是离散随机变量，该函数称作概率质量函数。它们关注的都是随机变量的“点概率”。



##### 定义：概率质量函数(probability mass function) 简称pmf

离散随机变量X的概率质量函数定义为：
$$
\displaystyle
\forall x (f_X(x) = P_X(X=x))
$$
概率质量函数的集合解释：$P_X(X=x),i.e f_X(x)$等于累积分布函数在x处的跃变高度。



推广到连续变量的情景，则有：
$$
\displaystyle
P(X\le x) = F_X(x) = \int_{-\infty}^{x}{f_X(t)dt}
$$

##### 定义：概率密度函数(probability density function)，pdf

连续随机变量X的概率密度函数，是满足下式的函数：
$$
\displaystyle
F_X(x) = \int_{-\infty}^{x}{f_X(t)dt}, x任意
$$


##### 定理：PDF/PMF的性质

函数$f_X(x)$是随机变量X的概率密度函数（或概率质量函数），当且仅当它同时满足以下两个条件

* $\forall x ( f_X(x) \ge 0)$
* $\sum_x {f_X(x) = 1}$  （概率质量函数）或  $\int_{-\infty}^{\infty}{f_X(x)dx} = 1$ （概率密度函数）