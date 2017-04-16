# 统计分布：二项分布

[author]: # "Vonng (fengruohang@outlook.com)"
[tags]: # "分布，统计，数学"
[mtime]: #	"2017-03-28 15:46 "
二项分布详解。

----



## 概述

二项分布是n个独立的"是/非实验"中成功次数的离散概率分布，其中每次试验成功的概率为p。

| 指标     | 描述                                       |
| ------ | ---------------------------------------- |
| 中文名    | 二项分布                                     |
| 英文名    | Binomial                                 |
| 记号     | $B(n,p)$                                 |
| 参数$n$  | 试验次数，整数，$n \ge 0$                        |
| 参数$p$  | 成功概率，实数，$0 \le p \le 1$                  |
| 概率质量函数 | $\displaystyle \binom{n}{k} p^k(1-p)^{n-k}$ |
| 累积分布函数 | $I_{1-p}(n-\lfloor k\rfloor ,1+\lfloor k\rfloor)$ |
| 期望     | $np$                                     |
| 中位数    | $ {\{\lfloor np\rfloor ,\lceil np\rceil \}}$之一 |
| 众数     | $\{\lfloor (n+1)\,p\rfloor, {\lfloor (n+1)\,p\rfloor \!-1} \}$之一 |
| 方差     | $np(1-p)$                                |
| 偏度     | $\displaystyle {\frac {1-2\,p}{\sqrt {n\,p\,(1-p)}}}$ |
| 峰度     | $ {\displaystyle {\frac {1-6\,p\,(1-p)}{n\,p\,(1-p)}}\!}$ |
| 信息熵    | $ {\displaystyle {\frac {1}{2}}\ln \left(2\pi nep(1-p)\right)+O\left({\frac {1}{n}}\right)\!}$ |
| 动差生成函数 | $ {\displaystyle (1-p+p\,e^{t})^{n}}$    |
| 特征函数   | $ {\displaystyle (1-p+p\,e^{i\,t})^{n}\!}$ |

* $n$次成功概率为$p$的伯努利实验，成功次数$k$符合二项分布$B(n,p)$。


## 概率质量函数 (PMF)

随机变量$X$服从参数为$n$和$p$的二项分布，则记作$X\sim b(n,p)$ 或 $X \sim B(n,p)$。n次试验中正好得到$k$次成功的概率由概率质量函数给出：
$$
{\displaystyle f(k | n,p)=\Pr(K=k)={n \choose k}p^{k}(1-p)^{n-k}}
$$

其中$\displaystyle \binom {n}{k} = \frac{n!}{k!(n-k)!}$称为二项式系数。

因为$f(k | n, p) = f(n-k | n, 1-p)$，所以二项分布的PMF表格通常只填写$n/2$个值。

函数$f(k|n,p)$当 $k < M$ 时单调递增，$k >M$时单调递减，只有当$(n+ 1)p$是整数时例外。在这时，有两个值使函数值达到最大：$(n + 1)$p和$(n + 1)p − 1$。M是伯努利试验的最可能的结果，称为[众数](https://zh.wikipedia.org/wiki/%E4%BC%97%E6%95%B0)。注意它发生的概率可以很小。

```python
from scipy.stats import binom

n,p = 10,0.5
x = np.arange(n)
y = binom.pmf(x,n,p)
plt.stem(x,y)
```



## 与其他分布的关系

* 二项分布之和仍是二项分布：两个符合二项分布的随机变量之和，也符合二项分布。

  $X \sim B(n,p)$ ，$Y ~ B(n,p)$，且X与Y相互独立，那么$X+Y \sim B(n+m ,p)$

* 伯努利试验的泛化：当参数$n=1$时，二项分布退化为伯努利分布。

  $X \sim B(1,p) \Leftrightarrow X \sim Bern(p)$

* 泊松二项分布的特例：泊松二项分布是n次独立独立，但每次概率p不相同的伯努利试验中成功次数的分布。当$p_1=p_2=\cdots=p_n = p$时，那么$X \sim B(n,p)$

* 正态近似：如果n足够大使得分布偏度足够小，则正态分布$N(np,np(1-p))$可以作为二项分布$B(n,p)$的一个近似。

  p越接近0.5越好，n越大越好。通常认为$n\ge20, np \ge 10 , n(1-p) \ge 10$的条件下可以采用正态近似。

* 泊松近似：当实验次数趋于无穷大而乘积np固定时，二项分布收敛于泊松分布。

  $π(np) \sim B(n,p)$，当n足够大，p足够小时。









