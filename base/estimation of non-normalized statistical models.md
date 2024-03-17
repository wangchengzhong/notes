For the model density, we denote the score function by $\boldsymbol{\psi}(\boldsymbol{\xi} ; \boldsymbol{\theta})$ :
$$
\psi(\boldsymbol{\xi} ; \boldsymbol{\theta})=\left(\begin{array}{c}
\frac{\partial \log p(\boldsymbol{\xi} ; \boldsymbol{\theta})}{\partial \xi_1} \\
\vdots \\
\frac{\partial \log p(\boldsymbol{\xi} ; \boldsymbol{\theta})}{\partial \xi_n}
\end{array}\right)=\left(\begin{array}{c}
\psi_1(\boldsymbol{\xi} ; \boldsymbol{\theta}) \\
\vdots \\
\psi_n(\boldsymbol{\xi} ; \boldsymbol{\theta})
\end{array}\right)=\nabla_{\boldsymbol{\xi}} \log p(\boldsymbol{\xi} ; \boldsymbol{\theta})
$$
-----------


这个式子定义了模型密度的得分函数 $\boldsymbol{\psi}(\boldsymbol{\xi} ; \boldsymbol{\theta})$。得分函数是关于模型参数 $\boldsymbol{\theta}$ 的梯度，用于评估参数的变化如何影响模型对数据的拟合程度。
$\boldsymbol{\xi}$ 表示数据点。
$\boldsymbol{\theta}$ 表示模型参数。
$p(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 是给定参数 $\boldsymbol{\theta}$ 下，数据点 $\boldsymbol{\xi}$ 的概率密度函数。
得分函数 $\psi(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 是通过对 $\log p(\boldsymbol{\xi} ; \boldsymbol{\theta})$（即概率密度函数的对数）关于数据点 $\boldsymbol{\xi}$ 的每个维度 $\xi_i$ 的偏导数来计算的。这个过程形成了一个向量，其中每个元素 $\psi_i(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 是对数概率密度函数关于 $\xi_i$ 的偏导数。
$\frac{\partial \log p(\boldsymbol{\xi} ; \boldsymbol{\theta})}{\partial \xi_i}$ 表示对数概率密度函数关于数据点 $\boldsymbol{\xi}$ 的第 $i$ 维度的偏导数。
$\nabla_{\boldsymbol{\xi}} \log p(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 表示对数概率密度函数关于 $\boldsymbol{\xi}$ 的梯度，即得分函数。
总的来说，得分函数提供了一个量化指标，表明在参数 $\boldsymbol{\theta}$ 下，如何通过调整数据点 $\boldsymbol{\xi}$ 来最大化模型的对数概率密度。这在统计模型的估计和优化中是非常重要的。


这段文字介绍了得分匹配估计器（Score Matching Estimator）的概念和原理，用于估计统计模型中的参数 $\boldsymbol{\theta}$，特别是在模型的归一化常数 $Z(\boldsymbol{\theta})$ 难以直接计算的情况下。

得分函数的独立性

首先，强调得分函数 $\boldsymbol{\psi}(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 不依赖于模型的归一化常数 $Z(\boldsymbol{\theta})$。这是因为得分函数是对数概率密度 $\log q(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 关于数据 $\boldsymbol{\xi}$ 的梯度，而归一化常数在对数转换后成为加法项，其梯度为零，因此不影响得分函数的计算。

得分匹配估计器

提出通过最小化模型得分函数 $\boldsymbol{\psi}(. ; \boldsymbol{\theta})$ 与数据得分函数 $\boldsymbol{\psi}_{\mathbf{x}}(.)$ 之间的期望平方距离 $J(\boldsymbol{\theta})$ 来估计模型参数 $\boldsymbol{\theta}$。这个期望平方距离定义为所有数据点 $\boldsymbol{\xi}$ 上，两个得分函数差的平方的期望值。
### 优化目标通过最小化 
$J(\boldsymbol{\theta})$ 来找到最佳的参数 $\hat{\boldsymbol{\theta}}$。这种方法的动机是直接从 $q$ 计算得分函数，无需计算 $Z$，简化了参数估计的难度。
### 简化计算
尽管看起来需要非参数估计来获取数据得分函数 $\psi_{\mathbf{x}}$，实际上可以通过部分积分的技巧简化目标函数 $J$ 的计算。定理 1 表明，在一定的可微性和弱正则条件下，目标函数 $J$ 可以重新表达为一个关于 $\boldsymbol{\xi}$ 的积分，其中包含模型得分函数的第一和第二偏导数项，以及一个与 $\boldsymbol{\theta}$ 无关的常数项。
定理 1 的含义
$\psi_i(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 是模型得分函数的第 $i$ 个元素，表示对数概率密度函数关于 $\xi_i$ 的偏导数。
$\partial_i \psi_i(\boldsymbol{\xi} ; \boldsymbol{\theta})$ 是 $\psi_i$ 关于 $\xi_i$ 的偏导数，即对数概率密度函数关于 $\xi_i$ 的二阶偏导数。
这种方法避免了直接从观测样本中估计数据得分函数的复杂性，使得参数估计更加高效和实用。

协方差矩阵 $\Sigma$ 是对称正定矩阵（Symmetric Positive-Definite Matrix, SPD），其逆矩阵 $\Sigma^{-1}$ 也是对称正定矩阵。这里给出证明：
对称性（Symmetric）
首先，协方差矩阵 $\Sigma$ 的定义是：
$$
\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]
$$
其中，$X_i$ 和 $X_j$ 是随机变量，$\mu_i$ 和 $\mu_j$ 是它们的期望值。
由于协方差的性质 $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$，我们可以得出 $\Sigma_{ij} = \Sigma_{ji}$，即协方差矩阵是对称的。
对于任意矩阵 $A$，如果 $A$ 是对称的，即 $A = A^T$，那么 $A$ 的逆矩阵 $A^{-1}$ 也是对称的。这是因为：
$$
(A^{-1})^T = (A^T)^{-1} = A^{-1}
$$
所以，如果 $\Sigma$ 是对称的，那么 $\Sigma^{-1}$ 也是对称的。

正定性（Positive-Definite）

一个矩阵 $A$ 是正定的，如果对于所有非零向量 $x$，都有 $x^T A x > 0$。
协方差矩阵 $\Sigma$ 是正定的，因为对于任意非零向量 $x$，我们有：
$$
x^T \Sigma x = \text{Var}(x^T X) > 0
$$

这里，$X$ 是一个随机向量，$\text{Var}(x^T X)$ 表示 $x^T X$ 的方差，它是大于零的，因为 $x^T X$ 是一个非常数随机变量（假设 $x$ 不全为零）。
对于 $\Sigma^{-1}$ 的正定性，我们可以利用以下性质：如果 $A$ 是正定的，那么 $A^{-1}$ 也是正定的。这是因为对于任意非零向量 $x$，我们有：
$$
x^T A^{-1} x = (A^{-1/2} x)^T A (A^{-1/2} x) > 0
$$
这里，$A^{-1/2}$ 是 $A$ 的平方根的逆矩阵，它存在且唯一，因为 $A$ 是正定的。上式成立是因为 $A$ 是正定的，所以 $(A^{-1/2} x)^T A (A^{-1/2} x)$ 是大于零的。
综上所述，协方差矩阵 $\Sigma$ 的逆矩阵 $\Sigma^{-1}$ 是对称正定矩阵。


已知
$$
\tilde{J}(\boldsymbol{\theta})=\frac{1}{T} \sum_{t=1}^T \sum_{i=1}^n\left[\partial_i \psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})+\frac{1}{2} \psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})^2\right]+\text { const. }
$$
---------
$$
q(\mathbf{x})=\exp \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{M}(\mathbf{x}-\boldsymbol{\mu})\right)
$$
and we obtain
$$
\boldsymbol{\psi}(\mathbf{x} ; \mathbf{M}, \boldsymbol{\mu})=-\mathbf{M}(\mathbf{x}-\boldsymbol{\mu})
$$
and
$$
\partial_i \psi(\mathbf{x} ; \mathbf{M}, \boldsymbol{\mu})=-m_{i i}
$$

Thus, we obtain
$$
\tilde{J}(\mathbf{M}, \boldsymbol{\mu})=\frac{1}{T} \sum_{t=1}^T\left[\sum_i-m_{i i}+\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right]
$$
----------------------
请解释，为什么最后一个式子的转置在前面而不是后面

在给出的最后一个式子中，
$$
\tilde{J}(\mathbf{M}, \boldsymbol{\mu})=\frac{1}{T} \sum_{t=1}^T\left[\sum_i-m_{i i}+\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right]
$$
转置符号的位置看似有误，但实际上，这里的表达式是正确的，原因如下：
1. 向量与矩阵的乘法：在表达式 $(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 中，$\mathbf{x}(t)-\boldsymbol{\mu}$ 是一个列向量，因此 $(\mathbf{x}(t)-\boldsymbol{\mu})^T$ 是一个行向量。这个行向量左乘矩阵 $\mathbf{M}$，然后结果再乘以 $\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$，这是标准的向量-矩阵乘法操作，用于计算二次形式。
2. 二次形式：表达式的形式实际上是一个二次形式，通常表示为 $\mathbf{x}^T A \mathbf{x}$，其中 $\mathbf{x}$ 是向量，$A$ 是矩阵。在这个场景中，$A = \mathbf{M}\mathbf{M}$，$\mathbf{x} = \mathbf{x}(t)-\boldsymbol{\mu}$。二次形式用于计算向量经过线性变换后的长度（或能量），在这里反映了数据点相对于均值的分布情况。
3. 转置的作用：转置操作 $(\mathbf{x}(t)-\boldsymbol{\mu})^T$ 是必要的，因为我们需要一个行向量乘以矩阵 $\mathbf{M}$，然后再乘以列向量 $\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$，以保证最终结果是一个标量。如果没有转置，向量和矩阵的乘法就无法按照预期进行。
因此，转置符号在前面是为了确保表达式遵循正确的数学规则，以及保证二次形式的计算是合法和有意义的。这并不是一个错误，而是符合线性代数中处理此类表达式的标准做法。

为什么对于$\frac{1}{2} \psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})^2$，对应分量的平方之和等于二次型$\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$
--------------
这个问题涉及到如何从得分函数 $\boldsymbol{\psi}(\mathbf{x} ; \boldsymbol{\theta})$ 的平方到达二次型表达式。已知得分函数 $\boldsymbol{\psi}(\mathbf{x} ; \mathbf{M}, \boldsymbol{\mu})=-\mathbf{M}(\mathbf{x}-\boldsymbol{\mu})$，我们来详细解释这个转换过程。

得分函数的平方

首先，考虑得分函数 $\boldsymbol{\psi}(\mathbf{x} ; \boldsymbol{\theta})$ 的平方，对于每个分量 $\psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})$，其平方是 $\psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})^2$。对所有分量求和，得到：
$$
\sum_{i=1}^n \frac{1}{2} \psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})^2
$$
将得分函数 $\boldsymbol{\psi}(\mathbf{x} ; \mathbf{M}, \boldsymbol{\mu})=-\mathbf{M}(\mathbf{x}-\boldsymbol{\mu})$ 代入上式，可以得到：
$$
\frac{1}{2} \sum_{i=1}^n \left(-\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right)i^2$$
这里，$\left(-\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right)_i$ 表示向量 $-\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 的第 $i$ 个分量。
### 二次型表达式
二次型表达式 $\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 可以看作是向量 $(\mathbf{x}(t)-\boldsymbol{\mu})$ 经过矩阵 $\mathbf{M}$ 线性变换两次后的长度（或能量）的度量。
### 等价性的解释
当我们计算 $\frac{1}{2} \sum_{i=1}^n \left(-\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right)_i^2$，实际上是在计算向量 $-\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 的各个分量的平方和，这与计算二次型 $\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 是等价的。这是因为二次型本质上就是度量经过线性变换的向量的长度（或能量），而这里的线性变换是由矩阵 $\mathbf{M}$ 实现的。具体来说，$\mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 是向量 $(\mathbf{x}(t)-\boldsymbol{\mu})$ 经过矩阵 $\mathbf{M}$ 的线性变换，再次应用 $\mathbf{M}$ 并计算其与自身的点积（即平方和），就得到了二次型表达式。这个过程实际上是在计算变换后向量的长度（能量），与直接计算得分函数的平方之和是等价的。因此，$\frac{1}{2} \psi_i(\mathbf{x}(t) ; \boldsymbol{\theta})^2$ 对应分量的平方之和等于二次型 $\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 是因为这两种计算方式都在衡量经过矩阵 $\mathbf{M}$ 线性变换的向量 $(\mathbf{x}(t)-\boldsymbol{\mu})$ 的“长度”或能量。在这个上下文中，二次型表达式提供了一种计算这种变换向量长度的方法，而得分函数的平方之和则是从另一个角度提供了相同的度量。这种等价性是线性代数中向量和矩阵操作的一个典型例子，展示了不同数学表达式之间的内在联系。

$$
\tilde{J}(\mathbf{M}, \boldsymbol{\mu})=\frac{1}{T} \sum_{t=1}^T\left[\sum_i-m_{i i}+\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right]
$$

To minimize this with respect to $\boldsymbol{\mu}$, it is enough to compute the gradient
$$
\nabla_{\boldsymbol{\mu}} \tilde{J}=\mathbf{M} \mathbf{M} \boldsymbol{\mu}-\mathbf{M} \mathbf{M} \frac{1}{T} \sum_{t=1}^T \mathbf{x}(t)
$$
--------------------
为什么计算该梯度就可以最小化

为了最小化 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 关于 $\boldsymbol{\mu}$，我们需要计算 $\tilde{J}$ 关于 $\boldsymbol{\mu}$ 的梯度，并将其设为零。这是因为在多变量函数的最优化问题中，函数的局部最小值（或最大值）出现在其梯度为零的点。梯度向量的每个分量表示函数在对应方向上的变化率，因此梯度为零意味着在所有方向上函数的变化率都为零，这通常对应于函数的极值点。
给定的梯度表达式
$$
\nabla_{\boldsymbol{\mu}} \tilde{J}=\mathbf{M} \mathbf{M} \boldsymbol{\mu}-\mathbf{M} \mathbf{M} \frac{1}{T} \sum_{t=1}^T \mathbf{x}(t)
$$
是通过对 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 关于 $\boldsymbol{\mu}$ 进行偏导数计算得到的。为了找到最小化 $\tilde{J}$ 的 $\boldsymbol{\mu}$ 值，我们需要解梯度等于零的方程：
$$
\mathbf{M} \mathbf{M} \boldsymbol{\mu} - \mathbf{M} \mathbf{M} \frac{1}{T} \sum_{t=1}^T \mathbf{x}(t) = 0
$$
解这个方程可以找到使得 $\tilde{J}$ 最小的 $\boldsymbol{\mu}$ 值。这个过程等价于找到一个点，在这个点上，目标函数 $\tilde{J}$ 关于 $\boldsymbol{\mu}$ 的增长率在所有方向上都为零，这通常意味着在该点 $\tilde{J}$ 达到了局部最小值。
因此，计算该梯度并将其设为零是找到最小化 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 的 $\boldsymbol{\mu}$ 值的必要步骤。这是最优化理论中的一个标准方法，用于寻找函数的极值点。

对 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 关于 $\boldsymbol{\mu}$ 进行偏导数计算，我们首先需要回顾 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 的定义：
$$
\tilde{J}(\mathbf{M}, \boldsymbol{\mu})=\frac{1}{T} \sum_{t=1}^T\left[\sum_i-m_{i i}+\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right]
$$
注意到，只有项 $\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})$ 依赖于 $\boldsymbol{\mu}$。因此，我们只需对这部分进行偏导数计算。
展开二次项，我们有：
$$
\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu}) = \frac{1}{2}(\mathbf{x}(t)^T\mathbf{M}\mathbf{M}\mathbf{x}(t) - 2\mathbf{x}(t)^T\mathbf{M}\mathbf{M}\boldsymbol{\mu} + \boldsymbol{\mu}^T\mathbf{M}\mathbf{M}\boldsymbol{\mu})
$$
对 $\boldsymbol{\mu}$ 的偏导数为：
$$
\nabla_{\boldsymbol{\mu}} \left[\frac{1}{2}(\mathbf{x}(t)-\boldsymbol{\mu})^T \mathbf{M} \mathbf{M}(\mathbf{x}(t)-\boldsymbol{\mu})\right] = -\mathbf{M}\mathbf{M}\mathbf{x}(t) + \mathbf{M}\mathbf{M}\boldsymbol{\mu}
$$
因此，对整个 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 的偏导数为：
$$
\nabla_{\boldsymbol{\mu}} \tilde{J} = \frac{1}{T} \sum_{t=1}^T[-\mathbf{M}\mathbf{M}\mathbf{x}(t) + \mathbf{M}\mathbf{M}\boldsymbol{\mu}]
$$
简化后得到：
$$
\nabla_{\boldsymbol{\mu}} \tilde{J} = \mathbf{M}\mathbf{M}\boldsymbol{\mu} - \mathbf{M}\mathbf{M}\frac{1}{T}\sum_{t=1}^T\mathbf{x}(t)
$$
这个结果直接来自于对 $\tilde{J}(\mathbf{M}, \boldsymbol{\mu})$ 中依赖于 $\boldsymbol{\mu}$ 的部分进行偏导数计算，并考虑到求和与微分操作的线性性质。

The maximum likelihood estimator is known to be consistent, so the score matching estimator is consistent as well.
----------------
为什么最大似然估计是一致的？

最大似然估计（MLE）的一致性是指随着样本量的增加，最大似然估计量会以概率1收敛到真实参数值。最大似然估计的一致性基于几个关键的数学假设和定理，主要包括：
1. 独立同分布（IID）：样本数据必须是独立同分布的。这意味着每个样本都是从相同的分布中独立抽取的，且这个分布的参数是固定的。
2. 参数空间的紧致性：参数空间需要是紧致的，即参数空间是封闭且有界的。这个条件确保了参数的最优值不会在边界之外。
3. 似然函数的连续性和充分光滑性：似然函数关于参数需要在参数空间内连续，且有连续的一阶和二阶导数。
4. 识别条件：真实参数值是唯一的，意味着不同的参数值会导致不同的分布。
5. 弱大数定律（Weak Law of Large Numbers）：随着样本量的增加，样本均值会以概率1收敛到期望值。
6. 中心极限定理（Central Limit Theorem）：对于足够大的样本量，样本均值的分布会趋近于正态分布，无论原始数据的分布如何。
基于这些条件，可以证明最大似然估计量是一致的。具体来说，根据似然方程的性质和大数定律，可以证明当样本量趋向于无穷大时，最大似然估计量会以概率1收敛到使似然函数最大化的参数值，即真实参数值。这个性质是通过考虑似然函数在真实参数值附近的行为，以及似然函数在这一点的最大值性质来证明的。
总的来说，最大似然估计的一致性是统计学中的一个重要性质，它保证了在合适的条件下，随着样本量的增加，我们可以越来越准确地估计出模型的真实参数。