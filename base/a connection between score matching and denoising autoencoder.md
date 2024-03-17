请介绍parsen windows density estimator是什么，并且请介绍显式匹配(explicit matching)和隐式匹配
$$
\begin{aligned}
J_{I S M q_0}(\theta) & =\mathbb{E}_{q_0(\mathbf{x})}\left[\frac{1}{2}\|\psi(\mathbf{x} ; \theta)\|^2+\sum_{i=1}^d \frac{\partial \psi_i(\mathbf{x} ; \theta)}{\partial \mathbf{x}_i}\right] \\
& =\frac{1}{n} \sum_{t=1}^n\left(\frac{1}{2}\left\|\psi\left(\mathbf{x}^{(t)} ; \theta\right)\right\|^2+\sum_{i=1}^d \frac{\partial \psi_i\left(\mathbf{x}^{(t)} ; \theta\right)}{\partial \mathbf{x}_i}\right)
\end{aligned}
$$
的区别

Parzen窗口密度估计器（Parzen windows density estimator）是一种非参数密度估计方法，用于估计未知概率密度函数。它通过对数据点周围放置窗口（通常是高斯窗口）并对所有窗口的贡献求和来工作，从而得到整个数据集的密度估计。参数$\sigma$控制窗口的宽度，影响平滑程度。

显式匹配（Explicit Matching）指的是直接将模型的输出与目标值进行比较并最小化二者之间的差异。在上述背景中，显式匹配涉及到将模型得分$\psi(\mathbf{x} ; \theta)$与通过Parzen窗口密度估计器得到的目标得分$\frac{\partial \log q_\sigma(\tilde{\mathbf{x}})}{\partial \tilde{\mathbf{x}}}$进行比较，并通过最小化二者之间的欧氏距离来训练模型。

隐式匹配（Implicit Matching）则不直接比较模型输出与目标值，而是通过其他方式间接调整模型以达到目标。在能量分数匹配（ISM）的上下文中，隐式匹配涉及到最小化一个包含模型得分的能量函数，而不是直接与目标得分进行比较。
显式匹配和隐式匹配的主要区别在于，显式匹配直接对模型输出和目标进行比较和调整，而隐式匹配则通过间接的方式调整模型，可能不直接涉及到目标得分。在给定的数学表达式中，$J_{ESMq_\sigma}(\theta)$代表显式匹配的目标函数，直接比较模型得分与通过Parzen窗口估计得到的得分；而$J_{ISMq_0}(\theta)$则代表隐式匹配的情况，通过最小化一个包含模型得分的能量函数来间接调整模型。

详细介绍什么是Denoising score matching，以及与explicit score matching等价的原因

去噪得分匹配（Denoising Score Matching, DSM）是一种结合了得分匹配原理和去噪自编码器方法的目标函数。在去噪自编码器中，通过使用一对干净样本和被污染样本$(\mathbf{x}, \tilde{\mathbf{x}})$来训练模型，目的是使模型能够从被污染的输入$\tilde{\mathbf{x}}$恢复出干净的输入$\mathbf{x}$。DSM目标函数利用这一思想，通过最小化模型得分$\psi(\tilde{\mathbf{x}} ; \theta)$与给定条件下的对数概率密度梯度$\frac{\partial \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{\partial \tilde{\mathbf{x}}}$之间的欧氏距离，来训练模型。
DSM的直觉基础是，如果我们沿着某个被污染点$\tilde{\mathbf{x}}$的对数密度梯度$\psi$移动，理想情况下应该将我们引向干净样本$\mathbf{x}$。在考虑到使用高斯核时，我们有
$$
\frac{\partial \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{\partial \tilde{\mathbf{x}}}=\frac{1}{\sigma^2}(\mathbf{x}-\tilde{\mathbf{x}}) .
$$
这个方向$\frac{1}{\sigma^2}(\mathbf{x}-\tilde{\mathbf{x}})$明确地指向从$\tilde{\mathbf{x}}$回到干净样本$\mathbf{x}$的方向，我们希望$\psi$尽可能地匹配这个方向。
DSM与显式得分匹配（Explicit Score Matching）等价的原因在于，两者都旨在通过最小化模型得分与目标得分之间的差异来训练模型。在DSM中，目标得分是基于干净样本和被污染样本之间的关系得到的，这种方式隐含地引入了去噪的目标，即通过模型得分的方向引导被污染样本向干净样本移动。这与显式得分匹配的目标一致，即直接最小化模型得分与目标得分之间的差异，只不过在DSM中，目标得分是通过干净样本与被污染样本之间的关系动态定义的。因此，DSM可以被视为显式得分匹配的一种特殊情况，其中考虑了样本的去噪过程。


$\frac{\partial \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{\partial \tilde{\mathbf{x}}}$这个条件概率为什么能求出来
----------
条件概率密度$q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$通常假设为高斯分布，这是因为高斯分布在数学和统计建模中具有良好的性质，例如它由均值和方差完全确定，且形式简单，易于处理。在去噪得分匹配（DSM）的上下文中，$\tilde{\mathbf{x}}$表示被添加噪声的$\mathbf{x}$，因此$\tilde{\mathbf{x}}$相对于$\mathbf{x}$的分布可以自然地假设为以$\mathbf{x}$为均值的高斯分布。
具体来说，如果我们假设噪声是加性的且遵循高斯分布，那么$\tilde{\mathbf{x}} = \mathbf{x} + \epsilon$，其中$\epsilon \sim \mathcal{N}(0, \sigma^2I)$。因此，条件概率密度$q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$可以表示为：
$$
q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{\frac{d}{2}}} \exp\left(-\frac{\|\tilde{\mathbf{x}}-\mathbf{x}\|^2}{2\sigma^2}\right)
$$
其中$d$是数据的维度。
对上述条件概率密度取对数，然后对$\tilde{\mathbf{x}}$求偏导，我们得到：
$$
\frac{\partial \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{\partial \tilde{\mathbf{x}}} = -\frac{1}{\sigma^2}(\tilde{\mathbf{x}} - \mathbf{x})
$$
这个导数实际上指向从$\tilde{\mathbf{x}}$到$\mathbf{x}$的方向，其大小与两者之间的距离成正比，方向则指向$\mathbf{x}$，这与高斯分布的性质和去噪的直觉相符。因此，这个条件概率的梯度能够直接计算出来，是因为我们假设了$\tilde{\mathbf{x}}$相对于$\mathbf{x}$的分布是高斯的，且其数学形式允许我们容易地进行这样的计算。


而后，讨论了得分匹配（Score Matching）与去噪自编码器（Denoising Autoencoder）之间的联系，并提到了Kingma和LeCun在2010年引入的正则化得分匹配准则$J_{ISMreg}$。这个准则的动机是为了解决可能的不稳定性问题，通过近似$J_{ISMq_\sigma}$来实现。作者指出，他们提出的基于$q_\sigma$的四个准则（包括去噪自编码器在内）可以被视为正则化得分匹配的无近似形式的替代方案。
主要区别在于：
1. 训练样本的处理方式：在去噪自编码器训练中，通过污染训练样本（即从$q_\sigma$采样）来优化这些无近似的正则化准则的随机版本。而Kingma和LeCun优化$J_{ISMq_\sigma}$的近似，仅集中在训练样本上（即从$q_0$采样）。
2. 计算复杂性：$J_{ISMreg}$和其他ISM准则需要计算二阶导数，而提出的$J_{DSMq_\sigma}$准则的随机版本不需要，因此实现起来更简单。
这段话强调了通过引入核平滑的经验分布$q_\sigma$，展示了得分匹配和简单去噪自编码器之间的联系。同时，作者提出的基于$q_\sigma$的准则作为正则化得分匹配的无近似替代方案，不仅解决了可能的不稳定性问题，而且由于不需要计算二阶导数，使得实现更为简单。这为理解和应用得分匹配提供了新的视角和方法

介绍Variational Bound
----------------
变分界（Variational Bound）是机器学习和统计推断中的一个重要概念，尤其在变分推断（Variational Inference, VI）中扮演着核心角色。变分界是一种技术，用于近似复杂概率分布的性质，特别是当直接计算或推断这些性质不可行时。它通过引入一个简单的分布来近似目标复杂分布，并最小化这两个分布之间的差异，从而得到对目标分布的一个近似表示。
在变分推断中，我们通常关注的是后验分布$p(z|x)$，其中$z$是潜在变量，$x$是观测数据。直接计算后验分布往往是困难的，因为它涉及到对潜在变量的高维积分，这在计算上是不可行的。变分推断通过引入一个简单的参数化分布$q(z|\theta)$来近似后验分布$p(z|x)$，其中$\theta$是变分参数。
变分界的核心思想是最小化真实后验分布和近似分布之间的Kullback-Leibler (KL) 散度，即$KL(q(z|\theta) || p(z|x))$。由于KL散度直接最小化通常仍然很难，变分推断转而最大化证据下界（Evidence Lower BOund, ELBO），这是对模型证据（即观测数据的边缘概率）的下界。ELBO可以表示为：
$$
\text{ELBO}(\theta) = \mathbb{E}{q(z|\theta)}[\log p(x, z) - \log q(z|\theta)]$$这里，$\mathbb{E}{q(z|\theta)}[\cdot]$表示在近似分布$q(z|\theta)$下的期望值，$\log p(x, z)$是联合分布的对数，$\log q(z|\theta)$是近似分布的对数。
通过最大化ELBO，我们可以找到最佳的变分参数$\theta$，使得$q(z|\theta)$尽可能接近真实的后验分布$p(z|x)$。这个过程实际上是在寻找一个变分界，即ELBO，它为模型证据提供了一个下界估计，同时也为后验分布提供了一个近似。
总之，变分界是变分推断中的一个关键概念，它通过最大化ELBO来近似复杂的后验分布，从而使得在复杂模型中的推断变得可行。