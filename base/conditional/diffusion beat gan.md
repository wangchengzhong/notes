这段内容描述的是如何将一个无条件的逆噪声扩散模型 $p_\theta\left(x_t \mid x_{t+1}\right)$ 条件化到一个标签 $y$ 上。这个过程通过对每个转换采样来实现，采样公式为
$$
p_{\theta, \phi}\left(x_t \mid x_{t+1}, y\right)=Z p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)
$$
其中，$Z$ 是一个归一化常数（附录H中有证明）。直接从这个分布中采样通常是不可行的，但Sohl-Dickstein等人展示了如何将其近似为一个扰动的高斯分布。
接下来回顾了扩散模型是如何使用高斯分布来预测前一时间步 $x_t$ 从时间步 $x_{t+1}$ 的过程：
$$
\begin{aligned}
p_\theta\left(x_t \mid x_{t+1}\right) & =\mathcal{N}(\mu, \Sigma) \\
\log p_\theta\left(x_t \mid x_{t+1}\right) & =-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+C
\end{aligned}
$$
这里，$\mathcal{N}(\mu, \Sigma)$ 表示高斯分布，$\mu$ 和 $\Sigma$ 分别是高斯分布的均值和协方差矩阵。$\log p_\theta\left(x_t \mid x_{t+1}\right)$ 的表达式展示了如何计算给定 $x_{t+1}$ 时 $x_t$ 的对数概率，其中 $C$ 是一个常数，用于保证概率的归一化。
简而言之，这段内容介绍了如何通过结合无条件的逆噪声扩散过程和一个条件概率模型 $p_\phi\left(y \mid x_t\right)$ 来实现条件扩散模型，以及如何使用高斯分布来近似这个过程。

-----------------

We can assume that $\log _\phi p\left(y \mid x_t\right)$ has low curvature compared to $\Sigma^{-1}$. This assumption is reasonable in the limit of infinite diffusion steps, where $\|\Sigma\| \rightarrow 0$. In this case, we can approximate $\log p_\phi\left(y \mid x_t\right)$ using a Taylor expansion around $x_t=\mu$ as
$$
\begin{aligned}
\log p_\phi\left(y \mid x_t\right) & \left.\approx \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}+\left.\left(x_t-\mu\right) \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu} \\
& =\left(x_t-\mu\right) g+C_1
\end{aligned}
$$We can assume that $\log _\phi p\left(y \mid x_t\right)$ has low curvature compared to $\Sigma^{-1}$. This assumption is reasonable in the limit of infinite diffusion steps, where $\|\Sigma\| \rightarrow 0$. In this case, we can approximate $\log p_\phi\left(y \mid x_t\right)$ using a Taylor expansion around $x_t=\mu$ as
$$
\begin{aligned}
\log p_\phi\left(y \mid x_t\right) & \left.\approx \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}+\left.\left(x_t-\mu\right) \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu} \\
& =\left(x_t-\mu\right) g+C_1
\end{aligned}
$$
------------
为什么在相对低曲率的条件下做出这种近似，如果高曲率就不行吗

在假设 $\log \phi p\left(y \mid x_t\right)$ 相对于 $\Sigma^{-1}$ 曲率较低的条件下进行泰勒展开近似，是因为这样的假设允许我们将复杂的条件概率 $\log p\phi\left(y \mid x_t\right)$ 简化为一阶线性形式，从而便于计算和分析。这种近似在 $\|\Sigma\| \rightarrow 0$，即无限扩散步骤的极限情况下是合理的，因为在这种情况下，$x_t$ 的分布趋向于非常集中（即方差趋向于0），使得在 $\mu$ 附近的小范围内，$\log p_\phi\left(y \mid x_t\right)$ 可以被认为是相对平坦的，即曲率较低。
如果 $\log \phi p\left(y \mid x_t\right)$ 的曲率相对于 $\Sigma^{-1}$ 高，那么在 $\mu$ 附近的泰勒展开将不能很好地近似 $\log p\phi\left(y \mid x_t\right)$。这是因为高曲率意味着函数值在 $\mu$ 附近变化较快，一阶泰勒展开（即线性近似）将无法准确捕捉这种快速变化。在这种情况下，可能需要考虑更高阶的项来获得更准确的近似，但这会增加计算的复杂性和难度。
简而言之，低曲率条件下的近似是为了简化计算和分析，而在高曲率情况下，这种简化可能不再有效，需要采用更复杂的方法来准确描述函数的行为。


-------------
这个转换涉及到了一些代数操作和矩阵恒等式的应用。我们从原始的表达式开始：
$$
-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+\left(x_t-\mu\right) g+C_2
$$
我们的目标是将其重写为一个完全平方形式。首先，注意到 $\left(x_t-\mu\right) g$ 可以写作 $\frac{1}{2}2\left(x_t-\mu\right)^T g$，这样做是为了与前面的项保持一致的矩阵乘法形式。接下来，我们利用完全平方公式的思想，尝试将表达式重写为一个完全平方形式。
完全平方公式是：
$$
a^2 + 2ab + b^2 = (a + b)^2
$$
对应到我们的情况，我们可以将 $-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)$ 视为 $a^2$ 的形式，将 $2\left(x_t-\mu\right)^T g$ 视为 $2ab$ 的形式。为了形成完全平方，我们需要添加和减去一个 $b^2$ 的形式，这里 $b^2$ 对应于 $\frac{1}{2} g^T \Sigma g$。
因此，我们添加 $\frac{1}{2} g^T \Sigma g$ 并从 $C_2$ 中减去相同的量（或者可以认为是将其包含在新的常数 $C_3$ 中），以保持等式的平衡。这样，原始表达式就可以重写为：
$$
-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+\left(x_t-\mu\right) g+\frac{1}{2} g^T \Sigma g+C_2
$$
这等价于：
$$
-\frac{1}{2}\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)+\frac{1}{2} g^T \Sigma g+C_2
$$
这里，$\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)$ 形成了完全平方形式，而 $\frac{1}{2} g^T \Sigma g$ 是我们为了形成完全平方而添加和减去的项，最后 $C_2$ 是常数项。
这个转换的关键在于利用矩阵运算和完全平方公式的思想，将原始表达式重写为一个更易于理解和处理的形式。


