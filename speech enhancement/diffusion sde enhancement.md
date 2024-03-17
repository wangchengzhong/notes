Next, we derive the objective function used for training the score model $\mathrm{s}_\theta$. Since the SDE in (2) describes a Gaussian process, the mean and variance of the process state $\mathbf{x}_t$ can be derived when its initial conditions are known [41]. This allows for direct sampling of $\mathbf{x}_t$ at an arbitrary time step $t$ given $\mathbf{x}_0$ and y by using the so-called perturbation kernel,
$$
p_{0 t}\left(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{y}\right)=\mathcal{N}_{\mathbb{C}}\left(\mathbf{x}_t ; \boldsymbol{\mu}\left(\mathbf{x}_0, \mathbf{y}, t\right), \sigma(t)^2 \mathbf{I}\right),
$$
where $\mathcal{N}_{\mathbb{C}}$ denotes the circularly-symmetric complex normal distribution
---------------
请解释什么是circularly-symmetric complex normal distribution


圆对称复正态分布（circularly-symmetric complex normal distribution）是一种特殊的复数值随机变量分布。在这种分布中，如果一个复数随机变量 $Z = X + iY$（其中 $X$ 和 $Y$ 是实数随机变量，$i$ 是虚数单位），那么 $Z$ 遵循圆对称复正态分布的条件是 $X$ 和 $Y$ 独立且具有相同的正态分布，通常是均值为0的正态分布。这意味着随机变量 $Z$ 的幅度 $|Z|$ 和相位 $\angle Z$ 是独立的，且其相位在 $[0, 2\pi)$ 范围内均匀分布，而幅度的平方 $|Z|^2$ 遵循指数分布。
简而言之，圆对称复正态分布的特点是，其分布在复平面上关于原点是圆对称的，这意味着从原点到分布中任意点的距离（即随机变量的幅度）遵循特定的分布，而这些点的角度（即随机变量的相位）在整个圆周上是均匀分布的。

圆对称复正态分布在相位上呈均匀分布，是因为该分布的定义本身就保证了在复平面上的对称性。具体来说，当一个复数随机变量 $Z = X + iY$ 遵循圆对称复正态分布时，其实部 $X$ 和虚部 $Y$ 是独立且具有相同分布的正态随机变量。这种结构导致了 $Z$ 的幅度和相位的特定分布特性：
1. 幅度：$Z$ 的幅度 $|Z| = \sqrt{X^2 + Y^2}$ 由于 $X$ 和 $Y$ 都是正态分布，且独立同分布，$|Z|^2$ 遵循卡方分布，进一步导致 $|Z|$ 遵循瑞利分布。这意味着 $Z$ 的幅度依赖于 $X$ 和 $Y$ 的分布，但与其相位无关。
2. 相位：$Z$ 的相位 $\theta = \arctan\left(\frac{Y}{X}\right)$，由于 $X$ 和 $Y$ 独立且具有相同的正态分布，这保证了无论 $Z$ 指向复平面上的哪个方向，其概率密度都是相同的。换句话说，$Z$ 的方向（或相位）在 $[0, 2\pi)$ 范围内是均匀分布的。这种均匀分布的相位是圆对称性的直接结果，因为在复平面上，从原点出发，任意方向的概率都是相等的。
因此，圆对称复正态分布的相位呈均匀分布，是因为其定义确保了在复平面上任意方向上的概率密度都是相同的，从而使得相位在整个可能的范围内均匀分布


求解微分方程：
$$
\frac{d \mathbf{m}}{d t} = \gamma(\mathbf{y}-\mathbf{x}_t)
$$，
其初始条件为：
$\mathbf{m}(0) = \mathbf{x}_t$

要解这个微分方程，我们可以使用分离变量法或直接积分，因为这是一个一阶线性微分方程。方程形式为：
$$
\frac{d \mathbf{m}}{d t} = \gamma(\mathbf{y}-\mathbf{x}t)$$其
中，$\mathbf{m}(0) = \mathbf{x}_t$ 是初始条件。由于 $\mathbf{x}_t$ 是初始条件下的 $\mathbf{m}$ 的值，我们可以将方程简化为：$$\frac{d \mathbf{m}}{d t} = \gamma(\mathbf{y}-\mathbf{m})$$这是一个标准的一阶线性微分方程，可以通过以下步骤解决：1. 将方程重写为：$$\frac{d \mathbf{m}}{d t} + \gamma\mathbf{m} = \gamma\mathbf{y}$$
2. 这个方程的解可以直接写出，因为它是一阶线性微分方程的标准形式。解为：$$\mathbf{m}(t) = \mathbf{y} + (\mathbf{x}_t - \mathbf{y})e^{-\gamma t}$$这里，$\mathbf{m}(t) = \mathbf{y} + (\mathbf{x}_t - \mathbf{y})e^{-\gamma t}$ 是方程的通解，其中 $\mathbf{x}_t$ 是 $\mathbf{m}(0)$，即 $\mathbf{m}$ 在 $t=0$ 时的值。解的推导基于指数函数的性质和一阶线性微分方程的解法。这个解表明，随着时间 $t$ 的增加，$\mathbf{m}(t)$ 会指数级地接近 $\mathbf{y}$，速率由 $\gamma$ 决定。

要详细解释如何求解一阶线性微分方程
$$
\frac{d \mathbf{m}}{d t} + \gamma\mathbf{m} = \gamma\mathbf{y}
$$
我们可以采用积分因子的方法。这个方法涉及到找到一个函数，乘以原方程的两边，使得左边成为一个乘积的导数，从而可以直接积分。

步骤 1: 确定积分因子

对于方程 $\frac{d \mathbf{m}}{d t} + \gamma\mathbf{m} = \gamma\mathbf{y}$，积分因子是 $e^{\int \gamma dt} = e^{\gamma t}$。积分因子的选择基于方程的系数，使得乘以此因子后，方程的左侧可以写成一个乘积的导数形式。

步骤 2: 乘以积分因子

将整个方程两边同时乘以积分因子 $e^{\gamma t}$，得到：
$$
e^{\gamma t}\frac{d \mathbf{m}}{d t} + \gamma e^{\gamma t}\mathbf{m} = \gamma e^{\gamma t}\mathbf{y}
$$
步骤 3: 重写左侧
注意到左侧可以重写为导数的形式：
$$
\frac{d}{dt}(e^{\gamma t}\mathbf{m}) = \gamma e^{\gamma t}\mathbf{y}
$$
这是因为 $e^{\gamma t}\mathbf{m}$ 的导数等于 $e^{\gamma t}$ 对 $t$ 的导数乘以 $\mathbf{m}$ 加上 $e^{\gamma t}$ 乘以 $\mathbf{m}$ 对 $t$ 的导数。
步骤 4: 积分
接下来，对方程两边关于 $t$ 进行积分：
$$
\int \frac{d}{dt}(e^{\gamma t}\mathbf{m}) dt = \int \gamma e^{\gamma t}\mathbf{y} dt
$$
这给出：
$$
e^{\gamma t}\mathbf{m} = e^{\gamma t}\mathbf{y} + C
$$
其中 $C$ 是积分常数，可以是向量。
步骤 5: 解出 $\mathbf{m}$
为了解出 $\mathbf{m}$，我们将方程两边同时除以 $e^{\gamma t}$：
$$
\mathbf{m} = \mathbf{y} + Ce^{-\gamma t}
$$
步骤 6: 应用初始条件
最后，应用初始条件 $\mathbf{m}(0) = \mathbf{x}t$ 来确定常数 $C$：$$\mathbf{m}(0) = \mathbf{y} + C = \mathbf{x}_t$$所以，$C = \mathbf{x}_t - \mathbf{y}$。### 最终解将 $C$ 的值代入，得到最终解：$$\mathbf{m}(t) = \mathbf{y} + (\mathbf{x}_t - \mathbf{y})e^{-\gamma t}$$这个解说明了 $\mathbf{m}(t)$ 随时间如何以指数速率从初始状态 $\mathbf{x}_t$ 趋近于 $\mathbf{y}$，其中 $\gamma$ 控制了这个过程的速率。

已知：
$$
\mathbf{f}\left(\mathbf{x}_t, \mathbf{y}\right):=\gamma\left(\mathbf{y}-\mathbf{x}_t\right)
$$

$$
\begin{aligned}
\frac{\mathrm{d} \mathbf{P}}{\mathrm{d} t}= & \mathrm{E}\left[\mathbf{f}(\mathbf{x}_t, \mathbf{y})(\mathbf{x}-\mathbf{m})^{\top}\right]+\mathrm{E}\left[(\mathbf{x}-\mathbf{m}) \mathbf{f}^{\top}(\mathbf{x}_t, \mathbf{y})\right] \\
& +\mathrm{E}\left[\mathbf{L}(\mathbf{x}, t) \mathbf{Q} \mathbf{L}^{\top}(\mathbf{x}, t)\right]
\end{aligned}
$$
其中，$\mathbf{P}$是相关函数
$$
\boldsymbol{m}\left(\mathbf{x}_0, \mathbf{y}, t\right)=\mathrm{e}^{-\gamma t} \mathbf{x}_0+\left(1-\mathrm{e}^{-\gamma t}\right) \mathbf{y}
$$
$$
L(t):=\sigma_{\min }\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^t \sqrt{2 \log \left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)}
$$
--------------
将$\boldsymbol{m}\left(\mathbf{x}_0, \mathbf{y}, t\right)$,
$$
\mathbf{f}\left(\mathbf{x}_t, \mathbf{y}\right):=\gamma\left(\mathbf{y}-\mathbf{x}_t\right)
$$
代入$\mathbf{P}$式，如何解微分方程解出来结果是
$$
\sigma(t)^2=\frac{\sigma_{\min }^2\left(\left(\sigma_{\max } / \sigma_{\min }\right)^{2 t}-\mathrm{e}^{-2 \gamma t}\right) \log \left(\sigma_{\max } / \sigma_{\min }\right)}{\gamma+\log \left(\sigma_{\max } / \sigma_{\min }\right)}
$$

$$
p(x, t) \triangleq p(x(t))=\mathrm{N}(x(t) \mid m(t), P(t))
$$
-------------
这里的P(t)是指方差吗？


是的，这里的 $P(t)$ 通常指的是方差（或协方差矩阵，如果 $x(t)$ 是多维的）。在概率论和统计学中，正态分布（或高斯分布）通常用两个参数来描述：均值（mean）和方差（variance）。对于多变量情况，方差被推广为协方差矩阵。
在给定的表达式
$$
p(x, t) \triangleq p(x(t))=\mathrm{N}(x(t) \mid m(t), P(t))
$$
中，$p(x, t)$ 表示时间 $t$ 时刻随机变量 $x(t)$ 的概率密度函数，$\mathrm{N}(x(t) \mid m(t), P(t))$ 表示 $x(t)$ 在时间 $t$ 服从均值为 $m(t)$，方差（或协方差矩阵）为 $P(t)$ 的正态分布。这里，$m(t)$ 是均值函数，描述了随机变量的平均位置随时间的变化；而 $P(t)$ 描述了随机变量的分散程度或不确定性随时间的变化。

# 试求解
$$
\frac{d\mathbf{P}}{dt} = -2\gamma \mathbf{P} + A
$$

要解这个一阶线性微分方程
$$
\frac{d\mathbf{P}}{dt} = -2\gamma \mathbf{P} + A
$$
我们可以采用类似之前解微分方程的方法。这里，$\mathbf{P}$ 是我们要找的函数，$\gamma$ 是一个常数，$A$ 也是一个常量（或在更一般的情况下，可以是时间的函数，但在这里我们假设它是常量）。

步骤 1: 重写方程

方程已经是简化形式，直接进行下一步。

步骤 2: 解常系数线性微分方程

这个方程是一个常系数线性微分方程，可以直接解析求解。我们首先解齐次方程
$$
\frac{d\mathbf{P}}{dt} + 2\gamma \mathbf{P} = 0
$$
其解为
$$
\mathbf{P}h(t) = C e^{-2\gamma t}$$
其中，$C$ 是积分常数，可能是一个矩阵或向量，取决于 $\mathbf{P}$ 的维度。
### 步骤 3: 寻找特解
对于非齐次项 $A$，我们可以猜一个特解的形式。由于非齐次项是常数，我们可以假设特解 $\mathbf{P}_p(t)$ 也是常数，即不依赖于 $t$。将 $\mathbf{P}_p$ 代入原方程：$$0 = -2\gamma \mathbf{P}_p + A$$解得$$\mathbf{P}_p = \frac{A}{2\gamma}$$
### 步骤 4: 合并解
因此，原方程的通解为齐次解和特解的和：$$\mathbf{P}(t) = \mathbf{P}_h(t) + \mathbf{P}_p = C e^{-2\gamma t} + \frac{A}{2\gamma}$$
### 步骤 5: 
应用初始条件如果给定初始条件 $\mathbf{P}(0) = \mathbf{P}0$，我们可以解出 $C$：
$$
\mathbf{P}0 = C e^{0} + \frac{A}{2\gamma} = C + \frac{A}{2\gamma}$$
解得
$$C = \mathbf{P}0 - \frac{A}{2\gamma}
$$
最终解
将 $C$ 的值代入通解中，得到最终解：
$$
\mathbf{P}(t) = \left(\mathbf{P}_0 - \frac{A}{2\gamma}\right)e^{-2\gamma t} + \frac{A}{2\gamma}$$
这个解描述了 $\mathbf{P}$ 随时间的变化情况，其中 $\gamma$ 控制衰减速率，$A$ 影响长期稳态解。


# 试求解
$$
\frac{d\mathbf{P}}{dt} = -2\gamma \mathbf{P} + \sigma^t
$$


给定的微分方程是
$$
\frac{d\mathbf{P}}{dt} = -2\gamma \mathbf{P} + \sigma^t
$$
其中，$\gamma$ 是一个常数，$\sigma^t$ 表示 $\sigma$ 的 $t$ 次幂，这使得方程成为一个非齐次线性微分方程。这里的挑战在于非齐次项 $\sigma^t$ 随时间 $t$ 变化。

步骤 1: 解齐次方程

首先，我们解对应的齐次方程
$$
\frac{d\mathbf{P}h}{dt} = -2\gamma \mathbf{P}_h$$
其解为$$\mathbf{P}_h(t) = C e^{-2\gamma t}$$其中 $C$ 是积分常数。
### 步骤 2: 
寻找特解对于非齐次项 $\sigma^t$，我们需要找到一个特解 $\mathbf{P}_p(t)$。由于非齐次项的形式，我们可以尝试一个形如 $At^k$ 的解，其中 $A$ 和 $k$ 是待定系数。但是，由于非齐次项直接是 $\sigma^t$，一个更自然的选择是假设特解的形式为 $A\sigma^t$，其中 $A$ 是待定常数。将假设的特解形式代入原微分方程：$$\frac{d}{dt}(A\sigma^t) = -2\gamma A\sigma^t + \sigma^t$$计算左侧的导数：$$A\ln(\sigma)\sigma^t = -2\gamma A\sigma^t + \sigma^t$$为了满足方程，我们需要：$$A\ln(\sigma) = -2\gamma A + 1$$解得：$$A = \frac{1}{\ln(\sigma) - 2\gamma}$$因此，特解为：$$\mathbf{P}_p(t) = \frac{\sigma^t}{\ln(\sigma) - 2\gamma}$$
### 步骤 3: 
合并解将齐次解和特解合并，得到通解：$$\mathbf{P}(t) = C e^{-2\gamma t} + \frac{\sigma^t}{\ln(\sigma) - 2\gamma}$$### 步骤 4: 应用初始条件如果给定初始条件 $\mathbf{P}(0) = \mathbf{P}0$，可以解出 $C$：
$$
\mathbf{P}0 = C + \frac{1}{\ln(\sigma) - 2\gamma}$$
解得：
$$
C = \mathbf{P}0 - \frac{1}{\ln(\sigma) - 2\gamma}
$$
最终解
代入 $C$ 的值，得到最终解：
$$
\mathbf{P}(t) = \left(\mathbf{P}0 - \frac{1}{\ln(\sigma) - 2\gamma}\right)e^{-2\gamma t} + \frac{\sigma^t}{\ln(\sigma) - 2\gamma}
$$
这个解描述了 $\mathbf{P}$ 随时间的变化情况，其中 $\gamma$ 控制衰减速率，而 $\sigma^t$ 影响了 $\mathbf{P}$ 的增长模式。

-----------------------
# 试求解
$$
\int_0^{t}\sigma^\tau e^{2\gamma \tau}d\tau
$$

要解这个积分
$$
\int_0^{t}\sigma^\tau e^{2\gamma \tau}d\tau
$$
我们可以使用换元积分法。首先，注意到 $\sigma^\tau = e^{\ln(\sigma)\tau}$，所以原积分可以重写为
$$
\int_0^{t}e^{\ln(\sigma)\tau} e^{2\gamma \tau}d\tau = \int_0^{t}e^{(\ln(\sigma) + 2\gamma)\tau}d\tau
$$
设 $k = \ln(\sigma) + 2\gamma$，则积分变为
$$
\int_0^{t}e^{k\tau}d\tau
$$
这个积分的解为
$$
\frac{e^{k\tau}}{k}\Big|0^{t} = \frac{e^{kt} - 1}{k}$$
将 $k = \ln(\sigma) + 2\gamma$ 代回，得到原积分的解为$$\frac{e^{(\ln(\sigma) + 2\gamma)t} - 1}{\ln(\sigma) + 2\gamma} = \frac{\sigma^t e^{2\gamma t} - 1}{\ln(\sigma) + 2\gamma}$$
因此，原积分的解为
$$\int0^{t}\sigma^\tau e^{2\gamma \tau}d\tau = \frac{\sigma^t e^{2\gamma t} - 1}{\ln(\sigma) + 2\gamma}
$$