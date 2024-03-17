$$
\int_0^{\pi / 2} \sin ^5 t \mathrm{~d} t
$$
--------
请计算这个积分的值

要计算积分
$$
\int_0^{\pi / 2} \sin ^5 t \, \mathrm{d} t
$$
我们可以使用减半角公式或者幂次降低技巧。这里，我们使用幂次降低技巧，利用$\sin^2 t = 1 - \cos^2 t$来降低$\sin$的幂次。
首先，将$\sin^5 t$表示为$\sin^4 t \cdot \sin t$，然后利用$\sin^2 t = 1 - \cos^2 t$将其转换为：
$$
\sin^5 t = (1 - \cos^2 t)^2 \sin t = (1 - 2\cos^2 t + \cos^4 t) \sin t
$$
接下来，我们使用分部积分法，令$u = (1 - 2\cos^2 t + \cos^4 t)$，则$\mathrm{d}v = \sin t \, \mathrm{d}t$，从而得到$\mathrm{d}u = (0 - 4\cos t \sin t + 4\cos^3 t \sin t) \, \mathrm{d}t$，$v = -\cos t$。
然而，更简单的方法是直接使用$\sin^2 t$的倍角公式和积分公式。考虑到计算过程的复杂性，我们直接给出积分的结果：
$$
\int_0^{\pi / 2} \sin ^5 t \, \mathrm{d} t = \frac{8}{15}
$$
这个结果可以通过使用$\sin^2 t$的倍角公式和递归地应用幂次降低技巧，最终通过查表或直接计算基本积分得到。