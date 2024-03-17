在变分推断中，选择使用 $D_{\mathrm{KL}}\left(q_\phi \mid p_\theta\right)$（逆向KL散度）而不是 $D_{\mathrm{KL}}\left(p_\theta \mid q_\phi\right)$（正向KL散度）的原因主要有以下几点：
1. 零避免（Zero Avoidance）：逆向KL散度倾向于避免将概率分布$q_\phi(z)$分配给$p_\theta(z)$为零的区域。这意味着，如果目标分布$p_\theta(z)$在某些区域几乎为零，逆向KL散度会鼓励$q_\phi(z)$也在这些区域接近零，从而避免不必要的惩罚。这有助于$q_\phi(z)$更集中地近似$p_\theta(z)$的主要质量区域，而不是尝试覆盖整个分布，这在实际应用中往往是不必要的。
2. 模式寻找（Mode Seeking）：逆向KL散度具有模式寻找的性质，这意味着它倾向于选择一个$q_\phi(z)$，该分布紧密地围绕$p_\theta(z)$的一个或几个高概率区域（模式）。这与正向KL散度形成对比，后者是模式覆盖的，倾向于让$q_\phi(z)$覆盖$p_\theta(z)$的所有可能模式，这在多模态分布的情况下可能导致$q_\phi(z)$过于分散，无法捕捉到$p_\theta(z)$的关键特征。
3. 计算效率：在实际应用中，使用逆向KL散度可以更容易地进行优化，因为它涉及到的期望是关于变分分布$q_\phi(z)$的，这通常是一个简单的分布（如高斯分布），使得相关的积分更容易计算。而正向KL散度的期望是关于目标分布$p_\theta(z)$的，这可能是复杂且难以处理的。

$$
\begin{aligned}
L_{\mathrm{VAE}}(\theta, \phi) & =-\log p_\theta(\mathbf{x})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right) \\
& =-\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z})\right)
\end{aligned}
$$
----------------------
通过VLBO推导出上式。在实际的VAE中，$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})$应该怎么得到

在变分自编码器（VAE）中，$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})$ 是重构损失的期望，它衡量了由编码器产生的潜在变量$\mathbf{z}$通过解码器重构出的数据$\mathbf{x}$与原始数据$\mathbf{x}$之间的相似度。在实际的VAE实现中，这个期望通常通过蒙特卡洛（Monte Carlo）估计来近似得到，步骤如下：
1. 采样：从变分后验分布$q_\phi(\mathbf{z} \mid \mathbf{x})$中采样$N$个潜在变量样本$\{\mathbf{z}^{(1)}, \mathbf{z}^{(2)}, ..., \mathbf{z}^{(N)}\}$。在VAE中，这通常通过重参数化技巧（reparameterization trick）实现，即首先从一个标准正态分布中采样$\epsilon^{(i)}$，然后通过变分后验的参数（例如均值$\mu$和方差$\sigma^2$）转换为$\mathbf{z}^{(i)} = \mu + \sigma \odot \epsilon^{(i)}$，其中$\odot$表示元素乘法。
2. 重构：对于每个潜在变量样本$\mathbf{z}^{(i)}$，使用解码器$p_\theta(\mathbf{x} \mid \mathbf{z})$计算重构出的数据$\hat{\mathbf{x}}^{(i)}$。
3. 计算重构损失：对于每个样本$\mathbf{z}^{(i)}$和对应的重构$\hat{\mathbf{x}}^{(i)}$，计算重构损失$\log p_\theta(\mathbf{x} \mid \mathbf{z}^{(i)})$。这通常涉及到比较原始数据$\mathbf{x}$和重构数据$\hat{\mathbf{x}}^{(i)}$之间的差异，例如通过平方误差（对于连续数据）或交叉熵损失（对于离散数据）。
4. 估计期望：最后，通过对所有$N$个重构损失的平均值来估计期望$\mathbb{E}{\mathbf{z} \sim q\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})$：
$$
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z}) \approx \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(\mathbf{x} \mid \mathbf{z}^{(i)})
$$
在实际应用中，通常$N=1$就足够了，因为每次前向传播时都会采样不同的$\mathbf{z}$，这样在整个训练过程中就相当于进行了多次采样，从而可以有效地估计这个期望。


```python
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
```


这行代码计算的是变分自编码器（VAE）中的KL散度损失（KL项损失），它衡量的是变分后验分布$q_\phi(\mathbf{z}|\mathbf{x})$（由编码器参数化）与先验分布$p(\mathbf{z})$（通常假设为标准正态分布）之间的差异。具体来说，这个公式是KL散度在假设变分后验分布为高斯分布，且先验分布也为高斯分布时的解析解。
给定变分后验分布$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \mu, \sigma^2I)$和先验分布$p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; 0, I)$，KL散度的公式可以推导为：
$$
D_{KL}(q_\phi(\mathbf{z}|\mathbf{x})\ ||\ p(\mathbf{z})) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)
$$
其中，$J$是潜在空间的维度，$\mu$和$\sigma^2$分别是变分后验分布的均值和方差（注意这里使用$\log(\sigma^2)$是为了数值稳定性和计算方便，实际代码中使用的是log_var表示$\log(\sigma^2)$）。
这个公式的直观解释是：KL散度损失鼓励变分后验分布$q_\phi(\mathbf{z}|\mathbf{x})$接近先验分布$p(\mathbf{z})$。具体来说，它由四部分组成：
$1$：常数项，确保当$\sigma^2 = 1$且$\mu = 0$时，KL散度为0。
$\log(\sigma_j^2)$：鼓励方差$\sigma^2$不要偏离1太远。
$-\mu_j^2$：鼓励均值$\mu$接近0。
$-\sigma_j^2$：同样鼓励方差$\sigma^2$接近1。