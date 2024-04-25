import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
"""
features = torch.randn(10,160)
mean = torch.mean(features, dim=0)
cov = torch.cov(features)
print(mean.shape, cov.shape)
"""
pseudo_sample = torch.randn(10,160)
z_mu = pseudo_sample.mean(dim=0).unsqueeze(1).unsqueeze(0)
z_sigma = pseudo_sample.std(dim=0).unsqueeze(1).unsqueeze(0)

anomal_feat = torch.randn(10,160)
a_mu = anomal_feat.mean(dim=0).unsqueeze(1).unsqueeze(0)
a_sigma = anomal_feat.std(dim=0).unsqueeze(1).unsqueeze(0)

# make kl divergence

kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
anomal_loss = torch.nn.functional.kl_div(pseudo_sample, anomal_feat, reduction="none").mean()
print(anomal_loss)
"""
kl_loss = nn.KLDivLoss(log_target=True)

batch = 1
n_classes = 2
mask_res = 256
pseudo_label = torch.ones((batch, n_classes, mask_res, mask_res))

features = torch.randn(10,160)
# generatoe gaussian distribution
mean = torch.mean(features, dim=0).unsqueeze(1).unsqueeze(0)
std = torch.std(features, dim=0).unsqueeze(1).unsqueeze(0)
# maybe have to make class1 feature generator

radom_feature = torch.rand_like(features)
# generating class 1 feature from random features


generator_net = AnomalFeatureGenerator()
random_feature = torch.rand_like(features)
output = generator_net(random_feature)
loss = torch.nn.functional.mse_loss(features, output)
"""

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

generator_net = AutoencoderKL()
# feature perturbation ... ?

"""
# [1] position
gt = torch.randn(1,2,256,256)
gt[:,1,:,:] = 1
gt[ : ,1, : ,:254] = 0
anomal_map = gt[:,1,:,:].contiguous().unsqueeze(1)
anomal_map = anomal_map.flatten() #
non_zero_index = torch.nonzero(anomal_map).flatten()

# [2] raw feature
feat = torch.randn(1,160,256,256)
feat =torch.flatten((feat), start_dim=2).squeeze().transpose(1,0) # pixel_num, dim
anomal_feat = feat[non_zero_index,:] # [512,160]

mean = torch.mean(anomal_feat, dim=0).unsqueeze(1).unsqueeze(0)
std = torch.std(anomal_feat, dim=0).unsqueeze(1).unsqueeze(0)
print(mean.shape)

batch, dim = 1, 160
sample = torch.randn(batch, dim,256*256)
pseudo_sample = (mean + std * sample).view(batch, dim, 256,256).contiguous()
print(pseudo_sample.shape)

# [3] generate virtual feature



"""
#print(non_zero_index)
"""
class DiagonalGaussianDistribution(object):

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
        #

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
"""