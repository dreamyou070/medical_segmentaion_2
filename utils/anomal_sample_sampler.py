import torch
import numpy as np
from typing import Tuple


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters: torch.Tensor, latent_dim):

        self.parameters = parameters # num
        self.mean = parameters.mean(dim=0).unsqueeze(0)
        self.var = parameters.var(dim=0).unsqueeze(0)
        self.std = torch.sqrt(self.var)
        self.logvar = torch.log(self.var)
        self.memory_iter = 0
        self.latent_dim = latent_dim

    def update (self, parameters):
        """ update with previous one and new parameters """
        # parameters = [num_samples, dim]
        self.parameters = self.parameters.detach()
        self.parameters = torch.cat([self.parameters, parameters], dim=0)

        # why this problem .. ?
        self.mean = self.parameters.mean(dim=0).unsqueeze(0)
        self.var = self.parameters.var(dim=0).unsqueeze(0)
        self.std = torch.sqrt(self.var)
        self.logvar = torch.log(self.var)
        self.memory_iter += 1
        if self.memory_iter> 70 :
            N = parameters.shape[0]
            self.parameters = self.parameters[N:,:]

    def sample(self, mask_res, device, weight_dtype):



        sample = torch.randn((mask_res * mask_res, self.latent_dim)).to(dtype=weight_dtype, device=device) # N, 160
        x = (self.mean + self.std * sample).permute(1, 0).contiguous()
        # [256*256, 160] -> [160,256*256] -> [160, 256, 256]
        x = x.view(self.latent_dim, mask_res, mask_res).unsqueeze(0)
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
                    dim=[1, 2, 3],)

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