import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

class Gauss():
    def __init__(self, sigma_prior=1.0):
        self.mu_prior = 0.0
        self.sigma_prior = sigma_prior

    def kl_divergence(self):
        kl = self.gaussian_kl_divergence(self.weight, self.mu_prior, torch.sqrt(torch.exp(self.log_sigma2)+1e-8), self.sigma_prior)
        if hasattr(self, 'bias') and self.bias is not None:
            kl += self.gaussian_kl_divergence(self.bias, self.mu_prior, torch.sqrt(torch.exp(self.bias_log_sigma2)+1e-8), self.sigma_prior)
        return kl

    def gaussian_kl_divergence(self, mu, mu_prior, sigma, sigma_prior):
        kl = 0.5 * (2 * torch.log(sigma_prior / (sigma+1e-8)+1e-8) - 1 + (sigma /
             sigma_prior).pow(2) + ((mu_prior - mu) / (sigma_prior)).pow(2)).sum()
        return kl

class GaussLinear(nn.Linear, Gauss):
    def __init__(self, in_features, out_features, bias=True, sigma_prior=1.0):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Gauss.__init__(self, sigma_prior=sigma_prior)
        self.log_sigma2 = torch.nn.Parameter(torch.normal(torch.ones_like(
            self.weight) * (-10.), torch.ones_like(self.weight) * 1e-2), requires_grad=True)
        if self.bias is not None:
            self.bias_log_sigma2 = torch.nn.Parameter(torch.normal(torch.ones_like(
                self.bias) * (-10.), torch.ones_like(self.bias) * 1e-2), requires_grad=True)

        # This is to disable the weight decay
        for p in self.parameters():
            p._no_wd = True

    def forward(self, input):
        mu = F.linear(input, self.weight, self.bias)
        var = F.linear(input.pow(2), torch.exp(self.log_sigma2), torch.exp(self.bias_log_sigma2) if self.bias is not None else None).clamp(min=1e-8)
        std = torch.sqrt(var)
        noise = torch.randn_like(std)
        return mu + noise * std

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sigma_prior={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sigma_prior)

# This is such that dropout will always be turned on!
class Dropout(nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)

class DropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.5):
        super().__init__(in_features, out_features, bias)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return super().forward(self.dropout(x))

class DirichletLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        outputs = super().forward(x)
        if self.training:
            return outputs
        alphas = outputs.double().exp()
        dirichlet = Dirichlet(alphas)
        outputs = (dirichlet.sample()+1e-8).log()
        return outputs
