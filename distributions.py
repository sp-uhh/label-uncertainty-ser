import constants as c
import torch
import math
import os

class Gaussian(object):
    # Reuse sampling code at Forward pass - Same as v1 forward pass
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        # print("Normal Gauss Prior -> , N1: (0, ", str(sigma1), "), N2: (0, ", str(sigma2), ")")
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        # Only epsilon sampled from N(0, 1)
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        # print("Scale Mix Gauss Prior -> , N1: (0, ", str(sigma1), "), N2: (0, ", str(sigma2), ")")
        self.gaussian1 = Gaussian(0, sigma1)
        self.gaussian2 = Gaussian(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()