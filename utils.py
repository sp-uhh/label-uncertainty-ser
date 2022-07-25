import torch

from distributions import ScaleMixtureGaussian
import constants as c


# Posterior intialization
def get_posterior_mu_init_range():
    range = (-.1, .1)
    return range
    
def get_posterior_rho_init_range():
    range = (-3, -2)
    return range

# Prior intialization
def get_prioir_init_dist(dist="gauss"):
    if dist == "gauss":
        return torch.distributions.Normal(0, c.PRIOR_VAR)
    elif dist == "mix-gauss":
        return ScaleMixtureGaussian(c.PI, c.SIGMA_1, c.SIGMA_2)