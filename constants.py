import torch
import math

# Prior P(w) constants
PRIOR_VAR = 1.0
PRIOR_DIST = "gauss"

# Posterior P(w|D) constnts
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])