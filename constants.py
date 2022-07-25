import torch
import math

# Prior Constants
PRIOR_VAR = 1.0
PRIOR_DIST = "gauss"

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])