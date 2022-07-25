from torch.distributions import Normal, kl, studentT
import torch
import math

# Note: KL Div is not symmetric. So choice of which distribution is given as 1st arg is important.

@kl.register_kl(studentT.StudentT, Normal)
def kl_tstud_normal(gt_tstudent, out_gaussian):
    # Calculating KL-divergence based of Information theory
    # i.e. KL(p, q) = H(p,q) - H(p)
    gt_entropy = gt_tstudent.entropy()
    gt_std_square = torch.square(gt_tstudent.stddev)
    out_std_square = torch.square(out_gaussian.stddev)
    mean_diffsqr = torch.square(gt_tstudent.mean - out_gaussian.mean)

    pq_cross_entropy = (torch.log(2*math.pi*gt_std_square)/2) + ((out_std_square+mean_diffsqr)/(2*gt_std_square))
    div = pq_cross_entropy - gt_entropy
    return div


def kl_dist_dist(gt, out):
    div = kl.kl_divergence(gt, out)
    return div
