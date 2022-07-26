from torch.distributions import Normal, kl, studentT
import torch
import math

# Note: KL Div is not symmetric. So choice of which distribution is given as 1st arg is important.

@kl.register_kl(studentT.StudentT, Normal)
def kl_tstud_normal(p, q):
    # Calculating KL-divergence based of Information theory
    # i.e. KL(p, q) = H(p,q) - H(p)
    # p - t-student distribution
    # q - normal distribution
    # Loss between studentT ground-truth and Guassian stochastic predictions
    # For derivation, Check - https://arxiv.org/abs/2207.12135
    # Paper - Raj Prabhu et al., 
    # "Label Uncertainty Modeling and Prediction for Speech Emotion Recognition using t-Distributions", 
    # Affective Computing and Intelligent Interaction (ACII), Nara, Japan, Oct. 2022.

    p_entropy = p.entropy()
    q_entropy = q.entropy()

    q_std_square = torch.square(q.stddev)
    p_std_square = torch.square(p.stddev)

    log_2pi_qscale_square= torch.log(2*math.pi*q_std_square)
    meanp_meanq_diffsqr = torch.square(p.mean - q.mean)

    pq_cross_entropy = (log_2pi_qscale_square/2) + ((p_std_square+meanp_meanq_diffsqr)/(2*q_std_square))
    div = pq_cross_entropy - p_entropy
    return div

def kl_dist_dist(gt, out):
    div = kl.kl_divergence(gt, out)
    return div
