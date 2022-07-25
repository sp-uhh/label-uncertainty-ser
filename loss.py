from torch.distributions import Normal, studentT
import torch

from kl_divergence_loss import kl_dist_dist

def CCC(data1, data2):
    mean1 = torch.mean(data1)
    mean2 = torch.mean(data2)
    std1 = torch.std(data1)
    std2 = torch.std(data2)
    dm = mean1 - mean2

    ccc = (
            (2 * (torch.mean((data1 - mean1) * (data2 - mean2)))) /
            ((std1 * std1) + (std2 * std2) + (dm * dm))
            )
    return ccc

def calcKLdivBBBnSegments(out, gt, agg, annot_dist):
    num_annot = gt.shape[-1]
    out_mean, out_std = torch.mean(out, 0), torch.std(out, 0)
    gt_mean, gt_std   = torch.mean(gt, -1), torch.std(gt, -1)

    out_dist_arousal = Normal(out_mean[:, 0], out_std[:, 0])
    out_dist_valence = Normal(out_mean[:, 1], out_std[:, 1])
    if annot_dist == "gauss":
        gt_dist_arousal = Normal(gt_mean[:, 0], gt_std[:, 0])
        gt_dist_valence = Normal(gt_mean[:, 1], gt_std[:, 1])
    elif annot_dist == "tstud":
        gt_dist_arousal = studentT.StudentT(df=num_annot, loc=gt_mean[:, 0], scale=gt_std[:, 0]) 
        gt_dist_valence = studentT.StudentT(df=num_annot, loc=gt_mean[:, 1], scale=gt_std[:, 1])

    arousal_kl = kl_dist_dist(gt_dist_arousal, out_dist_arousal)
    valence_kl = kl_dist_dist(gt_dist_valence, out_dist_valence)

    return agg(arousal_kl), agg(valence_kl)

def calcUncertaintyLoss(out_mean, out_all, y_mean, y_all, log_post, log_prior, out_meanw, loss_function):

    # Reshape y_mean to [samples, arousal/valence]
    batch_size, num_segments, num_outputs = out_mean.shape
    out_mean = torch.reshape(out_mean, (batch_size*num_segments, -1))
    y_mean = torch.reshape(y_mean, (batch_size*num_segments, -1))

    # Lccc(m) calc
    arousal_ccc = 1 - CCC(out_mean[:, 0], y_mean[:, 0])
    valence_ccc = 1 - CCC(out_mean[:, 1], y_mean[:, 1])
    loss_ccc = ((arousal_ccc + valence_ccc) / 2)

    # Lccc(m) for mean predictions from mean weights
    arousal_meanw_ccc = 1 - CCC(out_meanw[:, 0], y_mean[:, 0])
    valence_meanw_ccc = 1 - CCC(out_meanw[:, 1], y_mean[:, 1])

    # Lccc(s) calc
    arousal_std_ccc = 1 - CCC(torch.std(out_all, 0)[:, 0], torch.std(y_all, -1)[:, 0])
    valence_std_ccc = 1 - CCC(torch.std(out_all, 0)[:, 1], torch.std(y_all, -1)[:, 1])

    # L_KL calc
    arousal_kl_gauss, valence_kl_gauss = calcKLdivBBBnSegments(out_all, y_all, torch.mean, "gauss")
    arousal_kl_stud, valence_kl_stud = calcKLdivBBBnSegments(out_all, y_all, torch.mean, "tstud")

    #######################  FINAL LOSS Function for TRAIN based on Model used ##############################
    # log_like = -ve (Neg ELBO inverse.prop., to log_like) -> Better log_like better accuracy.
    if loss_function == "model_uncertainty": #Case 2
        # Aim :
        # Minimize - Dist between log_post & log_prior (so minimize (log_post - log_prior))
        # Minimize - loss_cc (so in '+' loss_cc)
        # So, Minimize - loss_cc + (log_post - log_prior)
        arousal_loss = arousal_ccc + log_post - log_prior
        valence_loss = valence_ccc + log_post - log_prior

        loss = loss_ccc + log_post - log_prior
    elif loss_function == "label_uncertainty": #Case 5 
        arousal_loss = log_post - log_prior + arousal_ccc + arousal_kl_gauss 
        valence_loss = log_post - log_prior + valence_ccc + valence_kl_gauss 

        loss = (log_post - log_prior) + loss_ccc + ((arousal_kl_gauss+valence_kl_gauss)/2) 
    elif loss_function == "tstud_label_uncertainty":
        arousal_loss = log_post - log_prior + arousal_ccc + arousal_kl_stud
        valence_loss = log_post - log_prior + valence_ccc + valence_kl_stud
        loss = (log_post - log_prior) + loss_ccc + ((arousal_kl_stud+valence_kl_stud)/2)

    print()
    print("Current Batch CCC- " + ", AVG: ", loss_ccc.item(), ", Arousal: ", arousal_ccc.item(), ", Valence: ", valence_ccc.item())
    print("Current Batch MeanW CCC- " + ", AVG: ", ((arousal_meanw_ccc+valence_meanw_ccc)/2), ", Arousal: ", arousal_meanw_ccc, ", Valence: ", valence_meanw_ccc)
    print()
    print("Current Batch STD CCC- " + ", AVG: ", ((arousal_std_ccc+valence_std_ccc)/2).item(), ", Arousal: ", arousal_std_ccc.item(), ", Valence: ", valence_std_ccc.item())
    print("Current Batch KL- " + ", AVG: ", ((arousal_kl_gauss + valence_kl_gauss) / 2).item(), ", Arousal: ", arousal_kl_gauss.item(), ", Valence: ", valence_kl_gauss.item())
    print("Current Batch t-Student KL- " + ", AVG: ", ((arousal_kl_stud + valence_kl_stud) / 2).item(), ", Arousal: ", arousal_kl_stud.item(), ", Valence: ", valence_kl_stud.item())
    print("Current Batch LOSS- " + loss_function + ", AVG: ", loss.item(), ", Arousal: ", arousal_loss.item(), ", Valence: ", valence_loss.item())
    print()

    # RETURSN LOSS TO MINIMIZE e.g. 1-CCC
    return loss, arousal_loss, valence_loss, arousal_ccc, valence_ccc, arousal_kl_gauss, valence_kl_gauss, arousal_std_ccc, valence_std_ccc, arousal_kl_stud, valence_kl_stud, arousal_meanw_ccc, valence_meanw_ccc 
