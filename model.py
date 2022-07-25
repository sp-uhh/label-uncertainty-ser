import torch.nn as nn
import torch

from layers import ParalingExtractor, TemporalExtractor, BayesMLP
import utils

##### Number of Parameter = 1,643,110
class UncertaintyModel(nn.Module):
    def __init__(self, nout=2, ninp_lstm=320, nhidden_lstm=256, nlstm=2, dropout=0.5, uncertainty_samples=30, bbb_nsegments=300):

        super().__init__()

        self.uncertaintySamples = uncertainty_samples

        post_mu_init = utils.get_posterior_mu_init_range()
        post_rho_init = utils.get_posterior_rho_init_range()

        self.paralinguisticExtractor = ParalingExtractor(dropout=dropout)
        self.temporalExtractor = TemporalExtractor(ninp=ninp_lstm, nhidden=nhidden_lstm, nlstm=nlstm, dropout=dropout)
        self.uncertaintyLayer = BayesMLP(ninp=nhidden_lstm, nout=nout, bbb_nsegments=bbb_nsegments,
                                         post_mu_init=post_mu_init, post_rho_init=post_rho_init)

    def sample_uncertainty_predictions(self, x, y):
        # Input:
        #     x - input samples
        #     y - target samples
        # Output:
        #     log_post - Mean Posterior of BayesMLP weights (sum across all MLP layers) across #uncertaintySamples
        #     log_prior - Mean Posterior of BayesMLP weights (sum across all MLP layers) across #uncertaintySamples

        # we calculate variables for the negative elbo loss, which will be one of the elements in our loss function
        # initialize uncertainty tensors and place in device
        outputs = torch.zeros(self.uncertaintySamples, y.shape[0]*y.shape[1], self.nout)
        log_priors = torch.zeros(self.uncertaintySamples)
        log_posts = torch.zeros(self.uncertaintySamples)

        # Feature extract 'x' - Reuse code Conv1D and LSTM where no weight uncertianty exists-hence no sampling required
        feat_x = self(x, mode="feat")

        # print("Uncertainty Granularity -> ", self.bbb_nsegments)
        # make multiple predictions, using the uncertainty layers MLP alone,
        # and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(self.uncertaintySamples):
            # Input to Uncertainty layers also has control on granularity of uncertainty
            outputs[i] = self.uncertaintyLayer(feat_x, self.training, True).reshape(y.shape[0]*y.shape[1], self.nout)  # make predictions
            log_priors[i] = self.uncertaintyLayer.log_prior()  # get log prior
            log_posts[i] = self.uncertaintyLayer.log_post()  # get log variational posterior

        # Predicitons from mu of gaussain wgts, biases
        # Sampling and Training variables set to --> FALSE
        outs_meanw = self.uncertaintyLayer(feat_x, False, False).reshape(y.shape[0]*y.shape[1], self.nout)

        # Below, y.shape[0]*y.shape[1] = number of samples tested
        num_samples = outputs.shape[1]
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean() / num_samples
        log_post = log_posts.mean() / num_samples

        # Calculate mean and std of outputs (across uncertaintySamples)
        outputs_mean = outputs.mean(0)
        outputs_var  = outputs.var(0)

        return outputs, outputs_mean, outputs_var, log_post, log_prior, outs_meanw

    def forward(self, x, mode):
        paraling_feat = self.paralinguisticExtractor(x) if self.paralinguisticExtractor is not None else x
        temporal_feat = self.temporalExtractor(paraling_feat)

        if mode== "pred":
            uncert_out = self.uncertaintyLayer(temporal_feat, True)
            return uncert_out
        else:
            return temporal_feat