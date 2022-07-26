import torch.nn as nn
import torch

from layers import ParalingExtractor, TemporalExtractor, BayesMLP
import utils

##### Number of Parameter = 1,643,110
class UncertaintyModel(nn.Module):
    def __init__(self, nout=2, ninp_lstm=320, nhidden_lstm=256, nlstm=2, dropout=0.5, uncertainty_samples=30, bbb_nsegments=50):
        """ Generates the Uncertainty Model object for label uncertainty aware Speech Emotion Recognition.
        The model introduced in the following papers,

        Navin Raj Prabhu, Guillaume Carbajal, Nale Lehmann-Willenbrock, Timo Gerkmann, 
        "End-To-End Label Uncertainty Modeling for Speech-based Arousal Recognition Using Bayesian Neural Networks",
        Interspeech, Incheon, Korea, Sep. 2022. https://arxiv.org/abs/2110.03299 

        and

        Navin Raj Prabhu, Nale Lehmann-Willenbrock, Timo Gerkmann, 
        "Label Uncertainty Modeling and Prediction for Speech Emotion Recognition using t-Distributions", 
        Affective Computing and Intelligent Interaction (ACII), Nara, Japan, Oct. 2022 https://arxiv.org/abs/2207.12135

        Args:
            nout (int, optional): The emotion output dimension. 
                                            Defaults to 2, for arousal and valence prediction.
            ninp_lstm (int, optional): Input dimension to the LSTM-based TemporalExtractor layer. 
                                            Defaults to 320.
            nhidden_lstm (int, optional): Output dimesion of the TemporalExtractor, number of temporal features. 
                                            Defaults to 256.
            nlstm (int, optional): Number of stacked LSTM layers. 
                                            Defaults to 2.
            dropout (float, optional): Dropout probability on the feature extraction layers. 
                                            Defaults to 0.5.
            uncertainty_samples (int, optional): Number of foward passes in the uncertainty layer, for stochastic outputs. 
                                            Defaults to 30, to acheive output distribtuion converging to a Gaussian.
            bbb_nsegments (int, optional): Granularity of dynamic uncertainty and stochastic weights. 
                                            Defaults to 50, leading to new weights sampled every 2 secs (40ms*50=2s).
        """
        super().__init__()

        self.nout = nout
        self.uncertaintySamples = uncertainty_samples

        post_mu_init = utils.get_posterior_mu_init_range()
        post_rho_init = utils.get_posterior_rho_init_range()

        self.paralinguisticExtractor = ParalingExtractor(dropout=dropout)
        self.temporalExtractor = TemporalExtractor(ninp=ninp_lstm, nhidden=nhidden_lstm, nlstm=nlstm, dropout=dropout)
        self.uncertaintyLayer = BayesMLP(ninp=nhidden_lstm, nout=nout, bbb_nsegments=bbb_nsegments,
                                         post_mu_init=post_mu_init, post_rho_init=post_rho_init)

    def sample_uncertainty_predictions(self, x):
        """ The uncertainty model's forward pass functions.
        While training/testing this model, use this function as the forward pass. 
        This function in turn uses the vanilla forward() + does additional stochastic output sampling. 

        Args:
            x (_type_): For Batch Size of 25 and Segment Size of 300 (300 segment size = 12s, 40ms*300 = 12 secs)-> torch.Size([25, 300, 320])

        Returns:
            outputs: All stochastic outputs. Dimension 0 has all uncertainty samples. example shape - torch.Size([uncertaintySamples, 25, 300, 2]) 
            outputs_mean: Mean (m_t) of stochastic ouputs. example shape - torch.Size([25, 300, 2]) 
            outputs_std: Standard deviation (s_t) of stochastic ouputs. example shape - torch.Size([25, 300, 2])  
            log_post: Mean Posterior of BayesMLP weights (sum across all MLP layers) across #uncertaintySamples.
            log_prior: Mean Posterior of BayesMLP weights (sum across all MLP layers) across #uncertaintySamples. 
            outs_meanw: Mean Emotions (m_t) predicitons using mu of gaussain wgts and biases, reducing the randomization effect of sampling.
        """

        # we calculate variables for the negative elbo loss, which will be one of the elements in our loss function
        # initialize uncertainty tensors and place in device
        outputs = torch.zeros(self.uncertaintySamples, x.shape[0]*x.shape[1], self.nout)
        log_priors = torch.zeros(self.uncertaintySamples)
        log_posts = torch.zeros(self.uncertaintySamples)

        # Feature extract 'x' - Reuse code Conv1D and LSTM where no weight uncertianty exists-hence no sampling required
        feat_x = self(x, mode="feat")

        # make multiple predictions, using the uncertainty layers MLP alone,
        # and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(self.uncertaintySamples):
            # Input to Uncertainty layers also has control on granularity of uncertainty
            outputs[i] = self.uncertaintyLayer(feat_x, self.training, True).reshape(x.shape[0]*x.shape[1], self.nout)  # make predictions
            log_priors[i] = self.uncertaintyLayer.log_prior()  # get log prior
            log_posts[i] = self.uncertaintyLayer.log_post()  # get log variational posterior

        # Predicitons from mu of gaussain wgts, biases
        # Sampling and Training variables set to --> FALSE
        outs_meanw = self.uncertaintyLayer(feat_x, False, False).reshape(x.shape[0]*x.shape[1], self.nout)

        # Below, y.shape[0]*y.shape[1] = number of samples tested
        num_samples = outputs.shape[1]
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean() / num_samples
        log_post = log_posts.mean() / num_samples

        # Calculate mean and std of outputs (across uncertaintySamples)
        outputs_mean = outputs.mean(0)
        outputs_std  = outputs.std(0)

        return outputs, outputs_mean, outputs_std, log_post, log_prior, outs_meanw

    def forward(self, x, mode):
        paraling_feat = self.paralinguisticExtractor(x) if self.paralinguisticExtractor is not None else x
        temporal_feat = self.temporalExtractor(paraling_feat)

        if mode== "pred":
            # Forward pass including the uncertainty layer.
            # Return the stochastic outputs of the model.
            uncert_out = self.uncertaintyLayer(temporal_feat, True)
            return uncert_out
        else:
            # Forward pass with only the E2E backbone.
            # Return the features extracted.
            return temporal_feat