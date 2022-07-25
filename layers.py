
import torch.nn as nn
import torch
import math

from distributions import ScaleMixtureGaussian, Gaussian
import torch.nn.functional as F
import constants as c
import utils

class ParalingExtractor(nn.Module):

    def __init__(self, dropout=0.5):
        super().__init__()
        # padding=[floor(kernal_sizes[0]/2) CAN ALSO ensures "same" padding as in tensorflow
        # "same" results in padding with zeros evenly to the left/right
        # of the input such that output has the same width dimension as the input.
        nfilters = [50, 125, 125]
        kernal_sizes = [8, 6, 6]
        pool_sizes = [10, 5, 5]

        self.conv_layer1 = nn.Conv1d(1, nfilters[0], kernal_sizes[0], padding="same")
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(pool_sizes[0])
        self.dropout1 = nn.Dropout(dropout)

        self.conv_layer2 = nn.Conv1d(nfilters[0], nfilters[1], kernal_sizes[1], padding="same")
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(pool_sizes[1])
        self.dropout2 = nn.Dropout(dropout)

        self.conv_layer3 = nn.Conv1d(nfilters[1], nfilters[2], kernal_sizes[2], padding="same")
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(pool_sizes[2])
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # Input: Raw audio, shape - [batch_size, num_segments, feature_dim]
        #        num_segments - Number of labeled segments e.g. In RECOLA labeled at 40ms windows,
        #                                                            so num_segments = TOTAL-RECOLA-DURATION/40ms
        #        feature_dim  - Number of audio frames in labeled segments e.g. In RECOLA labeled at 40ms windows,
        #                                                            so feature_dim = 40ms*audio_samplerate
        batch_size, num_segments, feature_dim = x.shape
        x = torch.reshape(x, (batch_size, num_segments*feature_dim, 1))
        x = x.permute(0, 2, 1)

        conv1_out = self.dropout1(self.maxpool1(self.relu1(self.conv_layer1(x))))
        conv2_out = self.dropout2(self.maxpool2(self.relu2(self.conv_layer2(conv1_out))))
        conv3_out = self.dropout3(self.maxpool3(self.relu3(self.conv_layer3(conv2_out))))

        # RESHAPE back to [batch_size, num_segments, feature_dim]
        paraling_out = conv3_out.permute(0, 2, 1)
        paraling_out = torch.reshape(paraling_out, (batch_size, num_segments, -1))

        return paraling_out

class TemporalExtractor(nn.Module):

    def __init__(self, ninp=320, nhidden=256, nlstm=2, dropout=0.5):
        super().__init__()
        self.stacked_lstm = nn.LSTM(ninp, nhidden, nlstm, batch_first=True, dropout=dropout)

    def forward(self, x):
        temporal_out, (hidden, cell) = self.stacked_lstm(x)
        return temporal_out

class BayesMLP(nn.Module):

    def __init__(self, ninp=256, nout=2, bbb_nsegments=300, post_mu_init=(-.1, .1), post_rho_init=(-3, -2)):
        # post_init - array of size 2, with lower limit @ 0 and upper @ 1
        super().__init__()

        self.ninp = ninp
        self.nout = nout
        self.bbb_nsegments = bbb_nsegments
        self.nneurons= [512, 256, nout]

        self.hidden1 = BayesianLinear(ninp, self.nneurons[0], post_mu_init, post_rho_init)
        self.hidden2 = BayesianLinear(self.nneurons[0], self.nneurons[1], post_mu_init, post_rho_init)
        self.hidden3 = BayesianLinear(self.nneurons[1], self.nneurons[2], post_mu_init, post_rho_init)
        self.out     = BayesianLinear(self.nneurons[2], nout, post_mu_init, post_rho_init)

        self.relu1 = nn.PReLU(num_parameters=self.nneurons[0])
        self.relu2 = nn.PReLU(num_parameters=self.nneurons[1])
        self.relu3 = nn.PReLU(num_parameters=self.nneurons[2])


    def log_prior(self):
        # calculate the log prior over all the layers
        log_prior_estimate= self.hidden1.log_prior + self.hidden2.log_prior + self.hidden3.log_prior + self.out.log_prior
        return log_prior_estimate

    def log_post(self):
        # calculate the log posterior over all the layers
        log_post_estimate = self.hidden1.log_post + self.hidden2.log_post + self.hidden3.log_post + self.out.log_post
        return log_post_estimate

    def forward(self, x, isTraining, isSampling=False, isCalcLogProba=False):
        batch_size, num_segments, feature_dim = x.shape
        x = torch.reshape(x, (batch_size * num_segments, feature_dim))

        # Batch Norming before Applying bayesian MLP
        # Following Loop:
        #    1. Forward passes each time-step sample at a time (In contrast to full sample)
        #    2. Achieves modeling uncertainty at each time-step
        #    3. At each time-step is expensive.
        #    4. Need to choose between -> each segment (300*40 ms, 12 secs), each sample (extreme), each time-step 40ms
        #    5. At what granularity do we need uncertainty. How long is uncertainty constant??
        # Control uncertainty granularity with different bbb_nsegments
        split_x = torch.split(x, self.bbb_nsegments, dim=0)
        out = torch.zeros(len(split_x), split_x[0].shape[0], self.nout)

        for i, seg in enumerate(split_x):
            temp_x = self.relu1(self.hidden1(seg, isTraining, isSampling, isCalcLogProba))
            temp_x = self.relu2(self.hidden2(temp_x, isTraining, isSampling, isCalcLogProba))
            temp_x = self.relu3(self.hidden3(temp_x, isTraining, isSampling, isCalcLogProba))
            
            temp_x = self.out(temp_x, isTraining, isSampling, isCalcLogProba)
        
            out[i] = temp_x
        return out

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, post_mu_init, post_rho_init):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(post_mu_init[0], post_mu_init[1]))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(post_rho_init[0], post_rho_init[1])) # -3, -2 (5_) || -4, -3 (2_)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(post_mu_init[0], post_mu_init[1]))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(post_rho_init[0], post_rho_init[1])) # -3, -2 # -3, -2 (5_) || -4, -3 (2_)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        
        # Prior distributions
        self.weight_prior = utils.get_prioir_init_dist(c.PRIOR_DIST)
        self.bias_prior = utils.get_prioir_init_dist(c.PRIOR_DIST)
        self.log_prior = 0
        self.log_post = 0

    def forward(self, input, isTraining, isSampling=False, isCalcLogProba=False):
        
        if isTraining or isSampling:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        if isTraining or isCalcLogProba:
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
            self.log_post = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_post = 0, 0

        return F.linear(input, weight, bias)

