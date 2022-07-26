import torch

from loss import calcUncertaintyLoss
from model import UncertaintyModel
from utils import ModelVariant

# Sample Input sequence, Shape same as in RECOLA, AVEC'16 for the following constanst,
batch_size = 25
audio_samplerate = 16 #in kHZ
label_samplerate = 40 #in ms
feature_dim = audio_samplerate * label_samplerate
segments_size = 300 # Segment size in a batch sample. If 300, then each segment is 12 secs 300*40ms
nout = 2 # For arousal and valence
nannotators = 6
# Input: Raw audio, shape - [batch_size, num_segments, feature_dim]
#        num_segments - Number of labeled segments 
#                       e.g. In RECOLA labeled at 40ms windows for 5min (300s) audio, num_segments = 300s/40ms
#        feature_dim  - Number of audio frames in labeled segments 
#                       e.g. In RECOLA labeled at 40ms windows, so feature_dim = 40ms*audio_samplerate

# Select Model variant, by adjusting loss fucntions.
model_variant = ModelVariant.tstud_label_uncertainty

# Unit test Uncertainty Model = Convolution Paralinguistic feature extractor block
#                               + LSTM Temporal Feature extractor block
#                               + Uncertainty Model block
model = UncertaintyModel(nout=2, 
                        ninp_lstm=320, nhidden_lstm=256, nlstm=2,
                        dropout=0.5, 
                        uncertainty_samples=30, 
                        bbb_nsegments=50
                        )

print("@@@@@@@@@@@@@@@@@@@ Model SUMMARY @@@@@@@@@@@@@@@@@@@")
print(model)
print("#parameters = ", sum(p.numel() for p in model.parameters()))
inp_seq = torch.randn(batch_size, segments_size, feature_dim)
y_mean = torch.randn(batch_size, segments_size, nout)
y_all = torch.randn(batch_size, segments_size, nout, nannotators)
print("#"*80)
print("Input Shape - ", inp_seq.shape)
print("Mean Label Shape - ", y_mean.shape)
print("All Label Shape - ", y_all.shape)
print("#"*80)

outputs, outputs_mean, outputs_std, log_post, log_prior, outs_meanw = model.sample_uncertainty_predictions(inp_seq)
print("#"*80)
print("All Outputs Shape - ", outputs.shape)
print("Mean Output Shape - ", outputs_mean.shape)
print("Std Output Shape - ", outputs_std.shape)
print("#"*80)

loss, arousal_loss, valence_loss, \
    arousal_ccc, valence_ccc, \
    arousal_kl_gauss, valence_kl_gauss, \
    arousal_std_ccc, valence_std_ccc, \
    arousal_kl_stud, valence_kl_stud, \
    arousal_meanw_ccc, valence_meanw_ccc  = calcUncertaintyLoss(outputs_mean, outputs, y_mean, y_all, log_post, log_prior, outs_meanw, model_variant)