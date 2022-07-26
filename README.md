# End-To-End Label Uncertainty Modeling for Speech Emotion Recognition (SER) Using Bayesian Neural Networks 

This repository contains code for the papers

Navin Raj Prabhu, Guillaume Carbajal, Nale Lehmann-Willenbrock, Timo Gerkmann, **"End-To-End Label Uncertainty Modeling for Speech-based Arousal Recognition Using Bayesian Neural Networks"**, Interspeech, Incheon, Korea, Sep. 2022. [[arxiv]](https://arxiv.org/abs/2110.03299)

and

Navin Raj Prabhu, Nale Lehmann-Willenbrock, Timo Gerkmann, **"Label Uncertainty Modeling and Prediction for Speech Emotion Recognition using t-Distributions"**, Affective Computing and Intelligent Interaction (ACII), Nara, Japan, Oct. 2022. [[arxiv]](https://arxiv.org/abs/2207.12135)

## Model Variants
Three variants of label uncertainty models, for SER, introduced by the above papers, is available in this repository.
### Model Uncertainty and Label Uncertainty models
![alt text](https://github.com/sp-uhh/label-uncertainty-ser/blob/main/images/SpeechEmotionBNN.png?raw=true)
### *t*-distribution Label Uncertainty model
![alt text](https://github.com/sp-uhh/label-uncertainty-ser/blob/main/images/t-distBNN.png?raw=true)

Available variants,
1. Model Uncertainty (MU): 
```python 
ModelVariant.model_uncertainty 
```
3. Label Uncertainty (MU+LU): ```python ModelVariant.label_uncertainty ```
4. *t*-distribution Label Uncertainty (t-LU): ```python ModelVariant.tstud_label_uncertainty ```

## Usage
The usage of these uncertainty models is demonstrated and available in the file [unit_test.py](https://github.com/sp-uhh/label-uncertainty-ser/blob/main/unit_test.py). 


<!-- **STAY TUNED FOR THE CODE!** -->
