# COMP541 - Term Project
This is a part of COMP541: Deep Learning course's project. Where I will be trying to replicate the DNN model DeepPPI presented by Du et al. (DOI: 10.1021/acs.jcim.7b00028)

## Paper summary
On one hand, there are a lot of experimental methods for the detection of protein protein interactions (PPI) such as x-ray crystallography, Nuclear Magnetic Resonance NMR, and Tandem Affinity Purification. However, these methods have many downsides, such as being costly, time consuming, labor-intensive, and highly affected by equipments resolution and environmental disturbances during the experiment. Thus, the need for a powerful computational methods to predict PPI is rising. This paper proposes using a Deep Neural Network model to predict PPI using protein descriptors extracted from empirical data stored in multiple databases such as Database of Interacting Proteins (DIP).
The novelty of using DDN, based on the paper is that it can automatically extract high-level meaningful and abstract features of proteins from noisy data instead of hand-picking and crafting discriminant features which in addition to requiring a solid domain knowledge,  might also be prone to errors due to the noise that might be present in the features. 
Based on the paper, applying the proposed DNN model achieved the following results:

  Accuracy      Precision       Recall        Specificity       Matthews Correlation Coefficient
  92.5%         94.38%          90.56%        94.49%            85.08%

DeepPPI Predictor performance was evaluated using eight different PPI datasets taken from literature, they are described in details in the paper.

