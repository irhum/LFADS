# LFADS
This is an unofficial implementation of LFADS(Latent Factor Analysis via Dynamical Systems), used to infer the underlying dynamics that generated a neural spike train, from single-trial spiking data. You can read more about the method [here](https://arxiv.org/abs/1608.06315), and its use in neuroscience [here](https://www.nature.com/articles/s41592-018-0109-9). 

This implementation is fairly incomplete, and presently exists mostly for me to understand this method. If you need a fully featured implementation, please check these instead:
* Official implementation in TensorFlow: https://github.com/tensorflow/models/tree/master/research/lfads
* Implementation in JAX by the original author: https://github.com/google-research/computation-thru-dynamics

## Current state:
Currently, this is a barebones working prototype. Only some regularization techniques have been implemented, and some models are missing. It trains on all 8 TPU cores on a v2-8 (thanks, TRC!), but is limited by data infeed (which could be improved by a proper tf.data pipeline)
### What's here
* Synthetic data generation tools for the Chaotic RNN used in the supplement [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-018-0109-9/MediaObjects/41592_2018_109_MOESM1_ESM.pdf)
    * A working example showing the autonomous LFADS being trained on this data is in `experiments/autonomous.ipynb`
* The *autonomous* LFADS model (Fig 1 from the Nature paper), that does *not* infer any inputs i.e. the spikes are generated purely by the dynamics that are to be inferred.
    * Row-normalized weights for computing factors.
    * L2 norm regularization on the generator's weight matrix for internal state.
    * A custom initializer for the GRU's bias is included. That initializes the hidden state bias to -1 (the rest are 0 per usual) to stabilize learning. 
        * This achieves the same as in the TF implementation [here](https://github.com/tensorflow/models/blob/master/research/lfads/lfads.py#L147). The sign flip is because the TF implementation uses a update gate GRU, whereas Haiku uses a reset gate GRU. 
    * Uses dropout on the input, and on the encoded sequence data (before conversion to mean/stddev)
### What's in progress
* The *inferred-input* LFADS model has yet to be built
    * That includes its corresponding AR1 autoregressive KL prior
* The *multi-session* LFADS model, capable of "stitching" together data from different recording sessions
* Currently does *not* use temporal dropout on the generator (will be added soon)
* KL loss exists, but uses a variance of 1 instead of 0.1 as in the paper


