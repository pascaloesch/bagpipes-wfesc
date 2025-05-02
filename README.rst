**Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation**

Bagpipes is a state of the art code for generating realistic model galaxy spectra and fitting these to spectroscopic and photometric observations. For further information please see the Bagpipes documentation at `bagpipes.readthedocs.io <http://bagpipes.readthedocs.io>`_.

This repository is a fork of the original bagpipes written by Adam Carnall, which you can find here <https://github.com/ACCarnall/bagpipes>.

The version here implements a picket fence model, but otherwise works exactly in the same way as bagpipes. Sampling for Lyman continuum escape fraction is possible by simply adding 
    nebular["fesc"] =  (0.001,1.0)
    nebular["fesc_prior"] = "log_10"
to your fit instructions.

If you make use of this version of bagpipes, please cite 'Giovinazzo et al. (2025, in prep)', as well as `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_ in any publications. You may also consider citing `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_, particularly if you are fitting spectroscopy.

Please note development of the code has been ongoing since these works were published, so certain parts of the code are no longer as described. Please inquire if in doubt.

