**Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation**

Bagpipes is a state of the art code for generating realistic model galaxy spectra and fitting these to spectroscopic and photometric observations. For further information please see the Bagpipes documentation at `bagpipes.readthedocs.io <http://bagpipes.readthedocs.io>`_.

This repository is a fork of the original bagpipes that allows you to fit for Lyman continuum escape fractions. The original version is written by Adam Carnall, and can be found here: https://github.com/ACCarnall/bagpipes.

The version here implements a picket fence model, but otherwise works exactly in the same way as bagpipes. Sampling for Lyman continuum escape fraction is possible by simply adding the following to your fit instructions:

    nebular["fesc"] =  (0.001,1.0) 

    nebular["fesc_prior"] = "log_10"

This version is based on upstream Bagpipes v1.3.6. It uses the BPASS v2.2.1 stellar grids and custom Cloudy nebular grids (hden=3.0, no grains, extended emission line list, see ``bagpipes/models/grids/cloudy_lines_ext.txt``).

Observed emission line fluxes can be fitted together with photometry using the upstream ``load_line_fluxes`` interface. The loader takes the galaxy ID and returns a list of Cloudy line labels and an array of shape (n_lines, 2) containing integrated fluxes and uncertainties in erg/s/cm^2:

.. code::

    def load_line_fluxes(ID):
        labels = ["H  1  6562.81A", "H  1  4861.33A"]
        fluxes = np.array([[5.1e-18, 3.1e-19],
                           [1.8e-18, 3.0e-19]])
        return labels, fluxes

    galaxy = pipes.galaxy(ID, load_phot, spectrum_exists=False,
                          filt_list=filt_list,
                          load_line_fluxes=load_line_fluxes)

The posterior then contains ``samples["chisq_emline"]`` as well as ``samples["line_fluxes"]`` (a dict keyed by line label), and ``fit.plot_spectrum_posterior()`` shows an extra panel with the observed and fitted line fluxes.


If you make use of this version of bagpipes, please cite Giovinazzo et al. (2025, in prep), as well as `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_ in any publications. You may also consider citing `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_, particularly if you are fitting spectroscopy.

Please note development of the code has been ongoing since these works were published, so certain parts of the code are no longer as described. Please inquire if in doubt.

