from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_galaxy import plot_galaxy
from .. import config


def plot_spectrum_posterior(fit, show=False, save=True):
    """ Plot the observational data and posterior from a fit object. """

    fit.posterior.get_advanced_quantities()

    update_rcParams()

    # First plot the observational data (including lineflux panel if present)
    fig, ax, y_scale = plot_galaxy(fit.galaxy, show=False, return_y_scale=True)

    # Determine which axis corresponds to which panel.
    ax_idx = 0

    if fit.galaxy.spectrum_exists:
        add_spectrum_posterior(fit, ax[ax_idx], zorder=6, y_scale=y_scale[ax_idx])
        ax_idx += 1

    if fit.galaxy.photometry_exists:
        add_photometry_posterior(fit, ax[ax_idx], zorder=2,
                                y_scale=y_scale[ax_idx])
        ax_idx += 1

    # Overlay posterior on the lineflux panel created by plot_galaxy.
    if fit.galaxy.lineflux_list is not None:
        add_lineflux_posterior(fit, ax[ax_idx], zorder=2, y_scale=y_scale[ax_idx])

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_fit.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def add_photometry_posterior(fit, ax, zorder=4, y_scale=None, color1=None,
                             color2=None, skip_no_obs=False,
                             background_spectrum=True, label=None):

    if color1 == None:
        color1 = "navajowhite"

    if color2 == None:
        color2 = "darkorange"

    mask = (fit.galaxy.photometry[:, 1] > 0.)
    upper_lims = fit.galaxy.photometry[:, 1] + fit.galaxy.photometry[:, 2]
    ymax = 1.05*np.max(upper_lims[mask])

    if not y_scale:
        y_scale = float(int(np.log10(ymax))-1)

    # Calculate posterior median redshift.
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    # Plot the posterior photometry and full spectrum.
    log_wavs = np.log10(fit.posterior.model_galaxy.wavelengths*(1.+redshift))
    log_eff_wavs = np.log10(fit.galaxy.filter_set.eff_wavs)

    if background_spectrum:
        spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                                  (16, 84), axis=0).T*10**-y_scale

        spec_post = spec_post.astype(float)  # fixes weird isfinite error

        ax.plot(log_wavs, spec_post[:, 0], color=color1,
                zorder=zorder-1, label=label)

        ax.plot(log_wavs, spec_post[:, 1], color=color1,
                zorder=zorder-1)

        ax.fill_between(log_wavs, spec_post[:, 0], spec_post[:, 1],
                        zorder=zorder-1, color=color1, linewidth=0)

    phot_post = np.percentile(fit.posterior.samples["photometry"],
                              (16, 84), axis=0).T

    for j in range(fit.galaxy.photometry.shape[0]):

        if skip_no_obs and fit.galaxy.photometry[j, 1] == 0.:
            continue

        phot_band = fit.posterior.samples["photometry"][:, j]
        mask = (phot_band > phot_post[j, 0]) & (phot_band < phot_post[j, 1])
        phot_1sig = phot_band[mask]*10**-y_scale
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]

        if phot_1sig.min() < ymax*10**-y_scale:
            ax.scatter(wav_array, phot_1sig, color=color2,
                       zorder=zorder, alpha=0.05, s=100, rasterized=True)

def add_spectrum_posterior(fit, ax, zorder=4, y_scale=None):

    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])

    if not y_scale:
        y_scale = float(int(np.log10(ymax))-1)

    wavs = fit.galaxy.spectrum[:, 0]
    spec_post = np.copy(fit.posterior.samples["spectrum"])

    if "calib" in list(fit.posterior.samples):
        spec_post /= fit.posterior.samples["calib"]

    if "noise" in list(fit.posterior.samples):
        spec_post += fit.posterior.samples["noise"]

    post = np.percentile(spec_post, (16, 50, 84), axis=0).T*10**-y_scale

    ax.plot(wavs, post[:, 1], color="sandybrown", zorder=zorder, lw=1.5)
    ax.fill_between(wavs, post[:, 0], post[:, 2], color="sandybrown",
                    zorder=zorder, alpha=0.75, linewidth=0)


def add_lineflux_posterior(fit, ax, zorder=4, y_scale=None,
                            color=None, label=None):
    """ Add observed and posterior emission line fluxes to axes.

    Plots observed line fluxes as data points with error bars and
    posterior model predictions as scatter clouds, analogous to
    add_photometry_posterior. """

    if color is None:
        color = "darkorange"

    n_lines = len(fit.galaxy.lineflux_list)

    # Look up rest-frame wavelengths for each fitted emission line.
    line_wavs = np.zeros(n_lines)
    for i, name in enumerate(fit.galaxy.lineflux_list):
        idx = np.where(config.line_names == name)[0][0]
        line_wavs[i] = config.line_wavs[idx]


    #log rest-frame wavelengths
    log_rf_wavs = np.log10(line_wavs)

    # Observed line fluxes and errors.
    obs_fluxes = fit.galaxy.linefluxes[:, 0]
    obs_errors = fit.galaxy.linefluxes[:, 1]

    # Determine y-scale.
    mask = (obs_fluxes > 0.)
    upper_lims = obs_fluxes + obs_errors
    if np.any(mask):
        ymax = 1.05*np.max(upper_lims[mask])
    else:
        ymax = 1.05*np.max(upper_lims)

    if y_scale is None:
        y_scale = float(int(np.log10(ymax)) - 1)


    # Plot posterior line flux samples (1-sigma scatter cloud).
    line_post = np.percentile(fit.posterior.samples["line_fluxes"],
                              (16, 84), axis=0).T 

    for j in range(n_lines):
        line_samples = fit.posterior.samples["line_fluxes"][:, j]
        post_mask = ((line_samples > line_post[j, 0])
                     & (line_samples < line_post[j, 1]))
        line_1sig = line_samples[post_mask]*10**-y_scale
        wav_array = np.zeros(line_1sig.shape[0]) + log_rf_wavs[j]

        if line_1sig.shape[0] > 0:
            ax.scatter(wav_array, line_1sig, color=color,
                       zorder=zorder, alpha=0.05, s=100, rasterized=True)
