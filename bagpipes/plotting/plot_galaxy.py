from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_spectrum import add_spectrum
from .. import config


def plot_galaxy(galaxy, show=True, return_y_scale=False, y_scale_spec=None):
    """ Make a quick plot of the data loaded into a galaxy object. """

    update_rcParams()

    naxes = 1
    if (galaxy.photometry_exists and galaxy.spectrum_exists):
        naxes = 2

    has_linefluxes = (hasattr(galaxy, 'lineflux_list')
                      and galaxy.lineflux_list is not None)

    if has_linefluxes:
        naxes += 1

    y_scale = []

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    ax_idx = 0

    # Add observed spectroscopy to plot
    if galaxy.spectrum_exists:
        spec_ax = plt.subplot(gs[ax_idx, 0])
        ax_idx += 1

        y_scale_spec = add_spectrum(galaxy.spectrum, spec_ax,
                                    y_scale=y_scale_spec)

        if galaxy.photometry_exists:
            add_observed_photometry_linear(galaxy, spec_ax,
                                           y_scale=y_scale_spec)

        axes = [spec_ax]
        y_scale = [y_scale_spec]

    # Add observed photometry to plot
    if galaxy.photometry_exists and galaxy.spectrum_exists:
        phot_ax = plt.subplot(gs[ax_idx, 0])
        ax_idx += 1
        y_scale_phot = float(add_observed_photometry(galaxy, phot_ax))
        y_scale.append(y_scale_phot)
        axes.append(phot_ax)

    elif galaxy.photometry_exists:
        phot_ax = plt.subplot(gs[ax_idx, 0])
        ax_idx += 1
        y_scale_phot = float(add_observed_photometry(galaxy, phot_ax))
        y_scale = [y_scale_phot]
        axes = [phot_ax]

    # Add observed emission line fluxes to plot
    if has_linefluxes:
        line_ax = plt.subplot(gs[ax_idx, 0])
        y_scale_line = float(add_observed_linefluxes(galaxy, line_ax))
        y_scale.append(y_scale_line)
        axes.append(line_ax)

    if show:
        plt.show()
        plt.close(fig)

    if return_y_scale:
        return fig, axes, y_scale

    return fig, axes


def add_observed_photometry(galaxy, ax, x_ticks=None, zorder=4, ptsize=40,
                            y_scale=None, lw=1., skip_no_obs=False,
                            label=None, color="blue", marker="o"):
    """ Adds photometric data to the passed axes. """

    photometry = np.copy(galaxy.photometry)

    if skip_no_obs:
        mask = (photometry[:, 1] != 0.)
        photometry = photometry[mask, :]

    # Sort out axis limits
    ax.set_xlim((np.log10(galaxy.filter_set.eff_wavs.min()) - 0.025),
                (np.log10(galaxy.filter_set.eff_wavs.max()) + 0.025))

    mask = (photometry[:, 1] > 0.)
    ymax = 1.1*np.nanmax((photometry[:, 1]+photometry[:, 2])[mask])

    if y_scale is None:
        y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Plot the data
    ax.errorbar(np.log10(photometry[:, 0]),
                photometry[:, 1]*10**-y_scale,
                yerr=photometry[:, 2]*10**-y_scale, lw=lw,
                linestyle=" ", capsize=3, capthick=1, zorder=zorder-1,
                color="black")

    ax.scatter(np.log10(photometry[:, 0]),
               photometry[:, 1]*10**-y_scale, color=color, s=ptsize,
               zorder=zorder, linewidth=lw, facecolor=color,
               edgecolor="black", label=label, marker=marker)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    auto_axis_label(ax, y_scale, log_x=True)

    return y_scale


def add_observed_photometry_linear(galaxy, ax, zorder=4, y_scale=None,
                                   skip_no_obs=False, ptsize=40, lw=1.,
                                   marker="o", label=None, color="blue"):
    """ Adds photometric data to the passed axes without doing any
    manipulation of the axes or labels. """

    photometry = np.copy(galaxy.photometry)

    if skip_no_obs:
        mask = (photometry[:, 1] != 0.)
        photometry = photometry[mask, :]

    mask = (photometry[:, 1] > 0.)
    ymax = 1.05*np.nanmax((photometry[:, 1]+photometry[:, 2])[mask])

    if not y_scale:
        y_scale = int(np.log10(ymax))-1

    # Plot the data
    ax.errorbar(photometry[:, 0], photometry[:, 1]*10**-y_scale,
                yerr=photometry[:, 2]*10**-y_scale, lw=lw,
                linestyle=" ", capsize=3, capthick=lw, zorder=zorder-1,
                color="black")

    ax.scatter(photometry[:, 0], photometry[:, 1]*10**-y_scale, color=color,
               s=ptsize, zorder=zorder, linewidth=lw, facecolor=color,
               edgecolor="black", marker=marker, label=label)

    auto_axis_label(ax, y_scale, log_x=False)

    return ax


def add_observed_linefluxes(galaxy, ax, zorder=4, y_scale=None, ptsize=40,
                            lw=1., color="blue", marker="o", label=None):
    """ Adds observed emission line fluxes from lineflux_list to axes. """

    n_lines = len(galaxy.lineflux_list)
    linefluxes = np.copy(galaxy.linefluxes)

    # Look up rest-frame wavelengths for each emission line.
    line_wavs = np.zeros(n_lines)
    for i, name in enumerate(galaxy.lineflux_list):
        idx = np.where(config.line_names == name)[0][0]
        line_wavs[i] = config.line_wavs[idx]

    log_wavs = np.log10(line_wavs)

    # Determine y-scale from the data.
    obs_fluxes = linefluxes[:, 0]
    obs_errors = linefluxes[:, 1]

    mask = (obs_fluxes > 0.)
    upper_lims = obs_fluxes + 10 * obs_errors

    if np.any(mask):
        ymax = 1.1*np.max(upper_lims[mask])
    else:
        ymax = 1.1*np.max(np.abs(upper_lims))

    if y_scale is None:
        y_scale = int(np.log10(ymax)) - 1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Set x-axis limits.
    wav_range = log_wavs.max() - log_wavs.min()
    if wav_range < 0.1:
        wav_range = 0.2
    ax.set_xlim(log_wavs.min() - 0.1*wav_range - 0.025,
                log_wavs.max() + 0.1*wav_range + 0.025)

    # Plot the data.
    ax.errorbar(log_wavs, obs_fluxes*10**-y_scale,
                yerr=obs_errors*10**-y_scale, lw=lw,
                linestyle=" ", capsize=3, capthick=1, zorder=zorder-1,
                color="black")

    ax.scatter(log_wavs, obs_fluxes*10**-y_scale, color=color, s=ptsize,
               zorder=zorder, linewidth=lw, facecolor=color,
               edgecolor="black", label=label, marker=marker)

    # Annotate each point with the line name.
    for j in range(n_lines):
        name = galaxy.lineflux_list[j].strip()
        ax.annotate(name, (log_wavs[j], obs_fluxes[j]*10**-y_scale),
                    textcoords="offset points", xytext=(5, 10),
                    fontsize=7, rotation=45, ha='left', zorder=zorder+1)

    # Axis formatting.
    auto_x_ticks(ax)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    if tex_on:
        ax.set_ylabel("$\\mathrm{F_{line}}\\ \\mathrm{/\\ 10^{"
                      + str(int(y_scale))
                      + "}\\ erg\\ s^{-1}\\ cm^{-2}}$")

        ax.set_xlabel("$\\mathrm{log_{10}}\\big(\\lambda_\\mathrm{rest}"
                      + " / \\mathrm{\\AA}\\big)$")

    else:
        ax.set_ylabel("F_line / 10^" + str(int(y_scale))
                      + " erg s^-1 cm^-2")
        ax.set_xlabel("log_10(lambda_rest / A)")

    return y_scale
