**Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation**

Bagpipes is a state of the art code for generating realistic model galaxy spectra and fitting these to spectroscopic and photometric observations. For further information please see the Bagpipes documentation at [bagpipes.readthedocs.io](http://bagpipes.readthedocs.io).

This repository is a fork of the original bagpipes that allows you to fit for Lyman continuum escape fractions and observed emission line fluxes. The original version is written by Adam Carnall, and can be found here: https://github.com/ACCarnall/bagpipes.

This version (`1.3.6+wfesc`) is based on upstream Bagpipes v1.3.6. It uses the BPASS v2.2.1 stellar grids and custom Cloudy nebular grids (hden=3.0, no grains, extended emission line list, see `bagpipes/models/grids/cloudy_lines_ext.txt`).

## Installation

```bash
git clone https://github.com/pascaloesch/bagpipes-wfesc
cd bagpipes-wfesc
pip install -e .
```

The model grids (FITS files) are not included in the repository and must be placed in `bagpipes/models/grids/`:

- `bpass_2.2.1_bin_imf135_300_stellar_grids.fits`
- `bpass_2.2.1_bin_imf135_300_nebular_{cont,line}_grids_nodust_hden3_logUm4to0_extLines.fits`
- `dl07_grids_umin_only_no_norm.fits` and `dl07_grids_umin_umax_no_norm.fits`, which can be taken from the upstream PyPI package (`pip download bagpipes==1.3.6 --no-deps`, then unzip the wheel)
- `d_igm_grid_inoue14.fits` is computed automatically if missing

Check the installed version with `bagpipes.__version__`. The tests can be run with `python -m pytest tests/` (the fesc tests need the model grids).

## Lyman continuum escape fraction

The escape fraction is implemented as a picket fence model: a fraction `fesc` of the birth cloud light escapes without any nebular processing or dust attenuation, while all ionising photons from the remaining `1 - fesc` are reprocessed into nebular emission. Emission line fluxes therefore scale with `1 - fesc`. Sampling for the escape fraction is possible by simply adding the following to your fit instructions:

```python
nebular["fesc"] = (0.001, 1.0)
nebular["fesc_prior"] = "log_10"
```

Values of `fesc` outside the range 0 to 1 are clipped with a warning.

## Fitting emission line fluxes

Observed emission line fluxes can be fitted together with photometry using the `load_line_fluxes` interface. The loader takes the galaxy ID and returns a list of Cloudy line labels and an array of shape `(n_lines, 2)` containing integrated fluxes and uncertainties in erg/s/cm^2:

```python
def load_line_fluxes(ID):
    labels = ["H  1  6562.81A", "H  1  4861.33A"]
    fluxes = np.array([[5.1e-18, 3.1e-19],
                       [1.8e-18, 3.0e-19]])
    return labels, fluxes

galaxy = pipes.galaxy(ID, load_phot, phot_units="mujy",
                      spectrum_exists=False, filt_list=filt_list,
                      load_line_fluxes=load_line_fluxes)
```

The labels must match the names in `bagpipes/models/grids/cloudy_lines_ext.txt` exactly (including spaces), e.g. `"H  1  6562.81A"` for Halpha. Bagpipes checks that the array has the right shape, that all labels are known and that all uncertainties are > 0. If the loader returns no lines for an object, a warning is printed and only the photometry is fitted.

The likelihood adds a Gaussian chi-squared term in linear flux for the lines. Note that the IGM is not applied to the model line fluxes, so Lyman alpha should not be fitted.

### Blended lines

Blended lines, e.g. an [OIII]4960,5008 flux measured as a single line, can be fitted by joining the component labels with `"+"`:

```python
labels = ["O  3  5006.84A+O  3  4958.91A"]
```

The observed flux is then compared to the sum of the model fluxes of the components. The same works for any other blend, e.g. [OII]3726,3729 or [SII]6716,6731.

### Example loader for a line catalogue

The loader below reads a catalogue with `flux_<name>` and `err_<name>` columns (missing measurements flagged with `err = 99`). It removes any requested line that is unknown to bagpipes, not in the translation dictionary, not in the catalogue or not measured for the object, so the same list of lines can be used for all objects:

```python
import warnings

lineCat = Table.read(LineCatName).to_pandas()
ERROR_FLOOR = 0.05

# translate line labels from cloudy/bagpipes into catalogue column suffixes
# (labels joined with "+" are blends: catalogue flux = sum of the components)
lineDict = {"H  1  4.05113m": "BrA",
            "H  1  3.73951m": "PfG",
            "H  1  3.29607m": "PfD",
            "H  1  2.62513m": "BrB",
            "H  1  2.16551m": "BrG",
            "H  1  1.87510m": "PaA",
            "H  1  1.28180m": "PaB",
            "H  1  1.09380m": "PaG",
            "TOTL  1.08303m": "HeI-1083",  # HeI 10830 triplet, total in model
            "H  1  1.00493m": "PaD",
            "S  3  9530.62A+S  3  9068.62A": "SIII",
            "Blnd  7323.00A+Blnd  7332.00A": "OII-7325",
            "Ar 3  7135.79A": "ArIII-7138",
            "S  2  6716.44A+S  2  6730.82A": "SII",
            "N  2  6583.45A+N  2  6548.05A": "NII",
            "H  1  6562.81A": "Ha",
            "O  1  6300.30A+O  1  6363.78A": "OI-6302",
            "He 1  5875.64A": "HeI-5877",
            "O  3  5006.84A": "OIII-5007",
            "O  3  4958.91A": "OIII-4959",
            "H  1  4861.33A": "Hb",
            "Blnd  4363.00A": "OIII-4363",
            "H  1  4340.46A": "Hg",
            "H  1  4101.73A": "Hd",
            "H  1  3970.07A": "H7",
            "Ne 3  3868.76A": "NeIII-3867",
            "H  1  3889.05A": "H8",
            "H  1  3835.38A": "H9",
            "H  1  3797.89A": "H10",
            "Blnd  3726.00A+Blnd  3729.00A": "OII"}

# lines we would like to fit (only those found and measured are returned)
LINES_TO_FIT = list(lineDict)


def get_line_fluxes(ID, lineflux_list=LINES_TO_FIT):
    row = lineCat[lineCat.id == int(ID)]
    if len(row) == 0:
        raise ValueError(f"ID {ID} not found in {LineCatName}")
    row = row.iloc[0]

    labels, line_measurements = [], []

    for lineName in lineflux_list:

        # check the line (or all components of a blend) is known to bagpipes
        if not all(c in pipes.config.line_names for c in lineName.split("+")):
            warnings.warn(f"{lineName!r} not in bagpipes line list, removed")
            continue

        if lineName not in lineDict:
            warnings.warn(f"{lineName!r} not in lineDict, removed")
            continue

        fluxCol, errCol = f"flux_{lineDict[lineName]}", f"err_{lineDict[lineName]}"
        if fluxCol not in lineCat.columns or errCol not in lineCat.columns:
            warnings.warn(f"{fluxCol}/{errCol} not in {LineCatName}, removed")
            continue

        flux, err = float(row[fluxCol]), float(row[errCol])  # in erg/s/cm2

        # remove lines that are not measured for this object
        if (not np.isfinite(flux) or not np.isfinite(err) or flux == -99.
                or err <= 0 or err == 99.):
            continue

        # introduce an error floor
        if (flux > 0) and (err/flux < ERROR_FLOOR):
            err = ERROR_FLOOR * flux

        # make sure that negative fluxes are within 1sigma of 0
        elif (flux < 0) and (flux + err < 0):
            err = -1 * flux

        labels.append(lineName)
        line_measurements.append([flux, err])

    return labels, np.array(line_measurements, dtype=float).reshape(-1, 2)
```

Notes on the line list (FRESCO catalogue):

- [OIII]5007 and [OIII]4959 are fitted individually here. For catalogues with a single blended [OIII] flux use `"O  3  5006.84A+O  3  4958.91A"` instead, but never the blend together with its components, which would count the same flux twice.
- H8 (3890 Å) is unresolved from He I 3889 Å, and Hε (3971 Å) is close to [NeIII]3968 Å. If the catalogue fluxes include these, use the blends `"H  1  3889.05A+He 1  3888.63A"` and `"H  1  3970.07A+Ne 3  3967.47A"`.
- Doublets measured as a single flux (e.g. [SIII], [NII], [SII], [OII]) use blend labels, with the line ratios set by the Cloudy models.
- Strong lines also contribute to the broadband photometry, so their flux enters the likelihood twice. This is usually fine as the errors are independent, but keep it in mind when interpreting chi-squared values.

### Outputs

After `fit.posterior.get_advanced_quantities()` the posterior contains:

- `samples["chisq_emline"]`: the chi-squared of the line fluxes for each sample.
- `samples["line_fluxes"]`: a dict keyed by line label with the model flux samples for all lines in the model grid, plus an entry for each fitted blended label.

`fit.plot_spectrum_posterior()` and `galaxy.plot()` show an extra panel with the observed and fitted line fluxes.

Because `samples["line_fluxes"]` is a dict, it needs to be skipped when printing percentiles of all posterior quantities:

```python
print("{:<25s} {:>8s} {:>8s} {:>8s}".format("posteriors", "16th", "50th", "84th"))
print(54*"-")
for key in list(fit.posterior.samples):
    if isinstance(fit.posterior.samples[key], dict):
        continue  # line_fluxes, printed below
    prct = np.percentile(fit.posterior.samples[key], (16, 50, 84))
    print("{:<25s} {:>8.3f} {:>8.3f} {:>8.3f}".format(key, *prct))

# fitted emission lines: model posterior vs. observed (erg/s/cm2)
if fit.galaxy.line_labels is not None:
    print("\n{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "line_fluxes", "16th", "50th", "84th", "obs", "obs_err"))
    print(78*"-")
    for name, (obs, err) in zip(fit.galaxy.line_labels, fit.galaxy.line_fluxes):
        prct = np.percentile(fit.posterior.samples["line_fluxes"][name], (16, 50, 84))
        print("{:<25s} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e}".format(
            name, *prct, obs, err))
```

## Tips

### Plot labels

Plot labels are written in TeX format. By default they are rendered with matplotlib's built-in mathtext, which needs no LaTeX installation. To render them with LaTeX instead, set `text.usetex` before importing bagpipes:

```python
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True

import bagpipes as pipes
```

### OpenMPI shmem warning on macOS

Bagpipes imports `mpi4py`, which with OpenMPI on macOS may print a harmless warning like `shmem: mmap: an error occurred while determining whether or not .../sm_segment... could be created`. It can be silenced by switching OpenMPI to its POSIX shared memory backend, either for all OpenMPI programs:

```bash
mkdir -p ~/.openmpi && echo "shmem = posix" >> ~/.openmpi/mca-params.conf
```

or in a script before importing bagpipes:

```python
import os
os.environ["OMPI_MCA_shmem"] = "posix"
```

## Citing the code

If you make use of this version of bagpipes, please cite Giovinazzo et al. (2025, in prep), as well as [Carnall et al. (2018)](https://arxiv.org/abs/1712.04452) in any publications. You may also consider citing [Carnall et al. (2019b)](https://arxiv.org/abs/1903.11082), particularly if you are fitting spectroscopy.

Please note development of the code has been ongoing since these works were published, so certain parts of the code are no longer as described. Please inquire if in doubt.
