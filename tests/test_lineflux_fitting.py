"""
Tests for the emission line flux fitting and picket-fence fesc additions
on top of upstream Bagpipes v1.3.6:

1. fitted_model._lnlike_line_fluxes (upstream likelihood).
2. galaxy._check_line_fluxes validation of the load_line_fluxes output.
3. posterior.get_advanced_quantities stores chisq_emline and line_fluxes.
4. model_galaxy picket-fence fesc: lines scale as (1 - fesc), the full
   spectrum is linear in fesc (no double counting of escaping LyC), and
   out-of-range fesc values are clipped with a warning.

Tests 1-3 use mock objects, test 4 requires the model grids.
"""

import numpy as np
import sys
import os
import warnings

# Ensure bagpipes is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

HA = "H  1  6562.81A"
HB = "H  1  4861.33A"


# ──────────────────────────────────────────────────────────────────────
# 1. Likelihood
# ──────────────────────────────────────────────────────────────────────

def test_lnlike_line_fluxes():
    from bagpipes.fitting.fitted_model import fitted_model

    obs_fluxes = np.array([[1.0e-16, 1.0e-17],
                           [3.0e-17, 5.0e-18]])

    fm = object.__new__(fitted_model)
    fm.galaxy = SimpleNamespace(line_labels=[HA, HB], line_fluxes=obs_fluxes)
    fm.model_galaxy = SimpleNamespace(line_fluxes={HA: 1.1e-16, HB: 2.8e-17})
    fm.K_lines = -0.5*np.sum(np.log(2*np.pi*obs_fluxes[:, 1]**2))
    fm.inv_sigma_sq_lines = 1./obs_fluxes[:, 1]**2

    result = fm._lnlike_line_fluxes()

    expected_chisq = ((1.0e-16 - 1.1e-16)**2/(1.0e-17)**2
                      + (3.0e-17 - 2.8e-17)**2/(5.0e-18)**2)
    np.testing.assert_allclose(fm.chisq_lines, expected_chisq, rtol=1e-10)
    np.testing.assert_allclose(result, fm.K_lines - 0.5*expected_chisq)


# ──────────────────────────────────────────────────────────────────────
# 2. Validation of load_line_fluxes output
# ──────────────────────────────────────────────────────────────────────

def _make_galaxy(labels, fluxes):
    from bagpipes.input.galaxy import galaxy

    gal = object.__new__(galaxy)
    gal.line_labels = labels
    gal.line_fluxes = fluxes
    return gal


def _assert_raises(gal, msg):
    try:
        gal._check_line_fluxes()
    except ValueError as e:
        assert msg in str(e), str(e)
        return
    assert False, "Should have raised ValueError"


def test_check_line_fluxes_good():
    gal = _make_galaxy((HA, HB), [[5e-18, 3e-19], [2e-18, 3e-19]])
    gal._check_line_fluxes()

    assert gal.line_labels == [HA, HB]
    assert isinstance(gal.line_fluxes, np.ndarray)


def test_check_line_fluxes_wrong_shape():
    _assert_raises(_make_galaxy([HA, HB], np.ones((3, 2))), "shape")
    _assert_raises(_make_galaxy([HA], np.ones(2)), "shape")


def test_check_line_fluxes_unknown_label():
    # Upstream (non-extended) label format, not in cloudy_lines_ext.txt
    _assert_raises(_make_galaxy(["H  1  6562.80A"], np.ones((1, 2))),
                   "not in config.line_names")


def test_check_line_fluxes_bad_errors():
    _assert_raises(_make_galaxy([HA], np.array([[1e-18, 0.]])), "> 0")


# ──────────────────────────────────────────────────────────────────────
# 3. Posterior
# ──────────────────────────────────────────────────────────────────────

def test_posterior_chisq_emline():
    from unittest.mock import patch
    from bagpipes import config
    from bagpipes.fitting.posterior import posterior

    n_samples = 5
    model_line_fluxes = dict(zip(config.line_names,
                                 np.arange(len(config.line_names)) + 1.))

    post = object.__new__(posterior)
    post.n_samples = n_samples
    post.indices = np.arange(n_samples)
    post.samples2d = np.random.rand(20, 2)
    post.samples = {}

    post.galaxy = SimpleNamespace(line_labels=[HA, HB],
                                  photometry_exists=False,
                                  spectrum_exists=False, filt_list=None,
                                  spec_wavs=None, index_list=None,
                                  out_units="ergscma")

    mock_model = SimpleNamespace(spectrum_full=np.zeros(100),
                                 line_fluxes=model_line_fluxes)

    mock_fm = SimpleNamespace(model_components={"nebular": {}},
                              model_galaxy=mock_model, chisq_lines=0.)

    calls = [0]

    def fake_lnlike(param):
        mock_fm.chisq_lines = 1.5 + 0.1*calls[0]
        calls[0] += 1

    mock_fm._update_model_components = lambda param: None
    mock_fm.lnlike = fake_lnlike

    post.fitted_model = mock_fm
    post.model_galaxy = mock_model

    with patch("bagpipes.fitting.posterior.model_galaxy",
               lambda *args, **kwargs: mock_model):
        post.get_advanced_quantities()

    np.testing.assert_allclose(post.samples["chisq_emline"],
                               1.5 + 0.1*np.arange(n_samples))

    for name in [HA, HB]:
        np.testing.assert_allclose(post.samples["line_fluxes"][name],
                                   model_line_fluxes[name])


# ──────────────────────────────────────────────────────────────────────
# 4. Picket-fence fesc in model_galaxy
# ──────────────────────────────────────────────────────────────────────

def _model(fesc, dust=True):
    from bagpipes import model_galaxy

    comps = {"redshift": 0.,
             "t_bc": 0.01,
             "constant": {"age_max": 0.1, "age_min": 0.,
                          "massformed": 9., "metallicity": 0.2},
             "nebular": {"logU": -2.5, "fesc": fesc}}

    if dust:
        comps["dust"] = {"type": "Calzetti", "Av": 0.5, "eta": 2.}

    return model_galaxy(comps)


def test_fesc_line_scaling():
    m0, m3 = _model(0.), _model(0.3)

    for name in [HA, HB]:
        np.testing.assert_allclose(m3.line_fluxes[name],
                                   0.7*m0.line_fluxes[name], rtol=1e-10)


def test_fesc_spectrum_linear():
    """ The picket-fence spectrum is (1 - fesc)*covered + fesc*escaping,
    so it must be exactly linear in fesc, including the LyC region. """

    for dust in [True, False]:
        s0 = _model(0., dust=dust).spectrum_full
        s1 = _model(1., dust=dust).spectrum_full
        s3 = _model(0.3, dust=dust).spectrum_full

        np.testing.assert_allclose(s3, 0.7*s0 + 0.3*s1, rtol=1e-8,
                                   atol=1e-12*np.max(s1))


def test_fesc_clipping_warns():
    s1 = _model(1.).spectrum_full

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s_high = _model(1.5).spectrum_full

    assert any("clipping" in str(x.message) for x in w)
    np.testing.assert_allclose(s_high, s1)


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASSED: " + name)

    print("\nAll tests passed!")
