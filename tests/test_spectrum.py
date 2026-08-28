"""Pruebas del análisis espectral.

El oráculo aquí es la frecuencia del tono: un seno puro tiene que caer en la
banda que contiene esa frecuencia y en ninguna otra.
"""

import numpy as np
import pytest

from mixassist_mcp.dsp.spectrum import BANDS, analyze_spectrum

from conftest import SAMPLERATE, sine


@pytest.mark.parametrize("frequency,expected_band", [
    (40, "sub_bass"),
    (120, "bass"),
    (350, "low_mid"),
    (1000, "mid"),
    (3000, "high_mid"),
    (5000, "presence"),
    (10000, "air"),
])
def test_pure_tone_lands_in_its_band(frequency, expected_band):
    result = analyze_spectrum(sine(frequency, seconds=1.0), SAMPLERATE)

    assert result["dominant_band"] == expected_band
    share = next(b["energy_share_percent"] for b in result["bands"]
                 if b["band"] == expected_band)
    assert share > 95.0


def test_spectral_centroid_tracks_the_tone():
    low = analyze_spectrum(sine(100), SAMPLERATE)["spectral_centroid_hz"]
    high = analyze_spectrum(sine(8000), SAMPLERATE)["spectral_centroid_hz"]

    assert low == pytest.approx(100, rel=0.15)
    assert high == pytest.approx(8000, rel=0.15)
    assert low < high


def test_band_shares_do_not_exceed_one_hundred_percent():
    total = sum(b["energy_share_percent"]
                for b in analyze_spectrum(sine(1000), SAMPLERATE)["bands"])
    assert total <= 100.0001


def test_silence_is_rejected_with_a_useful_message():
    silence = np.zeros((SAMPLERATE, 1))

    with pytest.raises(ValueError, match="silent"):
        analyze_spectrum(silence, SAMPLERATE)


def test_bands_above_nyquist_are_omitted():
    """A 8 kHz de samplerate no hay banda de aire que reportar."""
    result = analyze_spectrum(sine(1000, samplerate=8000), 8000)

    reported = {b["band"] for b in result["bands"]}
    assert "air" not in reported
    assert "mid" in reported


@pytest.mark.parametrize("window_size", [100, 3000, 0])
def test_invalid_window_size_is_rejected(window_size):
    with pytest.raises(ValueError):
        analyze_spectrum(sine(1000), SAMPLERATE, window_size=window_size)


def test_band_table_is_contiguous_and_ordered():
    """Las bandas no deben dejar huecos ni solaparse."""
    for (_, _, high, _), (_, next_low, _, _) in zip(BANDS, BANDS[1:]):
        assert high == next_low
