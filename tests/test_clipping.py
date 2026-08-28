"""Pruebas del detector de clipping.

Cada caso se contrasta contra un valor calculable a mano: una señal de
amplitud 0.5 tiene un pico de exactamente 20*log10(0.5) = -6.0206 dBFS, y un
seno recortado tiene dos regiones planas por ciclo.
"""

import numpy as np
import pytest

from mixassist_mcp.dsp.clipping import detect_clipping

from conftest import SAMPLERATE, sine


def test_clean_signal_reports_no_clipping():
    result = detect_clipping(sine(440, amplitude=0.5), SAMPLERATE)

    assert result["severity"] == "clean"
    assert result["clipping_events"] == 0
    assert result["samples_at_or_over_threshold"] == 0
    assert result["peak_dbfs"] == pytest.approx(-6.02, abs=0.01)


def test_hard_clipped_signal_is_detected():
    frequency, seconds = 440, 1.0
    clipped = np.clip(sine(frequency, seconds, amplitude=1.5), -1.0, 1.0)

    result = detect_clipping(clipped, SAMPLERATE)

    assert result["severity"] == "clipping"
    assert result["peak_dbfs"] == pytest.approx(0.0, abs=0.01)
    # Un seno recortado arriba y abajo produce dos mesetas por ciclo.
    assert result["clipping_events"] == 2 * frequency * seconds


def test_isolated_peak_is_not_called_clipping():
    """Un solo sample en el techo es un pico, no distorsión audible."""
    samples = sine(440, amplitude=0.5)
    samples[1000, 0] = 1.0

    result = detect_clipping(samples, SAMPLERATE)

    assert result["severity"] == "peaks_at_ceiling"
    assert result["samples_at_or_over_threshold"] == 1
    assert result["clipping_events"] == 0


def test_events_are_reported_worst_first():
    samples = sine(440, amplitude=1.2)
    samples = np.clip(samples, -1.0, 1.0)

    events = detect_clipping(samples, SAMPLERATE)["worst_events"]

    lengths = [e["consecutive_samples"] for e in events]
    assert lengths == sorted(lengths, reverse=True)
    assert all(0 <= e["start_seconds"] <= 1.0 for e in events)


def test_stereo_is_reported_per_channel():
    result = detect_clipping(sine(440, amplitude=0.5, channels=2), SAMPLERATE)

    assert result["file_info"]["channel_layout"] == "stereo"
    assert [c["channel"] for c in result["per_channel"]] == [0, 1]


@pytest.mark.parametrize("kwargs", [
    {"threshold_dbfs": 1.0},
    {"min_consecutive_samples": 0},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        detect_clipping(sine(440), SAMPLERATE, **kwargs)
