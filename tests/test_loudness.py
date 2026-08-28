"""Pruebas de la medición de loudness.

El oráculo principal es el propio estándar: BS.1770-4 publica los coeficientes
de los filtros para 48 kHz, así que se comparan contra la tabla; y define la
escala de forma que un seno de 1 kHz a -23 dBFS RMS mida exactamente
-23.0 LUFS, así que esa señal sirve de calibración.
"""

import numpy as np
import pytest
from scipy.signal import freqz

from mixassist_mcp.dsp.loudness import (
    HIGHPASS_A_48K,
    HIGHPASS_B_48K,
    SHELF_A_48K,
    SHELF_B_48K,
    _highpass_coefficients,
    _shelf_coefficients,
    analyze_loudness,
    true_peak_dbtp,
)

from conftest import SAMPLERATE, sine


# --- K-weighting -----------------------------------------------------------

@pytest.mark.parametrize("designer,expected_b,expected_a", [
    (_shelf_coefficients, SHELF_B_48K, SHELF_A_48K),
    (_highpass_coefficients, HIGHPASS_B_48K, HIGHPASS_A_48K),
])
def test_coefficients_match_the_standard_at_48k(designer, expected_b, expected_a):
    """A 48 kHz los filtros deben ser exactamente los tabulados en BS.1770-4."""
    b, a = designer(48000)

    assert b == pytest.approx(expected_b, abs=1e-15)
    assert a == pytest.approx(expected_a, abs=1e-15)


@pytest.mark.parametrize("designer", [_shelf_coefficients, _highpass_coefficients])
@pytest.mark.parametrize("samplerate", [44100, 88200, 96000])
def test_redesigned_filters_keep_the_same_response(designer, samplerate):
    """Re-discretizar a otro rate debe conservar la respuesta en frecuencia."""
    probe_hz = np.array([50.0, 200.0, 1000.0, 5000.0, 15000.0])

    b48, a48 = designer(48000)
    b_new, a_new = designer(samplerate)

    _, h48 = freqz(b48, a48, worN=2 * np.pi * probe_hz / 48000)
    _, h_new = freqz(b_new, a_new, worN=2 * np.pi * probe_hz / samplerate)

    difference_db = 20 * np.log10(np.abs(h48)) - 20 * np.log10(np.abs(h_new))
    assert np.abs(difference_db).max() < 0.05


# --- Calibración de la escala ----------------------------------------------

def test_reference_tone_reads_minus_23_lufs():
    """Calibración del estándar: 1 kHz a -23 dBFS RMS son -23.0 LUFS."""
    amplitude = 10 ** (-23.0 / 20.0) * np.sqrt(2)  # RMS de -23 dBFS
    result = analyze_loudness(sine(1000, seconds=5.0, amplitude=amplitude), SAMPLERATE)

    assert result["integrated_lufs"] == pytest.approx(-23.0, abs=0.1)


def test_doubling_amplitude_adds_six_lu():
    quiet = analyze_loudness(sine(1000, seconds=5.0, amplitude=0.1), SAMPLERATE)
    loud = analyze_loudness(sine(1000, seconds=5.0, amplitude=0.2), SAMPLERATE)

    assert loud["integrated_lufs"] - quiet["integrated_lufs"] == pytest.approx(6.02, abs=0.05)


def test_measurement_is_independent_of_samplerate():
    at_48k = analyze_loudness(sine(1000, seconds=5.0), 48000)["integrated_lufs"]
    at_44k = analyze_loudness(sine(1000, seconds=5.0, samplerate=44100), 44100)["integrated_lufs"]

    assert at_48k == pytest.approx(at_44k, abs=0.1)


def test_duplicating_a_channel_adds_three_lu():
    """Sumar un canal idéntico duplica la potencia: +3 LU, no +6."""
    mono = analyze_loudness(sine(1000, seconds=5.0), SAMPLERATE)["integrated_lufs"]
    stereo = analyze_loudness(sine(1000, seconds=5.0, channels=2), SAMPLERATE)["integrated_lufs"]

    assert stereo - mono == pytest.approx(3.01, abs=0.05)


# --- Compuertas -------------------------------------------------------------

def test_relative_gate_ignores_a_quiet_intro():
    """Un pasaje muy suave no debe arrastrar hacia abajo la medición.

    Es justo lo que hace la compuerta relativa, y lo que se rompe si solo se
    implementa la absoluta: la intro está a -40 dB, muy por encima de la
    compuerta absoluta de -70 LUFS, así que solo la relativa la descarta.

    La tolerancia no es cero por un efecto de borde inevitable: los bloques de
    400 ms que caen a caballo entre la intro y la parte fuerte contienen algo de
    cada una, pasan la compuerta y bajan un poco el promedio. Su peso decrece
    con la duración de la parte fuerte (verificado: 0.23 LU con 3 s, 0.02 LU con
    30 s), o sea que es dilución, no un error del gating.
    """
    loud = sine(1000, seconds=10.0, amplitude=0.2)
    with_intro = np.vstack([loud[:len(loud) // 2] * 0.01, loud])

    only_loud = analyze_loudness(loud, SAMPLERATE)
    gated = analyze_loudness(with_intro, SAMPLERATE)

    assert gated["integrated_lufs"] == pytest.approx(only_loud["integrated_lufs"], abs=0.1)
    # Los bloques de la intro se descartaron: se usaron menos de los que hay.
    assert gated["gating"]["blocks_used"] < gated["gating"]["blocks_total"]


def test_without_the_relative_gate_the_reading_would_be_wrong():
    """Comprueba que la intro suave sí superaría la compuerta absoluta.

    Si no fuera así, el test anterior pasaría por la razón equivocada: estaría
    verificando la compuerta absoluta y no la relativa.
    """
    quiet_intro = sine(1000, seconds=5.0, amplitude=0.2) * 0.01

    intro_alone = analyze_loudness(quiet_intro, SAMPLERATE)["integrated_lufs"]

    assert intro_alone > -70.0


def test_silence_reports_no_integrated_loudness():
    result = analyze_loudness(np.zeros((SAMPLERATE * 2, 1)), SAMPLERATE)

    assert result["integrated_lufs"] is None
    assert "silence" in result["verdict"].lower()


def test_audio_shorter_than_one_block_is_rejected():
    with pytest.raises(ValueError, match="shorter than"):
        analyze_loudness(sine(1000, seconds=0.2), SAMPLERATE)


# --- True peak --------------------------------------------------------------

def test_true_peak_is_at_least_the_sample_peak():
    samples = sine(997, seconds=1.0, amplitude=0.9)  # 997 Hz no divide el rate
    sample_peak_db = 20 * np.log10(np.abs(samples).max())

    assert true_peak_dbtp(samples, SAMPLERATE) >= sample_peak_db - 1e-9


def test_true_peak_catches_inter_sample_overshoot():
    """Una señal cuyo pico cae entre muestras supera el pico de sample."""
    samples = sine(SAMPLERATE / 4.0 - 1, seconds=1.0, amplitude=0.99)
    sample_peak_db = 20 * np.log10(np.abs(samples).max())

    assert true_peak_dbtp(samples, SAMPLERATE) > sample_peak_db


# --- Reporte ----------------------------------------------------------------

def test_platform_comparison_points_the_right_way():
    result = analyze_loudness(sine(1000, seconds=5.0, amplitude=0.5), SAMPLERATE)
    spotify = next(p for p in result["platform_targets"] if p["platform"] == "Spotify")

    assert spotify["target_lufs"] == -14.0
    if result["integrated_lufs"] > -13.0:
        assert "turn down" in spotify["action"]
