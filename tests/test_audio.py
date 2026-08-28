"""Pruebas de la carga de archivos (la única parte que toca el disco)."""

import numpy as np
import pytest
import soundfile as sf

from mixassist_mcp.dsp.audio import AudioError, amplitude_to_dbfs, load_audio, to_mono

from conftest import SAMPLERATE, sine


def test_missing_file_raises_a_readable_error():
    with pytest.raises(AudioError, match="not found"):
        load_audio("/no/existe/mezcla.wav")


def test_non_audio_file_is_rejected(tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("esto no es audio")

    with pytest.raises(AudioError):
        load_audio(str(fake))


def test_mono_file_is_loaded_as_two_dimensional(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, sine(440)[:, 0], SAMPLERATE)

    samples, samplerate = load_audio(str(path))

    assert samplerate == SAMPLERATE
    assert samples.ndim == 2
    assert samples.shape[1] == 1


def test_to_mono_averages_channels():
    stereo = np.column_stack([np.ones(10), np.zeros(10)])
    assert np.allclose(to_mono(stereo), 0.5)


def test_dbfs_conversion_matches_known_values():
    assert amplitude_to_dbfs(1.0) == pytest.approx(0.0)
    assert amplitude_to_dbfs(0.5) == pytest.approx(-6.0206, abs=1e-4)
    assert amplitude_to_dbfs(0.0) == -200.0  # piso, no -inf
