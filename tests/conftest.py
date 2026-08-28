"""Señales sintéticas compartidas por los tests.

Se generan en memoria en vez de versionar archivos de audio: un seno tiene
propiedades exactas y conocidas (pico, frecuencia), así que sirve como oráculo
para verificar los algoritmos contra números calculados a mano.
"""

import numpy as np
import pytest

SAMPLERATE = 48000


@pytest.fixture
def samplerate() -> int:
    return SAMPLERATE


def sine(frequency: float, seconds: float = 1.0, amplitude: float = 0.5,
         samplerate: int = SAMPLERATE, channels: int = 1) -> np.ndarray:
    """Seno de amplitud exacta, con forma (frames, canales)."""
    t = np.arange(int(seconds * samplerate)) / samplerate
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return np.tile(wave[:, None], (1, channels))
