"""Carga de archivos de audio y utilidades compartidas por los análisis.

Todo el resto de `dsp/` asume el formato que devuelve `load_audio`: un array
float64 de forma (frames, canales), siempre 2D aunque el archivo sea mono. Eso
evita que cada módulo tenga que preguntarse si le llegó mono o estéreo.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

# Tope de duración para no colgar al anfitrión con un archivo enorme: un
# análisis de mezcla se hace sobre canciones, no sobre sets de dos horas.
MAX_DURATION_SECONDS = 15 * 60


class AudioError(ValueError):
    """Error de entrada legible por el usuario (archivo faltante, ilegible, etc.).

    Hereda de ValueError para que el SDK de MCP lo reporte como un error de la
    herramienta y no como una caída del servidor.
    """


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Lee un archivo de audio y devuelve (samples, samplerate).

    `samples` es float64 con forma (frames, canales) y rango nominal [-1, 1].
    """
    expanded = os.path.expanduser(path)

    if not os.path.exists(expanded):
        raise AudioError(f"Audio file not found: {path}")
    if not os.path.isfile(expanded):
        raise AudioError(f"Not a file: {path}")

    try:
        info = sf.info(expanded)
    except Exception as exc:  # libsndfile no reconoce el formato
        raise AudioError(f"Could not read '{path}' as audio: {exc}") from exc

    if info.duration > MAX_DURATION_SECONDS:
        raise AudioError(
            f"File is {info.duration / 60:.1f} min long; the limit is "
            f"{MAX_DURATION_SECONDS // 60} min. Analyze a shorter excerpt."
        )

    samples, samplerate = sf.read(expanded, dtype="float64", always_2d=True)

    if samples.shape[0] == 0:
        raise AudioError(f"File contains no audio samples: {path}")

    return samples, samplerate


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Mezcla a mono promediando los canales. Devuelve un array 1D."""
    return samples.mean(axis=1)


def amplitude_to_dbfs(amplitude: float | np.ndarray, floor_db: float = -200.0):
    """Convierte amplitud lineal a dBFS, con piso para no devolver -inf en el silencio."""
    amplitude = np.abs(amplitude)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(amplitude)
    return np.maximum(db, floor_db)


def dbfs_to_amplitude(dbfs: float) -> float:
    """Inversa de `amplitude_to_dbfs`."""
    return float(10.0 ** (dbfs / 20.0))


def describe_channels(channel_count: int) -> str:
    """Nombre legible del layout, para los mensajes de salida."""
    return {1: "mono", 2: "stereo"}.get(channel_count, f"{channel_count}-channel")
