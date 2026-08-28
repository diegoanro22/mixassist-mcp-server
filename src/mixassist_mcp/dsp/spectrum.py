"""Análisis espectral por bandas.

El balance tonal de una mezcla se juzga por cuánta energía cae en cada región
del espectro, no por el nivel global. Aquí se calcula una STFT (ventaneo de
Hann con solape, FFT real por trama) y se promedia la potencia en el tiempo
para obtener un espectro promedio de la canción; ese espectro se agrega en las
bandas con las que se habla en el estudio ("sub", "bajos", "medios", "aire").

La STFT se arma a mano sobre `numpy.fft.rfft` en vez de usar una función que ya
devuelva el espectrograma, porque el punto del proyecto es el algoritmo, no la
llamada a la librería.
"""

from __future__ import annotations

import numpy as np

from .audio import describe_channels, to_mono

# Bandas de trabajo habituales en mezcla. Los límites no son un estándar
# formal: son las regiones con las que un productor razona (y con las que están
# rotulados los ecualizadores).
BANDS: tuple[tuple[str, float, float, str], ...] = (
    ("sub_bass",   20.0,    60.0, "Weight you feel more than hear; kick and 808 fundamentals."),
    ("bass",       60.0,   250.0, "Bass guitar, kick body, the low end of the groove."),
    ("low_mid",   250.0,   500.0, "Warmth. Too much here is the classic 'muddy' mix."),
    ("mid",       500.0,  2000.0, "Where most instruments and vocal fundamentals live."),
    ("high_mid", 2000.0,  4000.0, "Attack and intelligibility; the ear is most sensitive here."),
    ("presence", 4000.0,  6000.0, "Definition and edge; harshness when overdone."),
    ("air",      6000.0, 20000.0, "Openness and sheen from cymbals and breath."),
)

# 4096 muestras a 44.1 kHz son ~93 ms: resolución de ~10 Hz, suficiente para
# separar las bandas graves sin promediar de más en el tiempo.
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_HOP_RATIO = 0.5

# Tramas por debajo de este nivel se descartan: los silencios entre secciones
# no dicen nada del balance tonal y sesgarían el promedio.
SILENCE_FLOOR_RMS = 1e-5


def _stft_power(signal: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    """Potencia promedio por bin de frecuencia, promediada sobre las tramas.

    Devuelve un array 1D de longitud `window_size // 2 + 1`.
    """
    if signal.size < window_size:
        # Archivo más corto que una ventana: se rellena con ceros para poder
        # analizarlo igual (un loop de un compás es un caso legítimo).
        signal = np.pad(signal, (0, window_size - signal.size))

    window = np.hanning(window_size)
    frame_starts = range(0, signal.size - window_size + 1, hop_size)

    accumulated = np.zeros(window_size // 2 + 1)
    counted = 0

    for start in frame_starts:
        frame = signal[start:start + window_size]
        if np.sqrt(np.mean(frame ** 2)) < SILENCE_FLOOR_RMS:
            continue
        spectrum = np.fft.rfft(frame * window)
        accumulated += np.abs(spectrum) ** 2
        counted += 1

    if counted == 0:
        raise ValueError(
            "The file appears to be silent: every analysis frame is below the "
            "noise floor, so there is no spectral balance to report."
        )

    return accumulated / counted


def analyze_spectrum(
    samples: np.ndarray,
    samplerate: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    """Calcula el reparto de energía por bandas de `samples` (frames, canales)."""
    if window_size < 256 or window_size & (window_size - 1):
        raise ValueError("window_size must be a power of two and at least 256.")

    frame_count, channel_count = samples.shape
    mono = to_mono(samples)
    hop_size = max(1, int(window_size * DEFAULT_HOP_RATIO))

    power = _stft_power(mono, window_size, hop_size)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / samplerate)
    total_power = float(power.sum())

    nyquist = samplerate / 2.0
    bands = []
    for name, low, high, meaning in BANDS:
        if low >= nyquist:
            # Un archivo a 22 kHz no tiene banda de aire que reportar.
            continue
        effective_high = min(high, nyquist)
        selected = (freqs >= low) & (freqs < effective_high)
        band_power = float(power[selected].sum())
        share = band_power / total_power if total_power else 0.0
        bands.append({
            "band": name,
            "range_hz": [low, effective_high],
            "energy_share_percent": round(100.0 * share, 3),
            "relative_db": round(float(10.0 * np.log10(share)), 2) if share > 0 else -120.0,
            "meaning": meaning,
        })

    # Centroide espectral: el "centro de gravedad" del espectro en Hz. Sube
    # cuando la mezcla es brillante y baja cuando es oscura.
    centroid = float((freqs * power).sum() / total_power) if total_power else 0.0

    dominant = max(bands, key=lambda b: b["energy_share_percent"])

    return {
        "file_info": {
            "duration_seconds": round(frame_count / samplerate, 3),
            "samplerate": samplerate,
            "channels": channel_count,
            "channel_layout": describe_channels(channel_count),
        },
        "analysis": {
            "window_size": window_size,
            "hop_size": hop_size,
            "frequency_resolution_hz": round(samplerate / window_size, 3),
            "window": "hann",
        },
        "spectral_centroid_hz": round(centroid, 1),
        "dominant_band": dominant["band"],
        "bands": bands,
        "observations": _observations(bands, centroid),
    }


def _observations(bands: list[dict], centroid: float) -> list[str]:
    """Reglas de oído convertidas en texto.

    Son heurísticas de mezcla, no umbrales normativos: se redactan como
    observaciones para que el LLM las presente como tales y no como veredictos.
    """
    share = {b["band"]: b["energy_share_percent"] for b in bands}
    notes: list[str] = []

    low_end = share.get("sub_bass", 0.0) + share.get("bass", 0.0)
    if low_end > 75.0:
        notes.append(
            f"The low end holds {low_end:.1f}% of the energy, which is a lot even "
            "for bass-forward genres — check whether the kick and bass are stacking up."
        )
    elif low_end < 20.0:
        notes.append(
            f"Only {low_end:.1f}% of the energy is below 250 Hz; the mix will sound "
            "thin on systems with real low-end extension."
        )

    if share.get("low_mid", 0.0) > 25.0:
        notes.append(
            f"Low-mids sit at {share['low_mid']:.1f}% — this is the region that reads "
            "as muddiness when it builds up."
        )

    high_end = share.get("presence", 0.0) + share.get("air", 0.0)
    if high_end < 1.0:
        notes.append(
            f"Above 4 kHz there is only {high_end:.2f}% of the energy; the mix may "
            "sound dull or over-compressed in the top end."
        )

    if centroid < 500:
        notes.append(f"The spectral centroid is low ({centroid:.0f} Hz): a dark overall balance.")
    elif centroid > 4000:
        notes.append(f"The spectral centroid is high ({centroid:.0f} Hz): a bright, possibly harsh balance.")

    if not notes:
        notes.append("No band stands out as obviously over- or under-represented.")

    return notes
