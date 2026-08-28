"""Detección de clipping digital.

Contar samples que tocan el techo no basta: un pico aislado en 0 dBFS es
inofensivo, mientras que una tira de samples consecutivos pegados al techo es
una forma de onda con la cresta cortada, que es lo que realmente se escucha
como distorsión. Por eso el análisis agrupa los samples en exceso en *runs*
consecutivos y reporta los runs, no solo el conteo total.
"""

from __future__ import annotations

import numpy as np

from .audio import amplitude_to_dbfs, dbfs_to_amplitude, describe_channels

# Por defecto se considera "en el techo" cualquier sample a -0.1 dBFS o más
# alto. No se usa exactamente 0.0 dBFS porque la conversión a float y el
# remuestreo dejan los picos cortados apenas por debajo del máximo.
DEFAULT_THRESHOLD_DBFS = -0.1

# Un run de al menos 3 samples consecutivos en el techo se toma como clipping
# real; menos que eso es un pico aislado (criterio típico de los detectores de
# los DAW).
DEFAULT_MIN_RUN = 3

# Cuántos runs se devuelven en el detalle, ordenados del más largo al más corto.
MAX_REPORTED_EVENTS = 10


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Encuentra tiras de True consecutivos en un array booleano 1D.

    Devuelve una lista de (índice_inicio, longitud). Se hace con diferencias
    sobre el array con padding en vez de un bucle en Python, porque una canción
    a 48 kHz son millones de samples por canal.
    """
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def detect_clipping(
    samples: np.ndarray,
    samplerate: int,
    threshold_dbfs: float = DEFAULT_THRESHOLD_DBFS,
    min_consecutive_samples: int = DEFAULT_MIN_RUN,
) -> dict:
    """Analiza `samples` (frames, canales) en busca de clipping.

    Devuelve un diccionario plano, listo para serializar como resultado MCP.
    """
    if threshold_dbfs > 0:
        raise ValueError("threshold_dbfs must be 0 or negative (dBFS scale).")
    if min_consecutive_samples < 1:
        raise ValueError("min_consecutive_samples must be at least 1.")

    frame_count, channel_count = samples.shape
    duration = frame_count / samplerate
    threshold_amp = dbfs_to_amplitude(threshold_dbfs)

    peak_amp = float(np.abs(samples).max())
    peak_dbfs = float(amplitude_to_dbfs(peak_amp))

    total_over = 0
    events: list[dict] = []
    per_channel: list[dict] = []

    for channel in range(channel_count):
        channel_samples = np.abs(samples[:, channel])
        over_mask = channel_samples >= threshold_amp
        over_count = int(over_mask.sum())
        total_over += over_count

        runs = _find_runs(over_mask)
        clipped_runs = [(start, length) for start, length in runs
                        if length >= min_consecutive_samples]

        per_channel.append({
            "channel": channel,
            "peak_dbfs": round(float(amplitude_to_dbfs(channel_samples.max())), 3),
            "samples_at_or_over_threshold": over_count,
            "clipped_runs": len(clipped_runs),
            "longest_run_samples": max((l for _, l in clipped_runs), default=0),
        })

        for start, length in clipped_runs:
            events.append({
                "channel": channel,
                "start_seconds": round(start / samplerate, 4),
                "duration_ms": round(1000.0 * length / samplerate, 3),
                "consecutive_samples": length,
            })

    # Los runs más largos son los que más se escuchan: se reportan primero.
    events.sort(key=lambda event: event["consecutive_samples"], reverse=True)
    total_events = len(events)

    if total_events:
        severity = "clipping"
    elif total_over:
        severity = "peaks_at_ceiling"
    else:
        severity = "clean"

    return {
        "file_info": {
            "duration_seconds": round(duration, 3),
            "samplerate": samplerate,
            "channels": channel_count,
            "channel_layout": describe_channels(channel_count),
        },
        "threshold_dbfs": threshold_dbfs,
        "min_consecutive_samples": min_consecutive_samples,
        "peak_dbfs": round(peak_dbfs, 3),
        "samples_at_or_over_threshold": total_over,
        "percent_of_samples_at_or_over_threshold": round(
            100.0 * total_over / (frame_count * channel_count), 6
        ),
        "clipping_events": total_events,
        "worst_events": events[:MAX_REPORTED_EVENTS],
        "per_channel": per_channel,
        "severity": severity,
        "verdict": _verdict(severity, total_events, total_over, peak_dbfs),
    }


def _verdict(severity: str, event_count: int, over_count: int, peak_dbfs: float) -> str:
    """Redacta la conclusión en lenguaje natural que el LLM le repite al usuario."""
    if severity == "clean":
        headroom = -peak_dbfs
        return (
            f"No clipping detected. The loudest peak sits at {peak_dbfs:.2f} dBFS, "
            f"leaving {headroom:.2f} dB of headroom."
        )
    if severity == "peaks_at_ceiling":
        return (
            f"No clipping, but {over_count} isolated sample(s) touch the ceiling "
            f"(peak {peak_dbfs:.2f} dBFS). Single samples at full scale are usually "
            "harmless, though they leave no headroom for lossy encoding."
        )
    return (
        f"Clipping detected: {event_count} run(s) of consecutive samples pinned at "
        f"the ceiling (peak {peak_dbfs:.2f} dBFS). The waveform is being flattened, "
        "which is audible as distortion. Lower the output gain before the limiter "
        "and re-render."
    )
