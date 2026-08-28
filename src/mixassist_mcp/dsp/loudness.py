"""Medición de loudness según ITU-R BS.1770-4 y EBU R128.

Esta es la pieza no trivial del servidor: el estándar no se resuelve con una
llamada a una librería, sino que define un procedimiento completo que aquí se
implementa paso a paso.

1. **K-weighting**: dos biquads en cascada que aproximan cómo el oído pondera
   las frecuencias. El primero es un shelf agudo que modela la cabeza como
   obstáculo acústico; el segundo, un pasa-altos que descarta los graves que no
   aportan a la sensación de volumen. El estándar publica los coeficientes solo
   para 48 kHz; aquí se derivan de los parámetros analógicos (fc, Q, ganancia)
   para cualquier samplerate, y los tests verifican que a 48 kHz reproducen los
   valores de la tabla del estándar.

2. **Bloques con solape**: la señal ponderada se corta en bloques de 400 ms con
   75% de solape (paso de 100 ms) y de cada bloque se saca la media cuadrática.

3. **Doble compuerta (gating)**: aquí es donde casi todas las implementaciones
   caseras se equivocan. Primero se descartan los bloques por debajo de
   -70 LUFS (compuerta absoluta: silencio). Luego se calcula el promedio de los
   que sobrevivieron y se descartan además los que estén 10 LU por debajo de
   ese promedio (compuerta relativa). El loudness integrado es el promedio de
   los que pasan ambas. Sin la compuerta relativa, una canción con pasajes
   suaves mide varios LU por debajo de lo que realmente se percibe.

Referencia: ITU-R BS.1770-4 (2015) y EBU Tech 3342 para el loudness range.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import bilinear_zpk, lfilter, resample_poly, tf2zpk, zpk2tf

from .audio import describe_channels

# --- K-weighting -----------------------------------------------------------
#
# BS.1770-4 publica los coeficientes de los dos biquads solo para 48 kHz
# (tablas 1 y 2). Para medir archivos a 44.1 kHz —o a cualquier otro rate— hay
# que reconstruir el filtro, y aquí está la parte delicada: la fórmula estándar
# de shelf del "audio EQ cookbook" NO reproduce esos coeficientes (se desvía un
# 3.6%, verificado numéricamente), porque el estándar parte de otro prototipo
# analógico.
#
# La solución correcta es no adivinar la fórmula: se toman los coeficientes
# tabulados, se recupera el filtro analógico que los originó aplicando la
# transformada bilineal al revés (s = 2*fs*(z-1)/(z+1) sobre polos y ceros), y
# ese prototipo se vuelve a discretizar al samplerate que toque. A 48 kHz el
# resultado es idéntico a la tabla por construcción, y a cualquier otro rate es
# el mismo filtro analógico, que es lo que el estándar realmente define.

REFERENCE_SAMPLERATE = 48000

# Tabla 1 de BS.1770-4: primer paso, shelf agudo (modela la cabeza del oyente).
SHELF_B_48K = (1.53512485958697, -2.69169618940638, 1.19839281085285)
SHELF_A_48K = (1.0, -1.69065929318241, 0.73248077421585)

# Tabla 2 de BS.1770-4: segundo paso, pasa-altos RLB.
HIGHPASS_B_48K = (1.0, -2.0, 1.0)
HIGHPASS_A_48K = (1.0, -1.99004745483398, 0.99007225036621)

# Offset del estándar: convierte la potencia ponderada en la escala LKFS/LUFS.
LOUDNESS_OFFSET_DB = -0.691

BLOCK_SECONDS = 0.400
BLOCK_OVERLAP = 0.75

ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0

# El loudness range usa ventanas de 3 s y una compuerta relativa más severa.
SHORT_TERM_SECONDS = 3.0
LRA_RELATIVE_GATE_LU = -20.0
LRA_LOWER_PERCENTILE = 10.0
LRA_UPPER_PERCENTILE = 95.0

# El true peak se mide sobre la señal sobremuestreada, porque el pico real de
# la forma de onda reconstruida cae entre samples.
TRUE_PEAK_OVERSAMPLING = 4

# Objetivos de las plataformas de streaming, en LUFS integrados.
PLATFORM_TARGETS: dict[str, float] = {
    "Spotify": -14.0,
    "YouTube": -14.0,
    "Apple Music": -16.0,
    "Amazon Music": -14.0,
    "Tidal": -14.0,
    "Club / DJ playback": -7.0,
}


def _to_analog(b, a, samplerate: int):
    """Transformada bilineal inversa: del biquad digital a su prototipo analógico.

    La bilineal mapea el plano s al z con z = (1 + sT/2)/(1 - sT/2); invertirla
    sobre cada polo y cada cero da s = 2*fs*(z-1)/(z+1). La ganancia se fija
    igualando la respuesta en un punto que la transformada preserva: continua
    (z=1 -> s=0) si el filtro deja pasar continua, o Nyquist (z=-1 -> s=inf) si
    no, que es el caso del pasa-altos.
    """
    zeros, poles, _ = tf2zpk(np.asarray(b), np.asarray(a))

    analog_zeros = 2.0 * samplerate * (zeros - 1.0) / (zeros + 1.0)
    analog_poles = 2.0 * samplerate * (poles - 1.0) / (poles + 1.0)

    b_arr, a_arr = np.asarray(b, dtype=float), np.asarray(a, dtype=float)
    dc_gain = b_arr.sum() / a_arr.sum()

    if abs(dc_gain) > 1e-12:
        # H_a(0) = k * prod(-ceros) / prod(-polos), que debe valer H_d(z=1).
        analog_gain = dc_gain * np.prod(-analog_poles) / np.prod(-analog_zeros)
    else:
        # Filtro sin respuesta en continua: se iguala en Nyquist, donde con
        # igual número de ceros y polos la respuesta analógica vale k.
        nyquist_gain = ((b_arr[0] - b_arr[1] + b_arr[2])
                        / (a_arr[0] - a_arr[1] + a_arr[2]))
        analog_gain = nyquist_gain

    return analog_zeros, analog_poles, np.real(analog_gain)


def _redesign(b, a, samplerate: int) -> tuple[np.ndarray, np.ndarray]:
    """Lleva un biquad tabulado a 48 kHz hacia otro samplerate."""
    if samplerate == REFERENCE_SAMPLERATE:
        return np.asarray(b, dtype=float), np.asarray(a, dtype=float)

    zeros, poles, gain = _to_analog(b, a, REFERENCE_SAMPLERATE)
    new_zeros, new_poles, new_gain = bilinear_zpk(zeros, poles, gain, fs=samplerate)
    new_b, new_a = zpk2tf(new_zeros, new_poles, new_gain)
    return np.real(new_b), np.real(new_a)


def _shelf_coefficients(samplerate: int) -> tuple[np.ndarray, np.ndarray]:
    """Biquad shelf agudo del primer paso del K-weighting."""
    return _redesign(SHELF_B_48K, SHELF_A_48K, samplerate)


def _highpass_coefficients(samplerate: int) -> tuple[np.ndarray, np.ndarray]:
    """Biquad pasa-altos (RLB) del segundo paso del K-weighting."""
    return _redesign(HIGHPASS_B_48K, HIGHPASS_A_48K, samplerate)


def k_weight(samples: np.ndarray, samplerate: int) -> np.ndarray:
    """Aplica los dos biquads del K-weighting a cada canal."""
    shelf_b, shelf_a = _shelf_coefficients(samplerate)
    hp_b, hp_a = _highpass_coefficients(samplerate)

    filtered = lfilter(shelf_b, shelf_a, samples, axis=0)
    return lfilter(hp_b, hp_a, filtered, axis=0)


def channel_weights(channel_count: int) -> np.ndarray:
    """Ponderación por canal del estándar (`G_i`).

    Frente y mono pesan 1.0; los surround pesan 1.41 porque llegan al oyente
    desde ángulos donde se perciben más fuerte. El canal LFE no se mide.
    """
    if channel_count == 6:  # L R C LFE Ls Rs
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.ones(channel_count)


def _block_mean_squares(
    weighted: np.ndarray, samplerate: int, block_seconds: float, overlap: float
) -> np.ndarray:
    """Media cuadrática por canal de cada bloque solapado.

    Devuelve un array (bloques, canales). Se arma con `as_strided` a través de
    `sliding_window_view` para no copiar la señal una vez por bloque.
    """
    block_size = int(round(block_seconds * samplerate))
    step = max(1, int(round(block_size * (1.0 - overlap))))

    if weighted.shape[0] < block_size:
        raise ValueError(
            f"Audio is shorter than one {block_seconds * 1000:.0f} ms analysis "
            "block; loudness cannot be measured on it."
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        weighted, block_size, axis=0
    )[::step]
    return np.mean(windows ** 2, axis=-1)


def _blocks_to_loudness(mean_squares: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Loudness en LKFS de cada bloque, sumando los canales ponderados."""
    summed = mean_squares @ weights
    with np.errstate(divide="ignore"):
        return LOUDNESS_OFFSET_DB + 10.0 * np.log10(summed)


def _gated_loudness(
    mean_squares: np.ndarray, weights: np.ndarray
) -> tuple[float, int, int]:
    """Loudness integrado con la doble compuerta del estándar.

    Devuelve (LUFS, bloques que pasaron, bloques totales).

    La media se hace sobre las *potencias* de los bloques, no sobre sus valores
    en dB: promediar decibelios daría un número distinto y equivocado.
    """
    block_loudness = _blocks_to_loudness(mean_squares, weights)

    above_absolute = block_loudness > ABSOLUTE_GATE_LUFS
    if not above_absolute.any():
        return float("-inf"), 0, len(block_loudness)

    # Nivel de referencia para la compuerta relativa: promedio de potencias de
    # los bloques que pasaron la compuerta absoluta.
    reference_power = np.mean(mean_squares[above_absolute] @ weights)
    relative_threshold = (
        LOUDNESS_OFFSET_DB + 10.0 * np.log10(reference_power) + RELATIVE_GATE_LU
    )

    passing = above_absolute & (block_loudness > relative_threshold)
    if not passing.any():
        return float("-inf"), 0, len(block_loudness)

    gated_power = np.mean(mean_squares[passing] @ weights)
    integrated = LOUDNESS_OFFSET_DB + 10.0 * np.log10(gated_power)
    return float(integrated), int(passing.sum()), len(block_loudness)


def _loudness_range(mean_squares: np.ndarray, weights: np.ndarray) -> float:
    """Loudness range (LRA) según EBU Tech 3342.

    Es la distancia entre el percentil 10 y el 95 de los bloques de 3 s que
    sobreviven a una compuerta relativa de -20 LU: cuánto respira la canción
    entre sus partes suaves y sus partes fuertes.
    """
    short_term = _blocks_to_loudness(mean_squares, weights)
    above_absolute = short_term > ABSOLUTE_GATE_LUFS
    if not above_absolute.any():
        return 0.0

    reference_power = np.mean(mean_squares[above_absolute] @ weights)
    threshold = (
        LOUDNESS_OFFSET_DB + 10.0 * np.log10(reference_power) + LRA_RELATIVE_GATE_LU
    )

    passing = short_term[above_absolute & (short_term > threshold)]
    if passing.size < 2:
        return 0.0

    low, high = np.percentile(passing, [LRA_LOWER_PERCENTILE, LRA_UPPER_PERCENTILE])
    return float(high - low)


def true_peak_dbtp(samples: np.ndarray, samplerate: int) -> float:
    """Pico real (dBTP), medido sobre la señal sobremuestreada 4x.

    El pico entre muestras de la señal reconstruida puede superar al mayor
    sample del archivo: por eso un archivo que "solo" llega a -0.1 dBFS igual
    puede saturar el conversor de un reproductor.
    """
    upsampled = resample_poly(samples, TRUE_PEAK_OVERSAMPLING, 1, axis=0)
    peak = float(np.abs(upsampled).max())
    if peak <= 0.0:
        return -200.0
    return float(20.0 * np.log10(peak))


def analyze_loudness(samples: np.ndarray, samplerate: int) -> dict:
    """Mide el loudness de `samples` (frames, canales) según BS.1770-4."""
    frame_count, channel_count = samples.shape
    duration = frame_count / samplerate

    weighted = k_weight(samples, samplerate)
    weights = channel_weights(channel_count)

    momentary_ms = _block_mean_squares(weighted, samplerate, BLOCK_SECONDS, BLOCK_OVERLAP)
    integrated, used_blocks, total_blocks = _gated_loudness(momentary_ms, weights)
    momentary = _blocks_to_loudness(momentary_ms, weights)

    # El short-term (3 s) solo se puede medir si la pista dura al menos eso.
    if duration >= SHORT_TERM_SECONDS:
        short_term_ms = _block_mean_squares(
            weighted, samplerate, SHORT_TERM_SECONDS, BLOCK_OVERLAP
        )
        short_term = _blocks_to_loudness(short_term_ms, weights)
        loudness_range = _loudness_range(short_term_ms, weights)
        short_term_max = float(short_term.max())
    else:
        loudness_range = 0.0
        short_term_max = float("-inf")

    peak_dbtp = true_peak_dbtp(samples, samplerate)
    measurable = np.isfinite(integrated)

    return {
        "file_info": {
            "duration_seconds": round(duration, 3),
            "samplerate": samplerate,
            "channels": channel_count,
            "channel_layout": describe_channels(channel_count),
        },
        "integrated_lufs": round(integrated, 2) if measurable else None,
        "momentary_max_lufs": round(float(momentary.max()), 2) if measurable else None,
        "short_term_max_lufs": (
            round(short_term_max, 2) if np.isfinite(short_term_max) else None
        ),
        "loudness_range_lu": round(loudness_range, 2),
        "true_peak_dbtp": round(peak_dbtp, 2),
        "gating": {
            "blocks_total": total_blocks,
            "blocks_used": used_blocks,
            "absolute_gate_lufs": ABSOLUTE_GATE_LUFS,
            "relative_gate_lu": RELATIVE_GATE_LU,
        },
        "platform_targets": _platform_comparison(integrated) if measurable else [],
        "verdict": _verdict(integrated, loudness_range, peak_dbtp, measurable),
        "standard": "ITU-R BS.1770-4 (K-weighting + two-stage gating); LRA per EBU Tech 3342",
    }


def _platform_comparison(integrated: float) -> list[dict]:
    """Cuánto hay que subir o bajar para cada plataforma."""
    comparison = []
    for platform, target in PLATFORM_TARGETS.items():
        delta = integrated - target
        comparison.append({
            "platform": platform,
            "target_lufs": target,
            "difference_lu": round(delta, 2),
            "action": (
                "within 1 LU of target" if abs(delta) <= 1.0
                else f"turn down {abs(delta):.1f} LU" if delta > 0
                else f"turn up {abs(delta):.1f} LU"
            ),
        })
    return comparison


def _verdict(integrated: float, loudness_range: float, peak_dbtp: float,
             measurable: bool) -> str:
    if not measurable:
        return (
            "The material is below the -70 LUFS absolute gate — effectively "
            "silence, so there is no integrated loudness to report."
        )

    parts = [f"Integrated loudness is {integrated:.1f} LUFS."]

    if integrated > -9:
        parts.append(
            "That is very loud: streaming platforms will turn it down, and the "
            "limiting needed to get there usually costs transient punch."
        )
    elif integrated < -18:
        parts.append(
            "That is quiet by streaming standards; it will sit noticeably below "
            "commercial releases unless the platform normalizes upward."
        )

    if peak_dbtp > -1.0:
        parts.append(
            f"True peak reaches {peak_dbtp:.2f} dBTP, above the -1 dBTP ceiling "
            "recommended for lossy encoding — inter-sample peaks can distort on "
            "playback even though the file itself never clips."
        )

    if loudness_range and loudness_range < 3:
        parts.append(
            f"The loudness range is only {loudness_range:.1f} LU, so the track is "
            "dynamically very flat from section to section."
        )
    elif loudness_range > 15:
        parts.append(
            f"The loudness range is wide ({loudness_range:.1f} LU); quiet sections "
            "may get lost on phone speakers or in a car."
        )

    return " ".join(parts)
