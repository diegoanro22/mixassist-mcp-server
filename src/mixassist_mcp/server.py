"""Punto de entrada del servidor MCP MixAssist.

Registra las herramientas de análisis de mezcla y las expone sobre el Model
Context Protocol (JSON-RPC 2.0). Esta capa es deliberadamente delgada: valida
la entrada, delega en los módulos de `mixassist_mcp.dsp` y devuelve datos
planos. Los algoritmos viven en `dsp/`, no aquí.

Transportes disponibles (SDK de MCP 2.x):
  - stdio            -> el proceso anfitrión hace spawn del servidor (por defecto)
  - streamable-http  -> el servidor escucha en un puerto, para uso por red

Nota sobre el SDK: en `mcp` 2.x la clase `FastMCP` de la 1.x se renombró a
`MCPServer`. Los ejemplos de la 1.x que circulan en línea no corren tal cual.
"""

import argparse
from contextlib import contextmanager
from collections.abc import Iterator

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .dsp import clipping as clipping_dsp
from .dsp import loudness as loudness_dsp
from .dsp import spectrum as spectrum_dsp
from .dsp.audio import AudioError, load_audio

INSTRUCTIONS = """\
MixAssist analyzes audio mixes and explains what is wrong with them and why.

It measures and reports; it never modifies or re-renders audio. Use it to
inspect loudness (ITU-R BS.1770), spectral balance across bands, digital
clipping, stereo phase correlation, and frequency masking between stems.

All tools take filesystem paths to WAV/FLAC/AIFF files and return plain
numeric measurements plus a short verdict, so the results can be quoted
directly to the user.\
"""

server = MCPServer(
    name="mixassist",
    title="MixAssist — audio mix analysis",
    version=__version__,
    instructions=INSTRUCTIONS,
)


@contextmanager
def _user_errors() -> Iterator[None]:
    """Convierte errores de entrada en `ToolError`.

    El SDK trata cualquier excepción que no sea `ToolError` como una caída del
    servidor y oculta el mensaje al cliente (para no filtrar detalles internos).
    Los problemas que sí son culpa de la entrada —archivo inexistente, formato
    ilegible, parámetro fuera de rango— tienen que llegarle al anfitrión con su
    texto intacto, porque es lo que el LLM le explica al usuario.
    """
    try:
        yield
    except (AudioError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


# --- Herramientas -----------------------------------------------------------
#
# Los docstrings de las tools van en inglés a propósito: no son comentarios
# internos, son la descripción que viaja por el protocolo en `tools/list` y que
# el LLM del anfitrión lee para decidir cuándo invocarlas.


@server.tool()
def detect_clipping(
    audio_file: str,
    threshold_dbfs: float = clipping_dsp.DEFAULT_THRESHOLD_DBFS,
    min_consecutive_samples: int = clipping_dsp.DEFAULT_MIN_RUN,
) -> dict:
    """Detect digital clipping in an audio file.

    Reports the true sample peak, how many samples sit at or above the ceiling,
    and — more importantly — groups them into runs of consecutive pinned
    samples. A single sample at full scale is harmless; a run of consecutive
    ones means the waveform crest was flattened, which is what actually sounds
    like distortion. The worst runs are returned with their timestamps so the
    user can go listen to them.

    Args:
        audio_file: Path to a WAV/FLAC/AIFF file to analyze.
        threshold_dbfs: Level at or above which a sample counts as pinned to the
            ceiling. Must be 0 or negative. Defaults to -0.1 dBFS.
        min_consecutive_samples: How many consecutive pinned samples constitute
            a clipping event rather than an isolated peak. Defaults to 3.
    """
    with _user_errors():
        samples, samplerate = load_audio(audio_file)
        return clipping_dsp.detect_clipping(
            samples,
            samplerate,
            threshold_dbfs=threshold_dbfs,
            min_consecutive_samples=min_consecutive_samples,
        )


@server.tool()
def analyze_spectrum(
    audio_file: str,
    window_size: int = spectrum_dsp.DEFAULT_WINDOW_SIZE,
) -> dict:
    """Report how a mix distributes its energy across the frequency spectrum.

    Computes a short-time Fourier transform (Hann window, 50% overlap, silent
    frames skipped), averages the power spectrum over time, and aggregates it
    into the seven bands producers actually talk about: sub-bass, bass,
    low-mid, mid, high-mid, presence and air. Each band comes back with its
    share of the total energy, its level in dB relative to the whole mix, and
    what that region is responsible for musically. Also returns the spectral
    centroid, a single number describing whether the balance is dark or bright.

    Use it to answer questions about tonal balance — muddiness, thin low end,
    dull or harsh top end — not about level, which is what analyze_loudness
    and detect_clipping cover.

    Args:
        audio_file: Path to a WAV/FLAC/AIFF file to analyze.
        window_size: FFT size in samples; must be a power of two, at least 256.
            Larger windows resolve low frequencies better at the cost of time
            resolution. Defaults to 4096.
    """
    with _user_errors():
        samples, samplerate = load_audio(audio_file)
        return spectrum_dsp.analyze_spectrum(samples, samplerate, window_size=window_size)


@server.tool()
def analyze_loudness(audio_file: str) -> dict:
    """Measure perceived loudness to ITU-R BS.1770-4 and compare it to platform targets.

    Returns integrated loudness in LUFS (the number streaming services
    normalize against), the momentary and short-term maxima, the loudness range
    in LU, and the true peak in dBTP measured on a 4x oversampled signal — so
    it catches inter-sample peaks that a plain sample-peak reading misses.

    The measurement implements the standard rather than approximating it:
    K-weighting filters taken from the specification's coefficient tables, 400 ms
    blocks at 75% overlap, and the two-stage gating (absolute at -70 LUFS, then
    relative at -10 LU below the ungated mean) that stops quiet passages from
    dragging the reading down.

    Also reports, per platform (Spotify, YouTube, Apple Music, Amazon, Tidal,
    club playback), how many LU the master would have to move to hit target.

    Use it for questions about level and release readiness. For tonal balance
    use analyze_spectrum; for distortion from over-driving, detect_clipping.

    Args:
        audio_file: Path to a WAV/FLAC/AIFF file. Must be at least 400 ms long.
    """
    with _user_errors():
        samples, samplerate = load_audio(audio_file)
        return loudness_dsp.analyze_loudness(samples, samplerate)


def main() -> None:
    """Arranca el servidor con el transporte pedido por línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="mixassist-mcp",
        description="MixAssist MCP server — audio mix analysis tools.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transporte MCP a usar (por defecto: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interfaz donde escuchar en transportes de red. Usar 0.0.0.0 "
             "para aceptar conexiones desde otras máquinas de la LAN.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto donde escuchar en transportes de red (por defecto: 8000).",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        # host y port van como kwargs de run(), que los reenvía al transporte.
        # En el SDK 1.x vivían en server.settings; en el 2.x esos campos ya no
        # existen y asignarlos revienta con ValueError antes de escuchar nada.
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
