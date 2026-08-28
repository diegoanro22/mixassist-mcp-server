# MixAssist MCP Server

An [MCP](https://modelcontextprotocol.io) server that analyzes audio mixes for independent music producers.

MixAssist **measures and explains** — it never modifies, re-renders, or "fixes" your audio. It answers questions like *"is my master loud enough for Spotify?"*, *"why does my low end sound muddy?"*, or *"will my mix collapse in mono?"* by running the actual DSP and reporting numbers a producer can act on.

The loudness metering implements ITU-R BS.1770-4 (K-weighting filters and the two-stage gating algorithm) directly on top of NumPy/SciPy rather than calling an existing loudness library.

Built for CC3067 Redes (Universidad del Valle de Guatemala), Project 1 — Use of an Existing Protocol.

## Tools

### `detect_clipping`

Finds digital clipping — samples pinned at the ceiling because the signal was pushed past full scale.

Counting samples at 0 dBFS is not enough on its own: an isolated peak at full scale is inaudible, while a *run* of consecutive pinned samples means the crest of the waveform was flattened, which is what actually sounds like distortion. This tool groups the offending samples into runs and reports the runs, with timestamps, so you can go listen to the worst ones.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file` | string | *required* | Path to a WAV/FLAC/AIFF file. |
| `threshold_dbfs` | number | `-0.1` | Level at or above which a sample counts as pinned to the ceiling. Must be 0 or negative. |
| `min_consecutive_samples` | integer | `3` | How many consecutive pinned samples make a clipping event rather than an isolated peak. |

Returns `peak_dbfs`, the total count and percentage of samples at the ceiling, the number of clipping events, the ten longest events (`channel`, `start_seconds`, `duration_ms`, `consecutive_samples`), a per-channel breakdown, a `severity` of `clean` / `peaks_at_ceiling` / `clipping`, and a plain-language `verdict`.

```jsonc
// detect_clipping({"audio_file": "mix.wav"}) on a deliberately overdriven 440 Hz tone
{
  "file_info": { "duration_seconds": 2.0, "samplerate": 48000, "channels": 1, "channel_layout": "mono" },
  "peak_dbfs": 0.0,
  "samples_at_or_over_threshold": 52000,
  "percent_of_samples_at_or_over_threshold": 54.166667,
  "clipping_events": 1760,
  "worst_events": [
    { "channel": 0, "start_seconds": 0.0003, "duration_ms": 0.625, "consecutive_samples": 30 }
  ],
  "severity": "clipping",
  "verdict": "Clipping detected: 1760 run(s) of consecutive samples pinned at the ceiling (peak 0.00 dBFS). The waveform is being flattened, which is audible as distortion. Lower the output gain before the limiter and re-render."
}
```

### `analyze_spectrum`

Reports how a mix distributes its energy across the frequency spectrum.

Computes a short-time Fourier transform (Hann window, 50% overlap, silent frames skipped), averages the power spectrum over time, and aggregates it into the seven bands producers actually talk about. Each band comes back with its share of total energy, its level in dB relative to the whole mix, and what that region is responsible for musically.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file` | string | *required* | Path to a WAV/FLAC/AIFF file. |
| `window_size` | integer | `4096` | FFT size in samples; power of two, minimum 256. Larger windows resolve low frequencies better at the cost of time resolution. |

Bands: `sub_bass` (20–60 Hz), `bass` (60–250), `low_mid` (250–500), `mid` (500–2k), `high_mid` (2k–4k), `presence` (4k–6k), `air` (6k–20k). Bands above Nyquist are omitted for low-samplerate files.

Also returns the **spectral centroid** — the centre of gravity of the spectrum in Hz, a single number for "dark" versus "bright" — the `dominant_band`, and a list of `observations` phrased as rules of thumb rather than verdicts.

```jsonc
// analyze_spectrum({"audio_file": "mix.wav"}) on a bass-heavy test signal
{
  "analysis": { "window_size": 4096, "hop_size": 2048, "frequency_resolution_hz": 11.719, "window": "hann" },
  "spectral_centroid_hz": 85.3,
  "dominant_band": "sub_bass",
  "bands": [
    { "band": "sub_bass", "range_hz": [20, 60],  "energy_share_percent": 79.707, "relative_db": -0.99,
      "meaning": "Weight you feel more than hear; kick and 808 fundamentals." },
    { "band": "bass",     "range_hz": [60, 250], "energy_share_percent": 20.245, "relative_db": -6.94,
      "meaning": "Bass guitar, kick body, the low end of the groove." }
  ],
  "observations": [
    "The low end holds 100.0% of the energy, which is a lot even for bass-forward genres - check whether the kick and bass are stacking up.",
    "The spectral centroid is low (85 Hz): a dark overall balance."
  ]
}
```

### `analyze_loudness`

Measures perceived loudness to ITU-R BS.1770-4 and says how far the master is from each platform's target.

This is the tool that answers *"is it loud enough?"*. It reports **integrated loudness** in LUFS — the single number Spotify, YouTube and Apple Music normalize against — plus momentary and short-term maxima, the **loudness range** (how much the track breathes between its quiet and loud sections, per EBU Tech 3342), and the **true peak** in dBTP measured on a 4x oversampled signal, which catches inter-sample peaks that a plain sample-peak reading misses entirely.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file` | string | *required* | Path to a WAV/FLAC/AIFF file, at least 400 ms long. |

```jsonc
// analyze_loudness({"audio_file": "master.wav"})
{
  "integrated_lufs": -5.6,
  "momentary_max_lufs": -5.6,
  "short_term_max_lufs": -5.6,
  "loudness_range_lu": 0.0,
  "true_peak_dbtp": -0.44,
  "gating": { "blocks_total": 77, "blocks_used": 77,
              "absolute_gate_lufs": -70.0, "relative_gate_lu": -10.0 },
  "platform_targets": [
    { "platform": "Spotify", "target_lufs": -14.0, "difference_lu": 8.4, "action": "turn down 8.4 LU" }
  ],
  "verdict": "Integrated loudness is -5.6 LUFS. That is very loud: streaming platforms will turn it down, and the limiting needed to get there usually costs transient punch. True peak reaches -0.44 dBTP, above the -1 dBTP ceiling recommended for lossy encoding...",
  "standard": "ITU-R BS.1770-4 (K-weighting + two-stage gating); LRA per EBU Tech 3342"
}
```

#### How the measurement is implemented

The standard is implemented rather than approximated, which is worth spelling out because the details are where loudness meters usually go wrong:

- **K-weighting.** Two cascaded biquads: a high shelf modelling the head as an acoustic obstacle, then a high-pass discarding low frequencies that contribute little to perceived loudness. BS.1770-4 tabulates these coefficients **for 48 kHz only**. The widely-used "audio EQ cookbook" shelf formula does *not* reproduce that table — it is off by 3.6%, because the standard starts from a different analog prototype. So instead of guessing a formula, this implementation takes the tabulated coefficients, recovers the underlying analog filter by inverting the bilinear transform on its poles and zeros, and re-discretizes it at whatever sample rate the file uses. At 48 kHz the result is bit-identical to the table; at 44.1 kHz the frequency response matches to within 0.003 dB.
- **Gated measurement.** 400 ms blocks at 75% overlap, then the two-stage gate: absolute at -70 LUFS, then relative at 10 LU below the ungated mean. The relative gate is the step naive implementations skip, and without it any track with quiet passages reads several LU too low. Block powers are averaged, never decibels.
- **Calibration.** A 1 kHz sine at -23 dBFS RMS reads exactly -23.0 LUFS, which is the calibration point the standard defines.

Cross-checked against `pyloudnorm`, an independent implementation, across sine, noise and gated program material at both 44.1 and 48 kHz: the readings agree to within a constant 0.043 LU. That offset is fully accounted for — it is the passband gain of the standard's own high-pass coefficients (`b = [1, -2, 1]` against that denominator gives exactly 0.04328 dB), which this implementation keeps and `pyloudnorm` normalizes away.

### Error handling

Input problems — a missing file, an unreadable format, a parameter out of range — come back as a normal MCP tool error (`is_error: true`) carrying a message meant to be read by a human:

```
Error executing tool detect_clipping: Audio file not found: no-existe.wav
Error executing tool detect_clipping: threshold_dbfs must be 0 or negative (dBFS scale).
```

### Roadmap

`detect_frequency_masking`, `analyze_phase_correlation` and `compare_to_reference` are in progress.

## Requirements

- Python 3.11+
- The dependencies declared in `pyproject.toml` (`mcp`, `numpy`, `scipy`, `soundfile`) — installed automatically.

Audio formats: anything `libsndfile` reads (WAV, FLAC, AIFF, OGG). MP3 support depends on your `libsndfile` build.

## Installation

```bash
git clone https://github.com/diegoanro22/mixassist-mcp-server.git
cd mixassist-mcp-server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### As a local (stdio) MCP server

This is how MCP hosts normally run it — the host spawns the process and speaks JSON-RPC over stdin/stdout:

```bash
mixassist-mcp                    # equivalent to: mixassist-mcp --transport stdio
```

Register it in your host's server configuration. For example, in a `servers.yaml`-style registry:

```yaml
- name: mixassist
  transport: stdio
  command: /path/to/mixassist-mcp-server/.venv/bin/mixassist-mcp
```

Or, in Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mixassist": {
      "command": "/path/to/mixassist-mcp-server/.venv/bin/mixassist-mcp"
    }
  }
}
```

### Over the network (HTTP)

The server also speaks MCP's streamable HTTP transport, so a host on another
machine can use it:

```bash
mixassist-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

Then point your host at `http://<server-ip>:8000/mcp`. Bind to `0.0.0.0` only on
a network you trust — the server has no authentication and reads local audio files.

### Inspecting it manually

To browse the tools and their JSON schemas without writing a host:

```bash
npx @modelcontextprotocol/inspector mixassist-mcp
```

## Protocol notes

MixAssist is built on the official Python MCP SDK, version **2.x**, where the
class formerly known as `FastMCP` is now `MCPServer`
(`from mcp.server.mcpserver import MCPServer`). Examples written for the 1.x SDK
will not run unchanged.

A session follows the standard MCP lifecycle over JSON-RPC 2.0: the host sends
`initialize`, the server replies with its capabilities, the host confirms with an
`initialized` notification, and from then on `tools/list` and `tools/call`
requests carry the actual work.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](./LICENSE).
