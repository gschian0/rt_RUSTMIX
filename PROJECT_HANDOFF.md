# rt_RUSTMIX project handoff

This document is a readable map of the work completed so far. It is intended
to make it possible to pick up the project without reconstructing the previous
development conversation.

## Project in one sentence

`rt_RUSTMIX` is a Rust real-time audio/graphics laboratory: an RTcmix-inspired
score language drives procedural voices and effects while their FFT activity
drives GPU Gaussian splats, reaction-diffusion fields, and experimental
shaders.

## Historical motivation and intended form

The project follows a line from Paul Lansky's pre-1985 C-language CMIX through
the real-time RTcmix work begun on SGI systems in the mid-1990s. RTcmix later
expanded with bus routing, dynamic pfields, scripting interfaces, a much larger
instrument library, and an embeddable C/C++ object. Its published history
specifically connects embedding with the need for close data sharing between
algorithmic audio and a host application:

- [The history of CMIX and RTcmix](https://rtcmix.org/about/index.html)
- [Creating an RTcmix Instrument](https://rtcmix.org/tutorials/instrumentdesign.html)

RTcmix instruments broadly separate note initialization from repeated real-time
sample computation. `rt_RUSTMIX` preserves that useful idea in Rust: parsing,
allocation, and note preparation belong outside the callback; scheduled
`Instrument` objects perform bounded processing inside it.

The desktop GUI is a demonstration host, not the intended boundary of the
system. The longer-term objective is a fully embeddable library for bespoke
audiovisual programs, installations, games, instruments, research software,
and embedded devices. The crate split already supports selective reuse, but
embedded-device support is not yet complete: each target still needs an audio
host, GPU or display strategy where applicable, memory/CPU limits, and
real-time validation.

## How the system is divided

The workspace is split into focused crates:

| Area | Location | Responsibility |
| --- | --- | --- |
| Engine | `crates/rtmix-core` | Real-time scheduling, buses, instruments, shared state, and audio-safe queues |
| DSP | `crates/rtmix-dsp` | Formant, FOF, chant, wavetable, LPC-style, and articulated speech synthesis |
| Score language | `crates/rtmix-minc` | RTcmix-like calls, pfields, note scheduling, and bus dispatch |
| Spatial audio | `crates/rtmix-spat` | Spatial routing and rotation/decode helpers |
| Graphics data | `crates/rtmix-gfx` | Splat/PLY/SPZ loading, FFT analysis, CPU preview, and graphics shader build path |
| GUI | `crates/rtmix-gui` | Desktop app, generated modes, controls, GPU callbacks, and live WGSL |
| Host | `crates/rtmix-host` | Audio device integration |
| WASM | `crates/rtmix-wasm` | Browser-oriented integration |
| Application | `apps/rtmix` | Thin executable entry point |

## Audio and score work

The score system supports RTcmix-style calls rather than embedding the upstream
RTcmix runtime. The implemented surfaces include:

- `FORMANT`: pitched resonant/formant tones.
- `CHANT`: vowel/formant phrase textures.
- `FOF`: overlapping formant-band grains.
- `LPC`: synthetic LPC-like vocal/noise behavior; it is not a drop-in
  `LPCPLAY` analyzer/player.
- `SPEAK`: typed phrases converted into phones with glottal excitation, vowel
  formants, coarticulation, prosody, jitter, grit, and panning.
- `WAVETABLE`: harmonic-table oscillators.
- `FILTER`, `DELAY`, `REVERB`, `MIX`: effects and bus shaping.
- `SPAT`, `ROTATE`, `DECODE`: experimental spatial routing.

The GUI generates 16-second looping patches. Current named modes are:

- **Constellation** — balanced vocal/formant texture.
- **Lansky Chatter** — overlapping short vocal cells.
- **Ritual** — slow, sparse ceremonial pulses.
- **Pulse** — fast formant bursts.
- **Prism** — bright spectral motion.
- **Speech** — articulated typed speech.
- **Whisper Swarm** — breathy layered speech fragments.
- **Robot Oracle** — quantized synthetic voices.
- **Broken Choir** — fractured octave speech layers.
- **Wavetable Lab** — harmonic tables through filters and time effects.
- **FOF Galaxy** — formant grain clouds with chant and echoes.
- **Phone Matrix** — LPC/SPEAK/FOF crosscutting.
- **Spatial Cyclone** — wavetable and FOF material through spatial effects.
- **Granular Bloom** — scale-quantized overlapping micro-grains, a sustained
  harmonic bed, FOF shimmer, delay, and reverb.

Granular Bloom is currently a musical score-level granular texture built from
short scheduled wavetable grains. It is intentionally deterministic enough to
have a tonal center while varying scale degree, octave, table, and pan. A future
dedicated sample/grain instrument could add source buffers, true time-stretch,
and grain envelopes without changing the score architecture.

## Speech implementation

[`crates/rtmix-dsp/src/speech.rs`](crates/rtmix-dsp/src/speech.rs) contains the
articulated speech synthesizer. The GUI exposes `SPEAK(start, dur, amp, pitch,
"text", grit, pan, seed)`. The implementation maps text to phone-like units,
smooths vowel transitions, shapes glottal excitation, and layers resonators.
The result is synthetic speech intended for expressive and intentionally
degraded textures, not natural-language production speech.

## Graphics pipeline

### Gaussian splats

[`crates/rtmix-gfx/src/splat.rs`](crates/rtmix-gfx/src/splat.rs) loads and
validates splat records. Assets are budget-loaded so multi-million-point scenes
do not have to be fully decoded when a smaller detail budget is selected.

The live desktop renderer is in
[`crates/rtmix-gui/src/gpu_visual.rs`](crates/rtmix-gui/src/gpu_visual.rs).
It uploads covariance-rich splat records to GPU storage buffers, sorts them in
Morton order, chunks large buffers to stay below the DirectX 12 storage binding
limit, and renders instanced oriented Gaussian ellipses.

The live shader is
[`crates/rtmix-gui/src/gpu_visual.wgsl`](crates/rtmix-gui/src/gpu_visual.wgsl).
It performs camera-space covariance projection, animated zoom/twist/displacement,
audio-reactive scaling, spectral color modulation, and chemical color/opacity
modulation.

### 3D Gaussian Gray-Scott-like reaction

Each uploaded splat receives a two-component chemical state `(A, B)`. Two GPU
compute passes update the state using neighboring Morton-local splats:

```text
-64, -16, -4, -1, +1, +4, +16, +64
```

This is a fast locality approximation, not a true Euclidean k-nearest-neighbor
graph. The reaction is Gray-Scott-like: it uses feed, kill, diffusion, and the
`A * B * B` reaction term. Audio bands and pulses inject additional B chemical.
The Gaussian shader uses B to increase splat scale, emission, opacity, and
color contrast.

Select **Gaussian Gray-Scott (3D GPU)** in the visual dropdown to see this
explicitly. Ordinary **Gaussian Splat (GPU)** uses the same live renderer but
does not advertise the reaction mode.

### 2D reaction visuals

The existing 2D visual modes use 512x288 ping-pong chemical buffers:

1. reaction pass A -> B,
2. reaction pass B -> A,
3. domain-transform/color compute pass,
4. fullscreen presentation.

The CPU reaction field in [`crates/rtmix-gfx/src/drip.rs`](crates/rtmix-gfx/src/drip.rs)
remains useful as a small testable fallback/reference.

## WGSL, wgpu, and Slang: what is actually used

These technologies have different roles:

- **wgpu** is the runtime graphics API used to create the GPU device, buffers,
  pipelines, bind groups, compute passes, and render passes.
- **WGSL** is the shader language used by the live GUI Gaussian/reaction path in
  `crates/rtmix-gui/src/gpu_visual.wgsl`.
- **Slang** is an optional shader authoring/compiler path for lower-level
  graphics pipelines in `crates/rtmix-gfx/shaders/`. The build script searches
  for `slangc`, compiles selected entry points to SPIR-V when available, and
  keeps matching WGSL shaders as the fallback.

In short: **wgpu runs the GPU work, WGSL currently drives the live GUI reaction
renderer, and Slang can compile optional graphics compute pipelines.**

## Running and testing

From the workspace root:

```powershell
cargo run --release -p rtmix
cargo test -p rtmix-gui
cargo test -p rtmix-gfx --no-default-features
cargo build --release -p rtmix
```

On Windows, a running `target\release\rtmix.exe` can lock the executable during
a rebuild. Stop that exact process before rebuilding if Cargo reports an access
denied error. The desktop path has been validated with `wgpu`'s DirectX 12
backend.

## Known limitations and next useful work

- The 3D reaction neighbor graph is Morton-local rather than geometric
  k-nearest-neighbor.
- Transparent splats do not yet use view-dependent depth sorting or weighted
  blended order-independent transparency.
- Granular Bloom is scheduled wavetable granulation, not yet a true sampled
  granular engine with a source buffer.
- The LPC implementation is synthetic and does not consume analyzed speech
  datasets.
- Example splat assets are third-party files and must be reviewed individually
  before redistribution.
- Musicality can continue to improve through explicit modal progressions,
  phrase-level motifs, and per-mode mix calibration.

## References and acknowledgements

This project builds on ideas from:

- [RTcmix](https://rtcmix.org/) and its
  [source repository](https://github.com/RTcmix/RTcmix)
- [RTcmix reference documentation](https://rtcmix.org/reference/)
- [Khronos Slang](https://github.com/shader-slang/slang) and its
  [documentation](https://shader-slang.org/)
- [Khronos Group](https://www.khronos.org/)
- [wgpu](https://wgpu.rs/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Hugging Face splat-data](https://huggingface.co/cakewalk/splat-data)

RTcmix and Slang are credited as upstream inspirations/tools and are not bundled
as proprietary copies in this repository.

## License

The original source in this repository is released under the
[Apache License, Version 2.0](LICENSE). Dependencies, shader toolchains, and
downloaded assets may have separate licenses.
