# rt_RUSTMIX

`rt_RUSTMIX` is an experimental real-time Rust instrument and visualizer inspired
by the musical patching model of [RTcmix](https://rtcmix.org/). It combines
procedural vocal synthesis, RTcmix-style score scheduling, GPU Gaussian splats,
reaction-diffusion visuals, and audio-reactive rendering.

This is a Rust implementation inspired by RTcmix techniques and score concepts;
it is not the upstream RTcmix codebase and does not claim binary or algorithmic
compatibility with every RTcmix instrument.

## Why this project exists

RTcmix descends from Paul Lansky's C-language CMIX work, which itself followed
the earlier MUSIC-family approach to computer music. CMIX predates 1985;
real-time RTcmix development began on SGI systems in the mid-1990s, and its
source became publicly available in 1999. Over time the project gained bus
routing, dynamic pfields, new instruments, scripting interfaces, and an
embeddable C/C++ object.

That embedded direction is especially important here. The
[RTcmix history](https://rtcmix.org/about/index.html) describes a desire to put
algorithmic music directly inside host software with close data sharing rather
than treating synthesis as an isolated external program. `rt_RUSTMIX` carries
that idea into Rust: the long-term goal is a small set of reusable crates that
can be embedded in one-off audiovisual works, installations, games, research
tools, instruments, and resource-constrained devices without requiring the
desktop GUI.

The [RTcmix instrument-design model](https://rtcmix.org/tutorials/instrumentdesign.html)
is another direct inspiration. An RTcmix instrument initializes a scheduled
note from score pfields and then repeatedly computes blocks of samples. Our
Rust engine follows the same broad separation: score parsing and note
construction happen outside the audio callback, while an `Instrument`
implementation performs bounded real-time sample processing. This is an
architectural influence, not a source-compatible port of the C++ API.

The current desktop application demonstrates the libraries, but the intended
architecture keeps synthesis, scheduling, spatial audio, graphics data, and
hosting in separate crates so future applications can select only the pieces
they need. True embedded-device targets still require platform hosts, resource
budgets, and target-specific validation.

## What is here

- Real-time stereo audio engine with lock-free audio callback paths.
- RTcmix-like score calls for `FORMANT`, `CHANT`, `FOF`, `LPC`, `SPEAK`,
  `WAVETABLE`, `FILTER`, `DELAY`, `REVERB`, `SPAT`, `ROTATE`, and `DECODE`.
- Articulated text-to-speech synthesis with phones, formants, prosody, grit,
  jitter, and stereo layering.
- Experimental modes including Speech, Whisper Swarm, Robot Oracle, Broken
  Choir, Wavetable Lab, FOF Galaxy, Phone Matrix, Spatial Cyclone, and
  Granular Bloom.
- `.splat`, PLY, and SPZ loading plus free Gaussian-splat scene support.
- GPU Gaussian rendering through `eframe`/`wgpu` and WGSL.
- GPU 2D reaction-diffusion/domain-transform visuals.
- A per-splat 3D Gray-Scott-like reaction state using Morton-local neighbors.
- Audio analysis with FFT bands, RMS, centroid, and flux driving the visuals.
- Optional Slang-to-SPIR-V compilation for the lower-level graphics pipelines,
  with matching WGSL fallbacks when `slangc` is unavailable.

## Quick start

Install the stable Rust toolchain, then run:

```powershell
cargo run --release -p rtmix
```

On Windows, the desktop renderer currently uses `wgpu` with DirectX 12. The
application can run without `slangc`; the build script automatically uses the
matching WGSL shader path when Slang is not installed.

Download the example scenes into [`assets/`](assets/) using the instructions in
[`assets/README.md`](assets/README.md). They are third-party assets with
individual source licenses; check those terms before redistributing them.

## Picking up the project

Read [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) for the detailed implementation
history, architecture map, current behavior, known limitations, and practical
next steps.

Important entry points:

- [`crates/rtmix-gui/src/lib.rs`](crates/rtmix-gui/src/lib.rs): GUI, score modes,
  visual controls, and generated patches.
- [`crates/rtmix-gui/src/gpu_visual.rs`](crates/rtmix-gui/src/gpu_visual.rs):
  GPU resources, callbacks, splat uploads, and reaction passes.
- [`crates/rtmix-gui/src/gpu_visual.wgsl`](crates/rtmix-gui/src/gpu_visual.wgsl):
  live Gaussian and reaction shaders.
- [`crates/rtmix-dsp/src/`](crates/rtmix-dsp/src/): synthesis instruments.
- [`crates/rtmix-minc/src/lib.rs`](crates/rtmix-minc/src/lib.rs): score-language
  dispatch.
- [`crates/rtmix-gfx/build.rs`](crates/rtmix-gfx/build.rs): optional Slang
  compilation.

## Acknowledgements and further reading

This project is a tribute to the experimental computer-music tradition and
draws especially on:

- [RTcmix](https://rtcmix.org/) — real-time computer music language and
  instruments.
- [RTcmix source repository](https://github.com/RTcmix/RTcmix)
- [RTcmix documentation](https://rtcmix.org/reference/)
- [The history of CMIX and RTcmix](https://rtcmix.org/about/index.html)
- [Creating an RTcmix Instrument](https://rtcmix.org/tutorials/instrumentdesign.html)
- [Khronos Slang](https://github.com/shader-slang/slang) — shader language and
  compiler used here as an optional authoring/compilation route.
- [Slang documentation](https://shader-slang.org/)
- [Khronos Group](https://www.khronos.org/) — open graphics and compute
  standards.
- [wgpu](https://wgpu.rs/) — Rust WebGPU implementation used by the desktop
  renderer.
- [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
  — the rendering technique behind the scene viewer.
- [Hugging Face splat-data](https://huggingface.co/cakewalk/splat-data) — source
  of the example scenes used during development.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).
Third-party assets and dependencies remain subject to their own licenses.
