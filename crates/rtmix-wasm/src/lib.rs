//! WebAudio entry points over the same core, spat, and Minc crates.
//! Splats stay behind the `webgpu` feature so audio still builds without a GPU.

use rtmix_core::Engine;
use rtmix_minc::execute;

pub const WORKLET_JS: &str = include_str!("worklet.js");

pub struct Mixer {
    engine: Engine,
    cursor: usize,
}

impl Mixer {
    pub fn new(sample_rate: u32, channels: u32, block: u32) -> Self {
        Self {
            engine: Engine::new(sample_rate.max(1), channels.max(1) as usize, block.max(1) as usize),
            cursor: 0,
        }
    }

    pub fn load_score(&mut self, score: &str) -> Result<(), String> {
        execute(score, &mut self.engine).map(|_| ()).map_err(|err| format!("line {}: {}", err.line, err.message))
    }

    pub fn set_listener(&self, yaw: f32, pitch: f32, roll: f32) {
        self.engine.set_listener(yaw, pitch, roll);
    }

    pub fn sample_time(&self) -> f64 {
        self.engine.seconds()
    }

    /// Interleaved output used by the AudioWorklet processor.
    pub fn process(&mut self, interleaved: &mut [f32]) {
        let channels = self.engine.output_channels().max(1);
        let block = self.engine.block_size();
        let frames = interleaved.len() / channels;
        let mut written = 0;
        while written < frames {
            if self.cursor == 0 {
                self.engine.process_block();
            }
            let n = (block - self.cursor).min(frames - written);
            let offset = written * channels;
            self.engine.copy_block_interleaved(
                self.cursor,
                n,
                &mut interleaved[offset..offset + n * channels],
                channels,
            );
            self.cursor += n;
            if self.cursor >= block {
                self.cursor = 0;
            }
            written += n;
        }
    }
}

#[cfg(feature = "webgpu")]
pub fn splats_need_webgpu() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
mod bindings {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmMixer {
        inner: super::Mixer,
    }

    #[wasm_bindgen]
    impl WasmMixer {
        #[wasm_bindgen(constructor)]
        pub fn new(sample_rate: u32, channels: u32, block: u32) -> WasmMixer {
            WasmMixer {
                inner: super::Mixer::new(sample_rate, channels, block),
            }
        }

        pub fn load_score(&mut self, score: &str) -> Result<(), JsValue> {
            self.inner.load_score(score).map_err(JsValue::from)
        }

        pub fn set_listener(&self, yaw: f32, pitch: f32, roll: f32) {
            self.inner.set_listener(yaw, pitch, roll);
        }

        pub fn process(&mut self, interleaved: &mut [f32]) {
            self.inner.process(interleaved);
        }

        pub fn sample_time(&self) -> f64 {
            self.inner.sample_time()
        }
    }

    #[wasm_bindgen]
    pub fn worklet_js() -> String {
        super::WORKLET_JS.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worklet_score_renders_without_webgpu() {
        let mut mixer = Mixer::new(48_000, 2, 128);
        mixer
            .load_score(
                r#"
                rtsetparams(48000, 2, 128)
                bus_config("WAVETABLE", "out")
                WAVETABLE(0, 0.02, 0.2, 440, 0.5)
            "#,
            )
            .unwrap();
        let mut out = vec![0.0f32; 256];
        mixer.process(&mut out);
        assert!(out.iter().any(|s| s.abs() > 0.0));
        assert!(WORKLET_JS.contains("registerProcessor"));
        assert!(WORKLET_JS.contains("RustmixProcessor"));
    }
}
