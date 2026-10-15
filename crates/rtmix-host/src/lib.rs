//! cpal output. The callback only drains the control queue on block boundaries
//! and copies already rendered buses. It does not lock, parse, or talk to the GPU.

use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtmix_core::{Engine, Shared};

#[derive(Debug)]
pub struct HostError {
    pub message: String,
}

pub struct AudioDriver {
    stream: cpal::Stream,
    shared: Arc<Shared>,
    sample_rate: u32,
    channels: usize,
}

impl AudioDriver {
    pub fn start(mut engine: Engine) -> Result<Self, (HostError, Engine)> {
        let shared = engine.shared();
        let sample_rate = engine.sample_rate();
        let block = engine.block_size();
        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            return Err((
                HostError {
                    message: "no output device".into(),
                },
                engine,
            ));
        };
        let config = match device.default_output_config() {
            Ok(config) => config,
            Err(err) => {
                return Err((
                    HostError {
                        message: err.to_string(),
                    },
                    engine,
                ))
            }
        };
        let channels = config.channels() as usize;
        let stream_config: cpal::StreamConfig = config.clone().into();
        if config.sample_format() != cpal::SampleFormat::F32 {
            return Err((
                HostError {
                    message: "output device is not f32".into(),
                },
                engine,
            ));
        }
        let err_fn = |err| eprintln!("rtmix audio: {err}");
        let mut cursor = 0usize;
        let built = device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                fill(&mut engine, &mut cursor, block, channels, data);
            },
            err_fn,
            None,
        );
        let stream = match built {
            Ok(stream) => stream,
            Err(err) => {
                return Err((
                    HostError {
                        message: err.to_string(),
                    },
                    // The closure already owns the scheduled engine.
                    Engine::new(sample_rate, channels.max(1), block),
                ));
            }
        };
        if let Err(err) = stream.play() {
            return Err((
                HostError {
                    message: err.to_string(),
                },
                Engine::new(sample_rate, channels.max(1), block),
            ));
        }
        Ok(Self {
            stream,
            shared,
            sample_rate,
            channels,
        })
    }

    pub fn shared(&self) -> Arc<Shared> {
        Arc::clone(&self.shared)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        let _ = &self.stream;
        self.channels
    }
}

fn fill(engine: &mut Engine, cursor: &mut usize, block: usize, channels: usize, data: &mut [f32]) {
    let channels = channels.max(1);
    let frames = data.len() / channels;
    let mut written = 0;
    while written < frames {
        if *cursor == 0 {
            engine.process_block();
        }
        let n = (block - *cursor).min(frames - written);
        let offset = written * channels;
        let end = (offset + n * channels).min(data.len());
        engine.copy_block_interleaved(*cursor, n, &mut data[offset..end], channels);
        *cursor += n;
        if *cursor >= block {
            *cursor = 0;
        }
        written += n;
    }
}
