//! Experimental vocal synthesis: FOF grains and randomized LPC-style resonators.

use alloc::boxed::Box;

use rtmix_core::{
    CoreError, Engine, Instrument, RunCtx, SetupCtx, SYNTH_FOF, SYNTH_LPC,
};

use crate::scale_amp;

pub struct FofNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub fundamental: f32,
    pub formant: f32,
    pub bandwidth: f32,
    pub jitter: f32,
    pub pan: f32,
    pub seed: u32,
}

pub struct LpcNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub root: f32,
    pub pan: f32,
    pub seed: u32,
}

struct Fof {
    amp: f32,
    fundamental: f32,
    formant: f32,
    bandwidth: f32,
    jitter: f32,
    pan: f32,
    phase: f32,
    mod_phase: f32,
    rng: u32,
    wobble: f32,
}

#[derive(Clone, Copy)]
struct Resonator {
    b0: f32,
    a1: f32,
    a2: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn new() -> Self {
        Self {
            b0: 1.0,
            a1: 0.0,
            a2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn design(&mut self, sample_rate: f32, freq: f32, bandwidth: f32) {
        let freq = freq.clamp(50.0, sample_rate * 0.45);
        let radius = libm::expf(-core::f32::consts::PI * bandwidth.max(20.0) / sample_rate);
        let angle = core::f32::consts::TAU * freq / sample_rate;
        self.b0 = 1.0 - radius;
        self.a1 = -2.0 * radius * libm::cosf(angle);
        self.a2 = radius * radius;
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

struct Lpc {
    amp: f32,
    root: f32,
    pan: f32,
    phase: f32,
    mod_phase: f32,
    rng: u32,
    resonators: [Resonator; 4],
    drift: [f32; 4],
    segment_left: u32,
    active_segment: bool,
    voiced_segment: bool,
    gate: f32,
    hp_x1: f32,
    hp_y1: f32,
    lp_y1: f32,
}

fn random01(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / 16_777_216.0
}

fn stereo_gains(pan: f32, channels: usize) -> (f32, f32) {
    if channels >= 2 {
        let pan = pan.clamp(0.0, 1.0);
        (libm::sqrtf(1.0 - pan), libm::sqrtf(pan))
    } else {
        (1.0, 0.0)
    }
}

pub fn schedule_fof(engine: &mut Engine, note: FofNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "FOF",
        note.start,
        note.dur,
        Box::new(Fof {
            amp: scale_amp(note.amp),
            fundamental: note.fundamental.max(20.0),
            formant: note.formant.max(80.0),
            bandwidth: note.bandwidth.max(20.0),
            jitter: note.jitter.clamp(0.0, 1.0),
            pan: note.pan.clamp(0.0, 1.0),
            phase: 0.0,
            mod_phase: 0.0,
            rng: note.seed.max(1),
            wobble: 0.0,
        }),
    )
}

pub fn schedule_lpc(engine: &mut Engine, note: LpcNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "LPC",
        note.start,
        note.dur,
        Box::new(Lpc {
            amp: scale_amp(note.amp),
            root: note.root.max(30.0),
            pan: note.pan.clamp(0.0, 1.0),
            phase: 0.0,
            mod_phase: 0.0,
            rng: note.seed.max(1),
            resonators: [Resonator::new(); 4],
            drift: [0.0; 4],
            segment_left: 0,
            active_segment: false,
            voiced_segment: true,
            gate: 0.0,
            hp_x1: 0.0,
            hp_y1: 0.0,
            lp_y1: 0.0,
        }),
    )
}

impl Instrument for Fof {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}

    fn configure(&mut self, _: &mut SetupCtx<'_>) {}

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let sr = ctx.sample_rate.max(1) as f32;
        let (left, right) = stereo_gains(self.pan, ctx.buses.channels(ctx.out_bus));
        let (gain, tone, motion) = ctx.synth_controls.load(SYNTH_FOF);
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let jitter = (random01(&mut self.rng) - 0.5) * self.jitter;
            self.wobble = self.wobble * 0.997 + jitter * 0.003;
            let slow = libm::sinf(self.mod_phase * core::f32::consts::TAU);
            let freq = (self.fundamental
                * (1.0 + self.wobble * 0.08 * motion + slow * 0.035 * motion))
                .max(20.0);
            let width = (freq / (self.bandwidth * 8.0)).clamp(0.015, 0.32);
            let distance = self.phase.min(1.0 - self.phase);
            let grain = libm::expf(-0.5 * (distance / width).powi(2));
            let swept_formant = self.formant * tone * (1.0 + slow * 0.18 * motion);
            let carrier = self.phase * core::f32::consts::TAU * swept_formant / freq;
            let shimmer = libm::sinf(carrier)
                + 0.65 * libm::sinf(carrier * 1.51 + self.wobble * 9.0)
                + 0.35 * libm::sinf(carrier * 2.37 - self.wobble * 13.0);
            let sample = grain * shimmer * self.amp * gain * 0.72;
            if ctx.buses.channels(ctx.out_bus) >= 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * left);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * right);
            } else {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.phase += freq / sr;
            self.phase -= libm::floorf(self.phase);
            self.mod_phase += (0.07 + self.jitter * 0.11) * motion / sr;
            self.mod_phase -= libm::floorf(self.mod_phase);
        }
    }
}

impl Instrument for Lpc {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}

    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        let sr = ctx.sample_rate.max(1) as f32;
        retune_lpc_formants(&mut self.resonators, sr, self.rng);
    }

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let sr = ctx.sample_rate.max(1) as f32;
        let (left, right) = stereo_gains(self.pan, ctx.buses.channels(ctx.out_bus));
        let (gain, tone, motion) = ctx.synth_controls.load(SYNTH_LPC);
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            if self.segment_left == 0 {
                self.active_segment = !self.active_segment;
                if self.active_segment {
                    self.voiced_segment = random01(&mut self.rng) > 0.28;
                    retune_lpc_formants(&mut self.resonators, sr, self.rng);
                    self.segment_left =
                        (sr * (0.055 + random01(&mut self.rng) * 0.17)) as u32;
                } else {
                    self.segment_left =
                        (sr * (0.018 + random01(&mut self.rng) * 0.11)) as u32;
                }
            }
            self.segment_left = self.segment_left.saturating_sub(1);
            let gate_target = if self.active_segment { 1.0 } else { 0.0 };
            let gate_rate = if gate_target > self.gate { 0.018 } else { 0.009 };
            self.gate += (gate_target - self.gate) * gate_rate;
            let slow = libm::sinf(self.mod_phase * core::f32::consts::TAU);
            let mod_root = self.root * (1.0 + slow * 0.035 * motion);
            let impulse = if self.voiced_segment && self.phase < 1.0 / sr {
                2.2
            } else {
                0.0
            };
            let noise_level = if self.voiced_segment { 0.035 } else { 0.42 };
            let breath = (random01(&mut self.rng) - 0.5) * noise_level;
            let mut sample = 0.0;
            for (index, resonator) in self.resonators.iter_mut().enumerate() {
                self.drift[index] = self.drift[index] * 0.999 + (random01(&mut self.rng) - 0.5) * 0.001;
                let brightness = 1.0 + (tone - 1.0) * index as f32 * 0.35;
                let spectral_motion = 1.0
                    + 0.25
                        * motion
                        * libm::sinf(
                            self.mod_phase * core::f32::consts::TAU * (index as f32 + 1.0),
                        );
                sample += resonator.tick(impulse + breath)
                    * spectral_motion
                    * brightness.max(0.0)
                    * (0.78 / (index as f32 + 1.0));
            }
            sample *= self.amp * gain * self.gate * 1.55;
            let hp_alpha = 0.962;
            let highpassed = hp_alpha * (self.hp_y1 + sample - self.hp_x1);
            self.hp_x1 = sample;
            self.hp_y1 = highpassed;
            let lp_alpha = 1.0 - libm::expf(-core::f32::consts::TAU * 3400.0 / sr);
            self.lp_y1 += lp_alpha * (highpassed - self.lp_y1);
            sample = libm::tanhf(self.lp_y1 * 2.4) * 0.72;
            sample = libm::roundf(sample * 48.0) / 48.0;
            if ctx.buses.channels(ctx.out_bus) >= 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * left);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * right);
            } else {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.phase += 1.0 / sr;
            if self.phase >= 1.0 / mod_root {
                self.phase -= 1.0 / mod_root;
            }
            self.mod_phase += 0.19 * motion / sr;
            self.mod_phase -= libm::floorf(self.mod_phase);
        }
    }
}

fn retune_lpc_formants(resonators: &mut [Resonator; 4], sample_rate: f32, seed: u32) {
    let formants = match seed % 5 {
        0 => [730.0, 1090.0, 2440.0, 3200.0],
        1 => [300.0, 870.0, 2240.0, 3000.0],
        2 => [570.0, 840.0, 2410.0, 3300.0],
        3 => [530.0, 1840.0, 2480.0, 3400.0],
        _ => [270.0, 2290.0, 3010.0, 3550.0],
    };
    for (index, resonator) in resonators.iter_mut().enumerate() {
        resonator.design(
            sample_rate,
            formants[index],
            85.0 + index as f32 * 38.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        #[test]
        fn resonator_rings_near_its_center_frequency() {
            let sample_rate = 48_000.0;
            let center = 730.0;
            let mut resonator = Resonator::new();
            resonator.design(sample_rate, center, 70.0);
            let mut previous = resonator.tick(1.0);
            let mut positive_crossings = 0;
            let samples = 4_800;
            for _ in 1..samples {
                let current = resonator.tick(0.0);
                if previous <= 0.0 && current > 0.0 {
                    positive_crossings += 1;
                }
                previous = current;
            }
            let measured = positive_crossings as f32 * sample_rate / samples as f32;
            assert!((measured - center).abs() < 20.0, "measured {measured} Hz");
        }

        #[test]
        fn fof_output_repeats_at_the_fundamental() {
            let mut engine = Engine::new(48_000, 2, 256);
            schedule_fof(
                &mut engine,
                FofNote {
                    start: 0.0,
                    dur: 0.2,
                    amp: 0.25,
                    fundamental: 100.0,
                    formant: 700.0,
                    bandwidth: 80.0,
                    jitter: 0.0,
                    pan: 0.5,
                    seed: 1,
                },
            )
            .unwrap();
            let mut rendered = Vec::new();
            for _ in 0..38 {
                engine.process_block();
                rendered.extend_from_slice(engine.bus_plane(engine.out_bus(), 0));
            }
            let lag = 480;
            let start = 960;
            let end = rendered.len() - lag;
            let mut correlation = 0.0;
            let mut energy_a = 0.0;
            let mut energy_b = 0.0;
            for i in start..end {
                correlation += rendered[i] * rendered[i + lag];
                energy_a += rendered[i] * rendered[i];
                energy_b += rendered[i + lag] * rendered[i + lag];
            }
            let normalized = correlation / (energy_a * energy_b).sqrt().max(1.0e-12);
            assert!(normalized > 0.85, "normalized correlation {normalized}");
    }
}
