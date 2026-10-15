//! Formant instruments driven by harmonic wavetables.
//!
//! `make_harmonics` builds a single-cycle table from a partial rule.
//! `FORMANT` holds a vowel. `CHANT` steps an Euclidean rhythm, a just-intonation
//! pitch cycle, and a vowel cycle. Filter state and the wavetable phase stay
//! across hits so rests do not click. Buffers are allocated in `configure`.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use rtmix_core::{
    CoreError, Engine, Instrument, RunCtx, SetupCtx, SYNTH_CHANT, SYNTH_FORMANT,
};

use crate::scale_amp;

#[derive(Clone, Copy)]
struct Vowel {
    freq: [f32; 3],
    bw: [f32; 3],
}

fn vowel_from_char(c: char) -> Vowel {
    match c.to_ascii_lowercase() {
        'e' => Vowel { freq: [530.0, 1840.0, 2480.0], bw: [60.0, 90.0, 120.0] },
        'i' => Vowel { freq: [270.0, 2290.0, 3010.0], bw: [60.0, 100.0, 120.0] },
        'o' => Vowel { freq: [570.0, 840.0, 2410.0], bw: [80.0, 80.0, 120.0] },
        'u' => Vowel { freq: [300.0, 870.0, 2240.0], bw: [60.0, 80.0, 110.0] },
        _ => Vowel { freq: [730.0, 1090.0, 2440.0], bw: [80.0, 90.0, 120.0] },
    }
}

/// Sum a harmonic series into one wavetable cycle and normalize the peak to 1.
///
/// `rule` is `all` (every partial), `odd` (square-like), or `even` (hollow, fundamental kept).
/// Partial `k` is weighted by `1/k^rolloff`.
pub fn make_harmonics(size: usize, partials: usize, rolloff: f32, rule: &str) -> Vec<f32> {
    let size = size.max(16);
    let partials = partials.clamp(1, 64);
    let rolloff = if rolloff <= 0.0 { 1.0 } else { rolloff };
    let mut data = vec![0.0f32; size];
    for k in 1..=partials {
        let keep = match rule {
            "odd" | "square" => k % 2 == 1,
            "even" | "hollow" => k == 1 || k % 2 == 0,
            _ => true,
        };
        if !keep {
            continue;
        }
        let amp = 1.0 / libm::powf(k as f32, rolloff);
        let harm = k as f32;
        for (i, sample) in data.iter_mut().enumerate() {
            let p = (i as f32) / (size as f32);
            *sample += amp * libm::sinf(p * core::f32::consts::TAU * harm);
        }
    }
    let peak = data.iter().fold(0.0f32, |m, s| m.max(s.abs())).max(1.0e-8);
    for sample in &mut data {
        *sample /= peak;
    }
    data
}

/// Bresenham Euclidean rhythm. `pulses` hits spread across `steps` slots.
pub fn euclid(pulses: usize, steps: usize) -> Vec<bool> {
    let steps = steps.max(1);
    let pulses = pulses.min(steps);
    let mut pattern = vec![false; steps];
    if pulses == 0 {
        return pattern;
    }
    let mut bucket = 0usize;
    for (i, slot) in pattern.iter_mut().enumerate() {
        let _ = i;
        bucket += pulses;
        if bucket >= steps {
            bucket -= steps;
            *slot = true;
        }
    }
    pattern
}

const JUST: [f32; 5] = [1.0, 5.0 / 4.0, 3.0 / 2.0, 7.0 / 4.0, 2.0];

struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn set_bandpass(&mut self, sample_rate: f32, freq: f32, bw: f32) {
        let freq = freq.clamp(40.0, sample_rate * 0.45);
        let q = (freq / bw.max(20.0)).clamp(0.5, 30.0);
        let w0 = core::f32::consts::TAU * freq / sample_rate;
        let cos_w = libm::cosf(w0);
        let sin_w = libm::sinf(w0);
        let alpha = sin_w / (2.0 * q);
        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.b1 = 0.0;
        self.b2 = -alpha / a0;
        self.a1 = (-2.0 * cos_w) / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

struct FormantVoice {
    amp: f32,
    freq: f32,
    pan: f32,
    phase: f32,
    env: f32,
    table_id: usize,
    filters: [Biquad; 3],
    vowel: Vowel,
    designed: bool,
    sample_rate: f32,
}

impl FormantVoice {
    fn design(&mut self, sample_rate: f32) {
        for i in 0..3 {
            self.filters[i].set_bandpass(sample_rate, self.vowel.freq[i], self.vowel.bw[i]);
        }
        self.designed = true;
        self.sample_rate = sample_rate;
    }

    fn render(&mut self, ctx: &mut RunCtx<'_>, release: bool, synth: usize) {
        if !self.designed {
            self.design(ctx.sample_rate as f32);
        }
        let sr = ctx.sample_rate.max(1) as f32;
        let channels = ctx.buses.channels(ctx.out_bus);
        let (gain_l, gain_r) = if channels >= 2 {
            let pan = self.pan.clamp(0.0, 1.0);
            (libm::sqrtf(1.0 - pan), libm::sqrtf(pan))
        } else {
            (1.0, 0.0)
        };
        let attack = 1.0 - libm::expf(-1.0 / (sr * 0.012));
        let rel = 1.0 - libm::expf(-1.0 / (sr * 0.04));
        let coef = if release { rel } else { attack };
        let target = if release { 0.0 } else { 1.0 };
        let (gain, tone, _) = ctx.synth_controls.load(synth);
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            self.env += (target - self.env) * coef;
            let raw = read_table(ctx, self.table_id, self.phase);
            let mut shaped = 0.0;
            for (index, filter) in self.filters.iter_mut().enumerate() {
                let color = 1.0 + (tone - 1.0) * index as f32 * 0.45;
                shaped += filter.tick(raw) * color.max(0.0);
            }
            let sample = shaped * (1.0 / 3.0) * self.env * self.amp * gain;
            if channels >= 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * gain_l);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * gain_r);
            } else if channels >= 1 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.phase += self.freq / sr;
            if self.phase >= 1.0 {
                self.phase -= libm::floorf(self.phase);
            }
        }
    }
}

pub struct FormantNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub freq: f32,
    pub pan: f32,
    pub vowel: char,
    pub table: Option<usize>,
}

pub struct ChantNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub root: f32,
    pub pan: f32,
    pub rate: f32,
    pub pulses: usize,
    pub steps: usize,
    pub vowels: Vec<char>,
    pub table: Option<usize>,
}

struct FormantInst {
    voice: FormantVoice,
}

struct ChantInst {
    voice: FormantVoice,
    root: f32,
    rate: f32,
    pulses: usize,
    steps: usize,
    pattern: Vec<bool>,
    vowels: Vec<char>,
    step_pos: f32,
    step_index: usize,
    hit_index: usize,
    gate: bool,
}

fn read_table(ctx: &RunCtx<'_>, id: usize, phase: f32) -> f32 {
    let table = ctx
        .tables
        .get(id)
        .map(|t| t.data.as_slice())
        .filter(|t| t.len() >= 2)
        .or_else(|| ctx.tables.first().map(|t| t.data.as_slice()))
        .filter(|t| t.len() >= 2);
    let Some(table) = table else {
        return 0.0;
    };
    let pos = phase * table.len() as f32;
    let i0 = (pos as usize) % table.len();
    let i1 = (i0 + 1) % table.len();
    let frac = pos - libm::floorf(pos);
    table[i0] + (table[i1] - table[i0]) * frac
}

pub fn schedule_formant(engine: &mut Engine, note: FormantNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "FORMANT",
        note.start,
        note.dur,
        Box::new(FormantInst {
            voice: FormantVoice {
                amp: scale_amp(note.amp),
                freq: note.freq.max(20.0),
                pan: note.pan.clamp(0.0, 1.0),
                phase: 0.0,
                env: 0.0,
                table_id: note.table.unwrap_or(0),
                filters: [Biquad::new(), Biquad::new(), Biquad::new()],
                vowel: vowel_from_char(note.vowel),
                designed: false,
                sample_rate: 0.0,
            },
        }),
    )
}

pub fn schedule_chant(engine: &mut Engine, note: ChantNote) -> Result<(), CoreError> {
    let vowels = if note.vowels.is_empty() {
        vec!['a', 'e', 'i', 'o', 'u']
    } else {
        note.vowels
    };
    engine.schedule_named(
        "CHANT",
        note.start,
        note.dur,
        Box::new(ChantInst {
            voice: FormantVoice {
                amp: scale_amp(note.amp),
                freq: note.root.max(20.0),
                pan: note.pan.clamp(0.0, 1.0),
                phase: 0.0,
                env: 0.0,
                table_id: note.table.unwrap_or(0),
                filters: [Biquad::new(), Biquad::new(), Biquad::new()],
                vowel: vowel_from_char(vowels[0]),
                designed: false,
                sample_rate: 0.0,
            },
            root: note.root.max(20.0),
            rate: note.rate.max(0.25),
            pulses: note.pulses.max(1),
            steps: note.steps.max(1),
            pattern: Vec::new(),
            vowels,
            step_pos: 0.0,
            step_index: 0,
            hit_index: 0,
            gate: false,
        }),
    )
}

impl Instrument for FormantInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        self.voice.design(ctx.sample_rate as f32);
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let release = ctx.note_end.saturating_sub(ctx.block_start) < (ctx.sample_rate as u64 / 20);
        self.voice.render(ctx, release, SYNTH_FORMANT);
    }
}

impl Instrument for ChantInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        self.pattern = euclid(self.pulses, self.steps);
        self.voice.design(ctx.sample_rate as f32);
        self.open_step();
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let sr = ctx.sample_rate.max(1) as f32;
        let step = self.rate / sr;
        let release_window = ctx.sample_rate as u64 / 20;
        let note_release = ctx.note_end.saturating_sub(ctx.block_start) < release_window;
        let channels = ctx.buses.channels(ctx.out_bus);
        let (gain_l, gain_r) = if channels >= 2 {
            let pan = self.voice.pan.clamp(0.0, 1.0);
            (libm::sqrtf(1.0 - pan), libm::sqrtf(pan))
        } else {
            (1.0, 0.0)
        };
        let attack = 1.0 - libm::expf(-1.0 / (sr * 0.004));
        let rel = 1.0 - libm::expf(-1.0 / (sr * 0.045));
        let (gain, tone, motion) = ctx.synth_controls.load(SYNTH_CHANT);
        if !self.voice.designed {
            self.voice.design(sr);
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            self.step_pos += step * motion;
            if self.step_pos >= 1.0 {
                self.step_pos -= libm::floorf(self.step_pos);
                self.step_index = (self.step_index + 1) % self.pattern.len().max(1);
                self.open_step();
            }
            let target = if note_release || !self.gate { 0.0 } else { 1.0 };
            let coef = if target > self.voice.env { attack } else { rel };
            self.voice.env += (target - self.voice.env) * coef;
            let raw = read_table(ctx, self.voice.table_id, self.voice.phase);
            let mut shaped = 0.0;
            for (index, filter) in self.voice.filters.iter_mut().enumerate() {
                let color = 1.0 + (tone - 1.0) * index as f32 * 0.45;
                shaped += filter.tick(raw) * color.max(0.0);
            }
            let sample = shaped * (1.0 / 3.0) * self.voice.env * self.voice.amp * gain;
            if channels >= 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * gain_l);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * gain_r);
            } else if channels >= 1 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.voice.phase += self.voice.freq / sr;
            if self.voice.phase >= 1.0 {
                self.voice.phase -= libm::floorf(self.voice.phase);
            }
        }
    }
}

impl ChantInst {
    fn open_step(&mut self) {
        if self.pattern.is_empty() {
            self.gate = false;
            return;
        }
        let hit = self.pattern[self.step_index % self.pattern.len()];
        self.gate = hit;
        if hit {
            let ratio = JUST[self.hit_index % JUST.len()];
            self.voice.freq = self.root * ratio;
            let ch = self.vowels[self.hit_index % self.vowels.len()];
            let next = vowel_from_char(ch);
            if next.freq != self.voice.vowel.freq {
                self.voice.vowel = next;
                if self.voice.sample_rate > 0.0 {
                    let sr = self.voice.sample_rate;
                    self.voice.design(sr);
                } else {
                    self.voice.designed = false;
                }
            }
            self.hit_index = self.hit_index.wrapping_add(1);
        }
    }
}

pub fn chant_pattern(pulses: usize, steps: usize) -> Vec<bool> {
    euclid(pulses, steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_rule_is_half_wave_symmetric() {
        let table = make_harmonics(256, 12, 1.0, "odd");
        let half = table.len() / 2;
        for i in 0..half {
            let sum = table[i] + table[i + half];
            assert!(sum.abs() < 1.0e-4, "partial {i} broke odd symmetry: {sum}");
        }
    }

    #[test]
    fn euclid_five_in_eight_has_five_hits() {
        let pattern = euclid(5, 8);
        assert_eq!(pattern.iter().filter(|hit| **hit).count(), 5);
        assert_eq!(pattern.len(), 8);
    }

    #[test]
    fn formant_vowel_stays_under_full_scale() {
        let mut engine = Engine::new(48_000, 2, 128);
        engine.bus_config("FORMANT", "out").unwrap();
        let table = engine.add_table(make_harmonics(1024, 12, 1.0, "odd"));
        schedule_formant(
            &mut engine,
            FormantNote {
                start: 0.0,
                dur: 0.2,
                amp: 12_000.0,
                freq: 196.0,
                pan: 0.4,
                vowel: 'a',
                table: Some(table),
            },
        )
        .unwrap();
        let mut energy = 0.0f32;
        let mut peak = 0.0f32;
        engine.process_seconds(0.2);
        for s in engine.bus_plane(engine.out_bus(), 0) {
            energy += s * s;
            peak = peak.max(s.abs());
        }
        assert!(energy > 1.0e-4, "silent formant");
        assert!(peak < 1.05, "clipped at {peak}");
    }
}
