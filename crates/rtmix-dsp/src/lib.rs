//! Phase-1 instruments: wavetable, biquad filter, delay, reverb, and table builders.
//!
//! Heavy buffers are allocated in `configure`. `run` only reads and writes them.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

mod formant;
mod speech;
mod vocal;

pub use formant::{
    chant_pattern, euclid, make_harmonics, schedule_chant, schedule_formant, ChantNote, FormantNote,
};
pub use speech::{schedule_speak, text_to_phones, Phone, SpeakNote};
pub use vocal::{schedule_fof, schedule_lpc, FofNote, LpcNote};

use rtmix_core::{CoreError, Engine, Instrument, Pfield, RunCtx, SetupCtx};

pub struct WavetableNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub freq: f32,
    pub pan: f32,
    pub table: Option<usize>,
}

pub struct MixNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
}

pub struct FilterNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub cutoff: f32,
    pub q: f32,
    pub kind: FilterKind,
}

pub struct DelayNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub delay_secs: f32,
    pub feedback: f32,
}

pub struct ReverbNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub room: f32,
    pub damp: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
}

struct Wavetable {
    amp: f32,
    freq: f32,
    pan: f32,
    phase: f32,
    table_id: usize,
}

struct Mix {
    amp: f32,
}

struct Filter {
    amp: f32,
    cutoff: f32,
    q: f32,
    kind: FilterKind,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    ready: bool,
}

struct Delay {
    amp: f32,
    feedback: f32,
    delay_secs: f32,
    buf: Vec<f32>,
    idx: usize,
}

struct Comb {
    buf: Vec<f32>,
    idx: usize,
    feedback: f32,
    damp: f32,
    filter: f32,
}

struct Allpass {
    buf: Vec<f32>,
    idx: usize,
    feedback: f32,
}

struct Reverb {
    amp: f32,
    combs: Vec<Comb>,
    allpasses: Vec<Allpass>,
}

pub fn scale_amp(amp: f32) -> f32 {
    if amp.abs() > 4.0 {
        amp / 32768.0
    } else {
        amp
    }
}

pub fn make_sine(size: usize) -> Vec<f32> {
    let size = size.max(4);
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let p = (i as f32) / (size as f32);
        data.push(libm::sinf(p * core::f32::consts::TAU));
    }
    data
}

pub fn make_triangle(size: usize) -> Vec<f32> {
    let size = size.max(4);
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let p = (i as f32) / (size as f32);
        let tri = if p < 0.5 {
            -1.0 + 4.0 * p
        } else {
            3.0 - 4.0 * p
        };
        data.push(tri);
    }
    data
}

pub fn make_saw(size: usize) -> Vec<f32> {
    let size = size.max(4);
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let p = (i as f32) / (size as f32);
        data.push(2.0 * p - 1.0);
    }
    data
}

pub fn install_table(engine: &mut Engine, kind: &str, size: usize) -> Result<usize, CoreError> {
    let data = match kind {
        "sine" | "wave" => make_sine(size),
        "triangle" => make_triangle(size),
        "saw" | "sawtooth" => make_saw(size),
        _ => return Err(CoreError::new("unknown maketable kind")),
    };
    Ok(engine.add_table(data))
}

pub fn schedule_wavetable(engine: &mut Engine, note: WavetableNote) -> Result<(), CoreError> {
    let inst = Wavetable {
        amp: scale_amp(note.amp),
        freq: note.freq.max(0.0),
        pan: note.pan.clamp(0.0, 1.0),
        phase: 0.0,
        table_id: note.table.unwrap_or(0),
    };
    engine.schedule_named("WAVETABLE", note.start, note.dur, Box::new(inst))
}

pub fn schedule_mix(engine: &mut Engine, note: MixNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "MIX",
        note.start,
        note.dur,
        Box::new(Mix {
            amp: scale_amp(note.amp),
        }),
    )
}

pub fn schedule_filter(engine: &mut Engine, note: FilterNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "FILTER",
        note.start,
        note.dur,
        Box::new(Filter {
            amp: scale_amp(note.amp),
            cutoff: note.cutoff.max(1.0),
            q: note.q.max(0.05),
            kind: note.kind,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            ready: false,
        }),
    )
}

pub fn schedule_delay(engine: &mut Engine, note: DelayNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "DELAY",
        note.start,
        note.dur,
        Box::new(Delay {
            amp: scale_amp(note.amp),
            feedback: note.feedback.clamp(-0.98, 0.98),
            delay_secs: note.delay_secs,
            buf: Vec::new(),
            idx: 0,
        }),
    )
}

pub fn schedule_reverb(engine: &mut Engine, note: ReverbNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "REVERB",
        note.start,
        note.dur,
        Box::new(Reverb {
            amp: scale_amp(note.amp),
            combs: Vec::new(),
            allpasses: Vec::new(),
        }),
    )
}

pub trait EngineDsp {
    fn schedule(&mut self, note: WavetableNote) -> Result<(), CoreError>;
}

impl EngineDsp for Engine {
    fn schedule(&mut self, note: WavetableNote) -> Result<(), CoreError> {
        schedule_wavetable(self, note)
    }
}

impl Instrument for Wavetable {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {}

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let table = ctx
            .tables
            .get(self.table_id)
            .map(|t| t.data.as_slice())
            .filter(|t| t.len() >= 2)
            .or_else(|| ctx.tables.first().map(|t| t.data.as_slice()));
        let Some(table) = table else {
            return;
        };
        let len = table.len() as f32;
        let sr = ctx.sample_rate.max(1) as f32;
        let channels = ctx.buses.channels(ctx.out_bus);
        let (gain_l, gain_r) = if channels == 2 {
            let pan = self.pan.clamp(0.0, 1.0);
            (libm::sqrtf(1.0 - pan), libm::sqrtf(pan))
        } else {
            (1.0, 0.0)
        };
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let pos = self.phase * len;
            let i0 = pos as usize % table.len();
            let i1 = (i0 + 1) % table.len();
            let frac = pos - libm::floorf(pos);
            let sample = (table[i0] + (table[i1] - table[i0]) * frac) * self.amp;
            if channels == 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * gain_l);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * gain_r);
            } else {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.phase += self.freq / sr;
            if self.phase >= 1.0 {
                self.phase -= libm::floorf(self.phase);
            }
        }
        let _ = Pfield::constant(self.amp);
    }
}

impl Instrument for Mix {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {}
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let chs = ctx
            .buses
            .channels(ctx.in_bus)
            .max(1)
            .min(ctx.buses.channels(ctx.out_bus).max(1));
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            for ch in 0..chs {
                let s = ctx.buses.sample(ctx.in_bus, ch, frame) * self.amp;
                ctx.buses.add(ctx.out_bus, ch, frame, s);
            }
        }
    }
}

impl Filter {
    fn design(&mut self, sample_rate: f32) {
        let cutoff = self.cutoff.clamp(1.0, sample_rate * 0.45);
        let q = self.q.max(0.05);
        let w0 = core::f32::consts::TAU * cutoff / sample_rate;
        let cos_w = libm::cosf(w0);
        let sin_w = libm::sinf(w0);
        let alpha = sin_w / (2.0 * q);
        let (b0, b1, b2, a0, a1, a2) = match self.kind {
            FilterKind::Lowpass => {
                let b0 = (1.0 - cos_w) * 0.5;
                let b1 = 1.0 - cos_w;
                let b2 = (1.0 - cos_w) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterKind::Highpass => {
                let b0 = (1.0 + cos_w) * 0.5;
                let b1 = -(1.0 + cos_w);
                let b2 = (1.0 + cos_w) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterKind::Bandpass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
        self.ready = true;
    }
}

impl Instrument for Filter {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        self.design(ctx.sample_rate as f32);
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if !self.ready {
            self.design(ctx.sample_rate as f32);
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let x = ctx.buses.sample(ctx.in_bus, 0, frame);
            let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            ctx.buses.add(ctx.out_bus, 0, frame, y * self.amp);
        }
    }
}

impl Instrument for Delay {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        if self.buf.is_empty() {
            let n = libm::roundf((self.delay_secs.max(0.001)) * ctx.sample_rate as f32) as usize;
            let n = n.clamp(1, ctx.sample_rate as usize * 4);
            self.buf = alloc::vec![0.0; n];
        }
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if self.buf.is_empty() {
            return;
        }
        let len = self.buf.len();
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let x = ctx.buses.sample(ctx.in_bus, 0, frame);
            let y = self.buf[self.idx];
            self.buf[self.idx] = x + y * self.feedback;
            self.idx += 1;
            if self.idx >= len {
                self.idx = 0;
            }
            ctx.buses.add(ctx.out_bus, 0, frame, y * self.amp);
        }
    }
}

impl Delay {
    fn prepare(delay_secs: f32, feedback: f32, sample_rate: u32) -> Self {
        let n = libm::roundf((delay_secs.max(0.001)) * sample_rate as f32) as usize;
        let n = n.clamp(1, sample_rate as usize * 4);
        Self {
            amp: 1.0,
            feedback,
            delay_secs: delay_secs.max(0.001),
            buf: alloc::vec![0.0; n],
            idx: 0,
        }
    }
}

impl Instrument for Reverb {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        let sr = ctx.sample_rate.max(1);
        let scale = sr as f32 / 44100.0;
        let comb_len = [
            (1557.0 * scale) as usize,
            (1617.0 * scale) as usize,
            (1491.0 * scale) as usize,
            (1422.0 * scale) as usize,
        ];
        let all_len = [(225.0 * scale) as usize, (556.0 * scale) as usize];
        let room = 0.84_f32;
        let damp = 0.2_f32;
        self.combs = comb_len
            .into_iter()
            .map(|n| Comb {
                buf: alloc::vec![0.0; n.max(8)],
                idx: 0,
                feedback: room,
                damp,
                filter: 0.0,
            })
            .collect();
        self.allpasses = all_len
            .into_iter()
            .map(|n| Allpass {
                buf: alloc::vec![0.0; n.max(8)],
                idx: 0,
                feedback: 0.5,
            })
            .collect();
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if self.combs.is_empty() {
            return;
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let x = ctx.buses.sample(ctx.in_bus, 0, frame);
            let mut acc = 0.0;
            for comb in &mut self.combs {
                let delayed = comb.buf[comb.idx];
                comb.filter = delayed * (1.0 - comb.damp) + comb.filter * comb.damp;
                comb.buf[comb.idx] = x + comb.filter * comb.feedback;
                comb.idx += 1;
                if comb.idx >= comb.buf.len() {
                    comb.idx = 0;
                }
                acc += delayed;
            }
            acc /= self.combs.len() as f32;
            for ap in &mut self.allpasses {
                let delayed = ap.buf[ap.idx];
                let y = -ap.feedback * acc + delayed;
                ap.buf[ap.idx] = acc + delayed * ap.feedback;
                ap.idx += 1;
                if ap.idx >= ap.buf.len() {
                    ap.idx = 0;
                }
                acc = y;
            }
            let y = acc * self.amp;
            let chs = ctx.buses.channels(ctx.out_bus).max(1);
            ctx.buses.add(ctx.out_bus, 0, frame, y);
            if chs > 1 {
                ctx.buses.add(ctx.out_bus, 1, frame, y * 0.97);
            }
        }
    }
}

/// Build a delay whose buffer length matches `delay_secs`. Used by the interpreter.
pub fn schedule_delay_prepared(engine: &mut Engine, note: DelayNote) -> Result<(), CoreError> {
    let mut delay = Delay::prepare(note.delay_secs, note.feedback.clamp(-0.98, 0.98), engine.sample_rate());
    delay.amp = scale_amp(note.amp);
    engine.schedule_named("DELAY", note.start, note.dur, Box::new(delay))
}

/// Reverb with explicit room and damping applied at schedule time.
pub fn schedule_reverb_prepared(engine: &mut Engine, note: ReverbNote) -> Result<(), CoreError> {
    let room = (0.2 + 0.78 * note.room.clamp(0.0, 1.0)).clamp(0.05, 0.98);
    let damp = note.damp.clamp(0.0, 0.95);
    engine.schedule_named(
        "REVERB",
        note.start,
        note.dur,
        Box::new(ReverbPrepared {
            amp: scale_amp(note.amp),
            room,
            damp,
            combs: Vec::new(),
            allpasses: Vec::new(),
        }),
    )
}

struct ReverbPrepared {
    amp: f32,
    room: f32,
    damp: f32,
    combs: Vec<Comb>,
    allpasses: Vec<Allpass>,
}

impl Instrument for ReverbPrepared {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        let sr = ctx.sample_rate.max(1);
        let scale = sr as f32 / 44100.0;
        let comb_len = [
            (1116.0 * scale) as usize,
            (1188.0 * scale) as usize,
            (1277.0 * scale) as usize,
            (1356.0 * scale) as usize,
        ];
        let all_len = [(225.0 * scale) as usize, (341.0 * scale) as usize];
        let room = self.room;
        let damp = self.damp;
        self.combs = comb_len
            .into_iter()
            .map(|n| Comb {
                buf: alloc::vec![0.0; n.max(8)],
                idx: 0,
                feedback: room,
                damp,
                filter: 0.0,
            })
            .collect();
        self.allpasses = all_len
            .into_iter()
            .map(|n| Allpass {
                buf: alloc::vec![0.0; n.max(8)],
                idx: 0,
                feedback: 0.5,
            })
            .collect();
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if self.combs.is_empty() {
            return;
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let x = ctx.buses.sample(ctx.in_bus, 0, frame);
            let mut acc = 0.0;
            for comb in &mut self.combs {
                let delayed = comb.buf[comb.idx];
                comb.filter = delayed * (1.0 - comb.damp) + comb.filter * comb.damp;
                comb.buf[comb.idx] = x + comb.filter * comb.feedback;
                comb.idx += 1;
                if comb.idx >= comb.buf.len() {
                    comb.idx = 0;
                }
                acc += delayed;
            }
            acc /= self.combs.len() as f32;
            for ap in &mut self.allpasses {
                let delayed = ap.buf[ap.idx];
                let y = -ap.feedback * acc + delayed;
                ap.buf[ap.idx] = acc + delayed * ap.feedback;
                ap.idx += 1;
                if ap.idx >= ap.buf.len() {
                    ap.idx = 0;
                }
                acc = y;
            }
            let y = acc * self.amp;
            ctx.buses.add(ctx.out_bus, 0, frame, y);
            if ctx.buses.channels(ctx.out_bus) > 1 {
                ctx.buses.add(ctx.out_bus, 1, frame, y);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_attenuates_high_frequency() {
        let mut engine = Engine::new(48_000, 1, 256);
        engine
            .bus_config("WAVETABLE", "aux")
            .unwrap();
        engine.bus_config("FILTER", "out").unwrap();
        engine.set_input_bus("FILTER", "aux").unwrap();
        schedule_wavetable(
            &mut engine,
            WavetableNote {
                start: 0.0,
                dur: 0.05,
                amp: 0.5,
                freq: 8_000.0,
                pan: 0.5,
                table: None,
            },
        )
        .unwrap();
        schedule_filter(
            &mut engine,
            FilterNote {
                start: 0.0,
                dur: 0.05,
                amp: 1.0,
                cutoff: 400.0,
                q: 0.707,
                kind: FilterKind::Lowpass,
            },
        )
        .unwrap();
        engine.process_seconds(0.05);
        let dry_energy = {
            let mut e = 0.0f32;
            let mut probe = Engine::new(48_000, 1, 256);
            probe.bus_config("WAVETABLE", "out").unwrap();
            schedule_wavetable(
                &mut probe,
                WavetableNote {
                    start: 0.0,
                    dur: 0.05,
                    amp: 0.5,
                    freq: 8_000.0,
                    pan: 0.5,
                    table: None,
                },
            )
            .unwrap();
            probe.process_seconds(0.05);
            for s in probe.bus_plane(probe.out_bus(), 0) {
                e += s * s;
            }
            // The last block only. Render the whole thing into an accumulator instead.
            e
        };
        let _ = dry_energy;
        let filtered = 0.0f32;
        // Re-render and sum every block.
        let mut engine = Engine::new(48_000, 1, 128);
        engine.bus_config("WAVETABLE", "aux").unwrap();
        engine.bus_config("FILTER", "out").unwrap();
        engine.set_input_bus("FILTER", "aux").unwrap();
        schedule_wavetable(
            &mut engine,
            WavetableNote {
                start: 0.0,
                dur: 0.08,
                amp: 0.8,
                freq: 6_000.0,
                pan: 0.5,
                table: None,
            },
        )
        .unwrap();
        schedule_filter(
            &mut engine,
            FilterNote {
                start: 0.0,
                dur: 0.08,
                amp: 1.0,
                cutoff: 300.0,
                q: 0.707,
                kind: FilterKind::Lowpass,
            },
        )
        .unwrap();
        let mut low = 0.0f32;
        let frames = (0.08 * 48000.0) as usize;
        let mut seen = 0usize;
        while seen < frames {
            engine.process_block();
            for s in engine.bus_plane(engine.out_bus(), 0) {
                low += s * s;
            }
            seen += engine.block_size();
        }
        let mut engine = Engine::new(48_000, 1, 128);
        engine.bus_config("WAVETABLE", "out").unwrap();
        schedule_wavetable(
            &mut engine,
            WavetableNote {
                start: 0.0,
                dur: 0.08,
                amp: 0.8,
                freq: 6_000.0,
                pan: 0.5,
                table: None,
            },
        )
        .unwrap();
        let mut high = 0.0f32;
        let mut seen = 0usize;
        while seen < frames {
            engine.process_block();
            for s in engine.bus_plane(engine.out_bus(), 0) {
                high += s * s;
            }
            seen += engine.block_size();
        }
        assert!(high > low * 4.0, "low {low} dry {high}");
        let _ = filtered;
    }
}
