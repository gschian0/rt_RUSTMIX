use alloc::boxed::Box;

use rtmix_core::{CoreError, Engine, Instrument, RunCtx, SetupCtx};

const MAX_GRAINS: usize = 24;

#[derive(Clone, Copy, Debug)]
pub struct GranularNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    pub grain_dur: f32,
    pub density: f32,
    pub pitch: f32,
    pub jitter: f32,
    pub pan: f32,
    pub seed: u32,
    pub table: Option<usize>,
}

#[derive(Clone, Copy)]
struct Grain {
    active: bool,
    position: f32,
    step: f32,
    age: u32,
    length: u32,
    amp: f32,
    pan: f32,
}

impl Default for Grain {
    fn default() -> Self {
        Self {
            active: false,
            position: 0.0,
            step: 1.0,
            age: 0,
            length: 1,
            amp: 0.0,
            pan: 0.5,
        }
    }
}

struct Granular {
    amp: f32,
    grain_dur: f32,
    density: f32,
    pitch: f32,
    jitter: f32,
    pan: f32,
    seed: u32,
    table_id: usize,
    next_grain: u32,
    grains: [Grain; MAX_GRAINS],
}

pub fn schedule_granular(engine: &mut Engine, note: GranularNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "GRANULAR",
        note.start,
        note.dur,
        Box::new(Granular {
            amp: super::scale_amp(note.amp),
            grain_dur: note.grain_dur.clamp(0.008, 1.5),
            density: note.density.clamp(0.1, 1200.0),
            pitch: note.pitch.clamp(0.05, 8.0),
            jitter: note.jitter.clamp(0.0, 1.0),
            pan: note.pan.clamp(0.0, 1.0),
            seed: note.seed.max(1),
            table_id: note.table.unwrap_or(0),
            next_grain: 0,
            grains: [Grain::default(); MAX_GRAINS],
        }),
    )
}

impl Granular {
    #[inline]
    fn random(&mut self) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.seed as f32) / (u32::MAX as f32)
    }

    fn spawn(&mut self, sample_rate: u32, table_len: usize) {
        let slot = self
            .grains
            .iter()
            .position(|grain| !grain.active)
            .unwrap_or_else(|| {
                self.grains
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, grain)| grain.age)
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            });
        let length = (self.grain_dur * sample_rate as f32).round().max(8.0) as u32;
        let pitch = self.pitch
            * (1.0 + (self.random() * 2.0 - 1.0) * self.jitter * 0.18);
        let pan = (self.pan + (self.random() * 2.0 - 1.0) * self.jitter * 0.35).clamp(0.0, 1.0);
        self.grains[slot] = Grain {
            active: true,
            position: self.random(),
            step: pitch / table_len.max(2) as f32,
            age: 0,
            length,
            amp: 0.55 + self.random() * 0.45,
            pan,
        };
    }
}

impl Instrument for Granular {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}

    fn configure(&mut self, _: &mut SetupCtx<'_>) {}

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let Some(table) = ctx
            .tables
            .get(self.table_id)
            .map(|table| table.data.as_slice())
            .filter(|table| table.len() >= 2)
            .or_else(|| ctx.tables.first().map(|table| table.data.as_slice()))
        else {
            return;
        };
        let table_len = table.len();
        let sample_rate = ctx.sample_rate.max(1);
        let interval = (sample_rate as f32 / self.density).max(1.0) as u32;
        let channels = ctx.buses.channels(ctx.out_bus).max(1);

        for i in 0..ctx.frame_count {
            if self.next_grain == 0 {
                self.spawn(sample_rate, table_len);
                self.next_grain = interval;
            }
            self.next_grain = self.next_grain.saturating_sub(1);

            let frame = ctx.frame_offset + i;
            let mut left = 0.0;
            let mut right = 0.0;
            for grain in &mut self.grains {
                if !grain.active {
                    continue;
                }
                let phase = grain.age as f32 / grain.length as f32;
                let window = libm::sinf(core::f32::consts::PI * phase).max(0.0);
                let position = grain.position * table_len as f32;
                let index = position as usize % table_len;
                let next = (index + 1) % table_len;
                let frac = position - index as f32;
                let value = table[index] + (table[next] - table[index]) * frac;
                let value = value * window * grain.amp * self.amp;
                let pan_l = libm::sqrtf(1.0 - grain.pan);
                let pan_r = libm::sqrtf(grain.pan);
                left += value * pan_l;
                right += value * pan_r;
                grain.position = (grain.position + grain.step).fract();
                grain.age += 1;
                if grain.age >= grain.length {
                    grain.active = false;
                }
            }
            if channels == 1 {
                ctx.buses.add(ctx.out_bus, 0, frame, (left + right) * 0.5);
            } else {
                ctx.buses.add(ctx.out_bus, 0, frame, left);
                ctx.buses.add(ctx.out_bus, 1, frame, right);
            }
        }
    }
}
