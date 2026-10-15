//! Listener pose and scene cue inlets. Writes are atomic; the callback never locks.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::instrument::{Instrument, RunCtx, SetupCtx};

/// Window the GUI analyses. Power of two so the FFT stays radix-2.
pub const SCOPE_LEN: usize = 2048;

/// A single-producer ring of recent output samples. The audio callback only
/// ever does relaxed stores, so it never blocks the GUI reader.
pub struct AudioScope {
    samples: [AtomicU32; SCOPE_LEN],
    write: AtomicUsize,
}

impl AudioScope {
    pub fn new() -> Self {
        Self {
            samples: core::array::from_fn(|_| AtomicU32::new(0)),
            write: AtomicUsize::new(0),
        }
    }

    /// Called from the audio callback once per output frame.
    pub fn push(&self, value: f32) {
        let index = self.write.load(Ordering::Relaxed);
        self.samples[index & (SCOPE_LEN - 1)].store(value.to_bits(), Ordering::Relaxed);
        self.write.store(index.wrapping_add(1), Ordering::Relaxed);
    }

    /// Copy the window oldest-sample-first for analysis.
    pub fn read(&self, out: &mut [f32]) {
        let end = self.write.load(Ordering::Relaxed);
        let len = out.len().min(SCOPE_LEN);
        for (offset, slot) in out.iter_mut().take(len).enumerate() {
            let index = end.wrapping_sub(len).wrapping_add(offset);
            *slot = f32::from_bits(self.samples[index & (SCOPE_LEN - 1)].load(Ordering::Relaxed));
        }
    }
}

impl Default for AudioScope {
    fn default() -> Self {
        Self::new()
    }
}

pub const SYNTH_FORMANT: usize = 0;
pub const SYNTH_CHANT: usize = 1;
pub const SYNTH_FOF: usize = 2;
pub const SYNTH_LPC: usize = 3;
pub const SYNTH_SPEAK: usize = 4;
pub const SYNTH_COUNT: usize = 5;

fn pack(v: f32) -> u32 {
    v.to_bits()
}

fn unpack(v: u32) -> f32 {
    f32::from_bits(v)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub struct PoseInlet {
    yaw: AtomicU32,
    pitch: AtomicU32,
    roll: AtomicU32,
    x: AtomicU32,
    y: AtomicU32,
    z: AtomicU32,
}

impl PoseInlet {
    pub fn new() -> Self {
        Self {
            yaw: AtomicU32::new(0),
            pitch: AtomicU32::new(0),
            roll: AtomicU32::new(0),
            x: AtomicU32::new(0.0f32.to_bits()),
            y: AtomicU32::new(1.6f32.to_bits()),
            z: AtomicU32::new(3.0f32.to_bits()),
        }
    }

    pub fn set_angles(&self, yaw: f32, pitch: f32, roll: f32) {
        self.yaw.store(pack(yaw), Ordering::Relaxed);
        self.pitch.store(pack(pitch), Ordering::Relaxed);
        self.roll.store(pack(roll), Ordering::Relaxed);
    }

    pub fn set_position(&self, x: f32, y: f32, z: f32) {
        self.x.store(pack(x), Ordering::Relaxed);
        self.y.store(pack(y), Ordering::Relaxed);
        self.z.store(pack(z), Ordering::Relaxed);
    }

    pub fn set_pose(&self, pose: Pose) {
        self.set_angles(pose.yaw, pose.pitch, pose.roll);
        self.set_position(pose.x, pose.y, pose.z);
    }

    pub fn load(&self) -> Pose {
        Pose {
            yaw: unpack(self.yaw.load(Ordering::Relaxed)),
            pitch: unpack(self.pitch.load(Ordering::Relaxed)),
            roll: unpack(self.roll.load(Ordering::Relaxed)),
            x: unpack(self.x.load(Ordering::Relaxed)),
            y: unpack(self.y.load(Ordering::Relaxed)),
            z: unpack(self.z.load(Ordering::Relaxed)),
        }
    }
}

impl Default for PoseInlet {
    fn default() -> Self {
        Self::new()
    }
}

/// Active splat cue. Kind: 0 none, 1 SPLAT, 2 SPLAT4D.
pub struct SceneInlet {
    kind: AtomicU32,
    asset: AtomicU32,
    t0: AtomicU32,
    t1: AtomicU32,
}

impl SceneInlet {
    pub fn new() -> Self {
        Self {
            kind: AtomicU32::new(0),
            asset: AtomicU32::new(0),
            t0: AtomicU32::new(0.0f32.to_bits()),
            t1: AtomicU32::new(1.0f32.to_bits()),
        }
    }

    pub fn set(&self, kind: u32, asset: u32, t0: f32, t1: f32) {
        self.kind.store(kind, Ordering::Relaxed);
        self.asset.store(asset, Ordering::Relaxed);
        self.t0.store(pack(t0), Ordering::Relaxed);
        self.t1.store(pack(t1), Ordering::Relaxed);
    }

    pub fn load(&self) -> (u32, u32, f32, f32) {
        (
            self.kind.load(Ordering::Relaxed),
            self.asset.load(Ordering::Relaxed),
            unpack(self.t0.load(Ordering::Relaxed)),
            unpack(self.t1.load(Ordering::Relaxed)),
        )
    }
}

impl Default for SceneInlet {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SynthControl {
    gain: AtomicU32,
    tone: AtomicU32,
    motion: AtomicU32,
}

impl SynthControl {
    fn new() -> Self {
        Self {
            gain: AtomicU32::new(1.0f32.to_bits()),
            tone: AtomicU32::new(1.0f32.to_bits()),
            motion: AtomicU32::new(1.0f32.to_bits()),
        }
    }

    pub fn set(&self, gain: f32, tone: f32, motion: f32) {
        self.gain.store(pack(gain), Ordering::Relaxed);
        self.tone.store(pack(tone), Ordering::Relaxed);
        self.motion.store(pack(motion), Ordering::Relaxed);
    }

    pub fn load(&self) -> (f32, f32, f32) {
        (
            unpack(self.gain.load(Ordering::Relaxed)),
            unpack(self.tone.load(Ordering::Relaxed)),
            unpack(self.motion.load(Ordering::Relaxed)),
        )
    }
}

pub struct SynthControls {
    slots: [SynthControl; SYNTH_COUNT],
}

impl SynthControls {
    pub fn new() -> Self {
        Self {
            slots: core::array::from_fn(|_| SynthControl::new()),
        }
    }

    pub fn set(&self, synth: usize, gain: f32, tone: f32, motion: f32) {
        if let Some(slot) = self.slots.get(synth) {
            slot.set(gain, tone, motion);
        }
    }

    pub fn load(&self, synth: usize) -> (f32, f32, f32) {
        self.slots.get(synth).map(SynthControl::load).unwrap_or((1.0, 1.0, 1.0))
    }
}

impl Default for SynthControls {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared with the control thread and the renderer. The audio thread is the
/// only seqlock writer and the only queue consumer.
pub struct Shared {
    pub pose: PoseInlet,
    pub scene: SceneInlet,
    pub synth_controls: SynthControls,
    pub queue: crate::queue::ControlQueue,
    pub snapshot: crate::seqlock::Seqlock,
    pub out_peak: [AtomicU32; 8],
    pub hoa_peak: [AtomicU32; 16],
    pub scope: AudioScope,
}

impl Shared {
    pub fn new(sample_rate: u32, queue_cap: usize) -> Self {
        Self {
            pose: PoseInlet::new(),
            scene: SceneInlet::new(),
            synth_controls: SynthControls::new(),
            queue: crate::queue::ControlQueue::with_capacity(queue_cap),
            snapshot: crate::seqlock::Seqlock::new(sample_rate),
            out_peak: core::array::from_fn(|_| AtomicU32::new(0)),
            hoa_peak: core::array::from_fn(|_| AtomicU32::new(0)),
            scope: AudioScope::new(),
        }
    }

    pub fn set_listener(&self, yaw: f32, pitch: f32, roll: f32) {
        self.pose.set_angles(yaw, pitch, roll);
    }

    pub fn listener(&self) -> (f32, f32, f32) {
        let p = self.pose.load();
        (p.yaw, p.pitch, p.roll)
    }
}

/// Score command that publishes a scene cue or a camera pose. No audio I/O.
pub struct SceneNote {
    pub kind: u32,
    pub asset: u32,
    pub t0: f32,
    pub t1: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

impl Instrument for SceneNote {
    fn init(&mut self, _ctx: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _ctx: &mut SetupCtx<'_>) {}

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if self.kind == 3 {
            ctx.pose.set_position(self.x, self.y, self.z);
            ctx.pose.set_angles(self.yaw, self.pitch, self.roll);
        } else if self.kind != 0 {
            ctx.scene.set(self.kind, self.asset, self.t0, self.t1);
        }
    }
}
