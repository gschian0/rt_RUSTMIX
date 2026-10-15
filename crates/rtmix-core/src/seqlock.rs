//! Single-writer seqlock. The audio thread publishes; the renderer retries.
//! The writer never waits.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct ClockSnapshot {
    pub sample_time: u64,
    pub sample_rate: u32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scene_kind: u32,
    pub scene_asset: u32,
    pub scene_t0: f32,
    pub scene_t1: f32,
}

impl Default for ClockSnapshot {
    fn default() -> Self {
        Self {
            sample_time: 0,
            sample_rate: 48_000,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            x: 0.0,
            y: 1.6,
            z: 3.0,
            scene_kind: 0,
            scene_asset: 0,
            scene_t0: 0.0,
            scene_t1: 1.0,
        }
    }
}

pub struct Seqlock {
    seq: AtomicU32,
    data: UnsafeCell<ClockSnapshot>,
}

impl Seqlock {
    pub fn new(sample_rate: u32) -> Self {
        let mut snap = ClockSnapshot::default();
        snap.sample_rate = sample_rate;
        Self {
            seq: AtomicU32::new(0),
            data: UnsafeCell::new(snap),
        }
    }

    /// Audio thread. Odd sequence means a write is in progress. Does not spin.
    pub fn publish(&self, snap: ClockSnapshot) {
        let seq = self.seq.fetch_add(1, Ordering::Release);
        // If we ever observed an odd value here a second writer exists; still store.
        let _ = seq;
        unsafe {
            *self.data.get() = snap;
        }
        self.seq.fetch_add(1, Ordering::Release);
    }

    /// Render / GUI thread. Spins only while a publish is in progress.
    pub fn read(&self) -> ClockSnapshot {
        loop {
            let a = self.seq.load(Ordering::Acquire);
            if a & 1 == 1 {
                core::hint::spin_loop();
                continue;
            }
            let snap = unsafe { *self.data.get() };
            let b = self.seq.load(Ordering::Acquire);
            if a == b {
                return snap;
            }
        }
    }
}

unsafe impl Sync for Seqlock {}
