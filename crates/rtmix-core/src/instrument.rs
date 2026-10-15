//! Instrument lifecycle. `configure` / `init` run on the control thread.
//! `run` fills only the current block and must not allocate.

use alloc::vec::Vec;

use crate::bus::BusBank;
use crate::scene::{PoseInlet, SceneInlet, SynthControls};

#[derive(Clone, Debug)]
pub struct Table {
    pub data: Vec<f32>,
}

pub struct SetupCtx<'a> {
    pub sample_rate: u32,
    pub block_size: usize,
    pub order: u8,
    pub tables: &'a [Table],
}

pub struct RunCtx<'a> {
    pub sample_rate: u32,
    pub block_size: usize,
    pub block_start: u64,
    pub frame_offset: usize,
    pub frame_count: usize,
    pub note_start: u64,
    pub note_end: u64,
    pub in_bus: usize,
    pub out_bus: usize,
    pub buses: &'a mut BusBank,
    pub tables: &'a [Table],
    pub pose: &'a PoseInlet,
    pub scene: &'a SceneInlet,
    pub synth_controls: &'a SynthControls,
    pub order: u8,
}

impl RunCtx<'_> {
    /// 0 at the first sample of the note, 1 at the last.
    #[inline]
    pub fn note_phase(&self, frame_in_block: usize) -> f32 {
        let abs = self.block_start + frame_in_block as u64;
        let dur = self.note_end.saturating_sub(self.note_start).max(1);
        let into = abs.saturating_sub(self.note_start);
        (into as f32) / (dur as f32)
    }
}

pub trait Instrument: Send {
    fn init(&mut self, ctx: &mut SetupCtx<'_>);
    fn configure(&mut self, ctx: &mut SetupCtx<'_>);
    fn run(&mut self, ctx: &mut RunCtx<'_>);
}
