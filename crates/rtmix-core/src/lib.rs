//! Real-time engine core: sample clock, buses, pfields, and a binary-heap scheduler.
//!
//! The audio callback (`Engine::process_block`) does not allocate, free, lock, or parse.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bus;
pub mod engine;
pub mod instrument;
pub mod math;
pub mod pfield;
pub mod queue;
pub mod scene;
pub mod seqlock;

pub use bus::{Bus, BusBank};
pub use engine::{CoreError, Engine, MeterSnapshot, Route, MAX_NOTES};
pub use instrument::{Instrument, RunCtx, SetupCtx, Table};
pub use pfield::Pfield;
pub use queue::ControlQueue;
pub use scene::{
    AudioScope, Pose, SceneInlet, SceneNote, Shared, SynthControls, SCOPE_LEN, SYNTH_CHANT,
    SYNTH_COUNT, SYNTH_FOF, SYNTH_FORMANT, SYNTH_LPC, SYNTH_SPEAK,
};
pub use seqlock::{ClockSnapshot, Seqlock};
