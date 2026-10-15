//! Sample-accurate binary heap scheduler.
//!
//! `process_block` drains a preallocated control queue, runs due notes for the
//! current block only, and publishes a seqlock snapshot. It does not allocate.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::sync::atomic::Ordering as AtomicOrdering;

use crate::bus::BusBank;
use crate::instrument::{Instrument, RunCtx, SetupCtx, Table};
use crate::math;
use crate::queue::Note;
use crate::scene::Shared;
use crate::seqlock::ClockSnapshot;

pub const MAX_NOTES: usize = 1024;

#[derive(Clone, Debug)]
pub struct CoreError {
    pub message: &'static str,
}

impl CoreError {
    pub fn new(message: &'static str) -> Self {
        Self { message }
    }
}

#[derive(Clone, Debug)]
pub struct Route {
    pub input: String,
    pub output: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MeterSnapshot {
    pub out_peak: [f32; 8],
    pub hoa_peak: [f32; 16],
}

struct HeapNote(Note);

impl PartialEq for HeapNote {
    fn eq(&self, other: &Self) -> bool {
        self.0.start == other.0.start && self.0.seq == other.0.seq
    }
}
impl Eq for HeapNote {}
impl PartialOrd for HeapNote {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapNote {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .start
            .cmp(&self.0.start)
            .then(other.0.seq.cmp(&self.0.seq))
    }
}

pub struct Engine {
    sample_rate: u32,
    block_size: usize,
    output_channels: usize,
    sample_time: u64,
    order: u8,
    next_seq: u64,
    buses: BusBank,
    tables: Vec<Table>,
    routes: BTreeMap<String, Route>,
    out_bus: usize,
    spat_bus: usize,
    aux_bus: usize,
    pending: BinaryHeap<HeapNote>,
    active: Vec<Note>,
    retired: Vec<Note>,
    shared: Arc<Shared>,
}

impl Engine {
    pub fn new(sample_rate: u32, channels: usize, block_size: usize) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let block_size = block_size.max(1);
        let mut buses = BusBank::new(block_size);
        let in_bus = buses.ensure("in", channels);
        let out_bus = buses.ensure("out", channels);
        let aux_bus = buses.ensure("aux", 1);
        let spat_bus = buses.ensure("spat", 4);
        let _ = (in_bus, aux_bus);

        let mut routes = BTreeMap::new();
        let route = |input: &str, output: &str| Route {
            input: String::from(input),
            output: String::from(output),
        };
        routes.insert(String::from("FORMANT"), route("aux", "out"));
        routes.insert(String::from("CHANT"), route("aux", "out"));
        routes.insert(String::from("FOF"), route("aux", "out"));
        routes.insert(String::from("LPC"), route("aux", "out"));
        routes.insert(String::from("SPEAK"), route("aux", "out"));
        routes.insert(String::from("WAVETABLE"), route("aux", "aux"));
        routes.insert(String::from("MIX"), route("aux", "out"));
        routes.insert(String::from("FILTER"), route("aux", "aux"));
        routes.insert(String::from("DELAY"), route("aux", "aux"));
        routes.insert(String::from("REVERB"), route("aux", "out"));
        routes.insert(String::from("SPAT"), route("aux", "spat"));
        routes.insert(String::from("ROTATE"), route("spat", "spat"));
        routes.insert(String::from("DECODE"), route("spat", "out"));
        routes.insert(String::from("BINAURAL"), route("spat", "out"));
        routes.insert(String::from("SPLAT"), route("out", "out"));
        routes.insert(String::from("SPLAT4D"), route("out", "out"));
        routes.insert(String::from("CAMERA"), route("out", "out"));

        let mut tables = Vec::new();
        tables.push(Table {
            data: default_sine(2048),
        });

        let mut pending = BinaryHeap::new();
        pending.reserve(MAX_NOTES);
        let mut active = Vec::new();
        active.reserve(MAX_NOTES);
        let mut retired = Vec::new();
        retired.reserve(MAX_NOTES);

        Self {
            sample_rate,
            block_size,
            output_channels: channels,
            sample_time: 0,
            order: 1,
            next_seq: 1,
            buses,
            tables,
            routes,
            out_bus,
            spat_bus,
            aux_bus,
            pending,
            active,
            retired,
            shared: Arc::new(Shared::new(sample_rate, MAX_NOTES)),
        }
    }

    pub fn shared(&self) -> Arc<Shared> {
        Arc::clone(&self.shared)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sample_time(&self) -> u64 {
        self.sample_time
    }

    pub fn order(&self) -> u8 {
        self.order
    }

    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    pub fn seconds(&self) -> f64 {
        self.sample_time as f64 / self.sample_rate as f64
    }

    /// Control thread. Resizes buses.
    pub fn set_params(&mut self, sample_rate: u32, channels: usize, block_size: usize) {
        self.sample_rate = sample_rate.max(1);
        self.output_channels = channels.max(1);
        self.block_size = block_size.max(1);
        self.buses.set_block_size(self.block_size);
        self.out_bus = self.buses.ensure("out", self.output_channels);
        let _ = self.buses.ensure("in", self.output_channels);
        self.aux_bus = self.buses.ensure("aux", 1);
        let hoa = hoa_channels(self.order);
        self.spat_bus = self.buses.ensure("spat", hoa);
    }

    pub fn set_order(&mut self, order: u8) -> Result<(), CoreError> {
        if order == 0 || order > 7 {
            return Err(CoreError::new("ambisonic order must be 1..=7"));
        }
        self.order = order;
        self.spat_bus = self.buses.ensure("spat", hoa_channels(order));
        Ok(())
    }

    pub fn bus_id(&self, name: &str) -> Option<usize> {
        self.buses.id(name)
    }

    pub fn bus_config(&mut self, instrument: &str, bus: &str) -> Result<(), CoreError> {
        let channels = if bus == "spat" {
            hoa_channels(self.order)
        } else if bus == "out" || bus == "in" {
            self.output_channels
        } else if let Some(id) = self.buses.id(bus) {
            self.buses.channels(id)
        } else {
            1
        };
        let id = self.buses.ensure(bus, channels);
        if bus == "out" {
            self.out_bus = id;
        } else if bus == "spat" {
            self.spat_bus = id;
        } else if bus == "aux" {
            self.aux_bus = id;
        }
        let route = self
            .routes
            .entry(String::from(instrument))
            .or_insert_with(|| Route {
                input: String::from("aux"),
                output: String::from("aux"),
            });
        route.output = String::from(bus);
        Ok(())
    }

    pub fn set_input_bus(&mut self, instrument: &str, bus: &str) -> Result<(), CoreError> {
        let _ = self.buses.ensure(bus, 1);
        let route = self
            .routes
            .entry(String::from(instrument))
            .or_insert_with(|| Route {
                input: String::from("aux"),
                output: String::from("aux"),
            });
        route.input = String::from(bus);
        Ok(())
    }

    pub fn add_table(&mut self, data: Vec<f32>) -> usize {
        let id = self.tables.len();
        self.tables.push(Table { data });
        id
    }

    pub fn table(&self, id: usize) -> Option<&[f32]> {
        self.tables.get(id).map(|t| t.data.as_slice())
    }

    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    pub fn schedule_named(
        &mut self,
        name: &str,
        start_sec: f64,
        dur_sec: f64,
        inst: Box<dyn Instrument>,
    ) -> Result<(), CoreError> {
        let (input, output) = if let Some(route) = self.routes.get(name) {
            (route.input.clone(), route.output.clone())
        } else {
            (String::from("aux"), String::from("out"))
        };
        let in_bus = self.buses.id_or_create(&input, 1);
        let out_channels = if output == "spat" {
            hoa_channels(self.order)
        } else if output == "out" || output == "in" {
            self.output_channels
        } else {
            self.buses
                .id(&output)
                .map(|id| self.buses.channels(id))
                .unwrap_or(1)
                .max(1)
        };
        let out_bus = self.buses.id_or_create(&output, out_channels);
        if output == "spat" {
            self.spat_bus = out_bus;
        }
        if output == "out" {
            self.out_bus = out_bus;
        }
        self.schedule_at_bus(start_sec, dur_sec, in_bus, out_bus, inst)
    }

    pub fn schedule_at_bus(
        &mut self,
        start_sec: f64,
        dur_sec: f64,
        in_bus: usize,
        out_bus: usize,
        inst: Box<dyn Instrument>,
    ) -> Result<(), CoreError> {
        let start = seconds_to_samples(start_sec, self.sample_rate);
        let dur = seconds_to_samples(dur_sec.max(0.0), self.sample_rate);
        self.schedule_samples(start, start.saturating_add(dur), in_bus, out_bus, inst)
    }

    pub fn schedule_samples(
        &mut self,
        start: u64,
        end: u64,
        in_bus: usize,
        out_bus: usize,
        mut inst: Box<dyn Instrument>,
    ) -> Result<(), CoreError> {
        let mut setup = SetupCtx {
            sample_rate: self.sample_rate,
            block_size: self.block_size,
            order: self.order,
            tables: &self.tables,
        };
        inst.configure(&mut setup);
        inst.init(&mut setup);
        let note = Note {
            start,
            end,
            seq: self.next_seq,
            in_bus,
            out_bus,
            inst,
        };
        self.next_seq = self.next_seq.wrapping_add(1);
        self.shared
            .queue
            .push(note)
            .map_err(|_| CoreError::new("control queue is full"))
    }

    pub fn set_listener(&self, yaw: f32, pitch: f32, roll: f32) {
        self.shared.set_listener(yaw, pitch, roll);
    }

    pub fn listener(&self) -> (f32, f32, f32) {
        self.shared.listener()
    }

    pub fn pose(&self) -> crate::scene::Pose {
        self.shared.pose.load()
    }

    pub fn read_snapshot(&self) -> ClockSnapshot {
        self.shared.snapshot.read()
    }

    pub fn meter_snapshot(&self) -> MeterSnapshot {
        let mut meters = MeterSnapshot::default();
        for (i, slot) in self.shared.out_peak.iter().enumerate() {
            meters.out_peak[i] = f32::from_bits(slot.load(AtomicOrdering::Relaxed));
        }
        for (i, slot) in self.shared.hoa_peak.iter().enumerate() {
            meters.hoa_peak[i] = f32::from_bits(slot.load(AtomicOrdering::Relaxed));
        }
        meters
    }

    pub fn out_bus(&self) -> usize {
        self.out_bus
    }

    pub fn spat_bus(&self) -> usize {
        self.spat_bus
    }

    pub fn aux_bus(&self) -> usize {
        self.aux_bus
    }

    pub fn bus_sample(&self, bus: usize, ch: usize, frame: usize) -> f32 {
        self.buses.sample(bus, ch, frame)
    }

    pub fn bus_plane(&self, bus: usize, ch: usize) -> &[f32] {
        self.buses.plane(bus, ch)
    }

    /// Render one block. Allocation-free when the engine was created with
    /// `Engine::new` and fewer than [`MAX_NOTES`] notes are in flight.
    pub fn process_block(&mut self) {
        self.drain_queue();

        let block_start = self.sample_time;
        let block_size = self.block_size;
        let block_end = block_start + block_size as u64;
        self.buses.silence();

        loop {
            let due = match self.pending.peek() {
                Some(note) => note.0.start < block_end,
                None => false,
            };
            if !due {
                break;
            }
            let Some(note) = self.pending.pop() else {
                break;
            };
            if self.active.len() < self.active.capacity() {
                self.active.push(note.0);
            } else {
                core::mem::forget(note);
            }
        }

        let sample_rate = self.sample_rate;
        let order = self.order;
        {
            let Engine {
                active,
                buses,
                tables,
                shared,
                retired,
                ..
            } = &mut *self;
            let mut i = 0;
            while i < active.len() {
                let start = active[i].start.max(block_start);
                let end = active[i].end.min(block_end);
                if start < end {
                    let frame_offset = (start - block_start) as usize;
                    let frame_count = (end - start) as usize;
                    let in_bus = active[i].in_bus;
                    let out_bus = active[i].out_bus;
                    let note_start = active[i].start;
                    let note_end = active[i].end;
                    let inst: *mut dyn Instrument = active[i].inst.as_mut();
                    let mut ctx = RunCtx {
                        sample_rate,
                        block_size,
                        block_start,
                        frame_offset,
                        frame_count,
                        note_start,
                        note_end,
                        in_bus,
                        out_bus,
                        buses,
                        tables,
                        pose: &shared.pose,
                        scene: &shared.scene,
                        synth_controls: &shared.synth_controls,
                        order,
                    };
                    unsafe {
                        (*inst).run(&mut ctx);
                    }
                }
                if active[i].end <= block_end {
                    let finished = active.swap_remove(i);
                    if retired.len() < retired.capacity() {
                        retired.push(finished);
                    } else {
                        core::mem::forget(finished);
                    }
                } else {
                    i += 1;
                }
            }
        }

        self.publish_meters();
        self.sample_time = block_end;
        self.publish_snapshot();
    }

    pub fn process_until(&mut self, end_sample: u64) {
        while self.sample_time < end_sample {
            self.process_block();
        }
    }

    pub fn process_seconds(&mut self, seconds: f64) {
        let samples = seconds_to_samples(seconds.max(0.0), self.sample_rate);
        self.process_until(self.sample_time.saturating_add(samples));
    }

    pub fn copy_block_interleaved(
        &self,
        frame_offset: usize,
        frame_count: usize,
        dst: &mut [f32],
        channels: usize,
    ) {
        let channels = channels.max(1);
        let src_ch = self.buses.channels(self.out_bus).min(channels);
        for i in 0..frame_count {
            let frame = frame_offset + i;
            for ch in 0..channels {
                let sample = if ch < src_ch {
                    self.buses.sample(self.out_bus, ch, frame)
                } else {
                    0.0
                };
                let idx = i * channels + ch;
                if idx < dst.len() {
                    dst[idx] = sample;
                }
            }
        }
    }

    fn drain_queue(&mut self) {
        loop {
            let popped = self.shared.queue.pop();
            let Some(note) = popped else {
                break;
            };
            if self.pending.len() < self.pending.capacity() {
                self.pending.push(HeapNote(note));
            } else {
                core::mem::forget(note);
            }
        }
    }

    fn publish_meters(&self) {
        let ch = self.buses.channels(self.out_bus).min(8);
        for c in 0..ch {
            let mut peak = 0.0f32;
            let plane = self.buses.plane(self.out_bus, c);
            for s in plane {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
            self.shared.out_peak[c].store(peak.to_bits(), AtomicOrdering::Relaxed);
        }
        // Tap channel 0 so the GUI can run an FFT without touching audio memory.
        if ch > 0 {
            for s in self.buses.plane(self.out_bus, 0) {
                self.shared.scope.push(*s);
            }
        }
        let hoa = self.buses.channels(self.spat_bus).min(16);
        for c in 0..hoa {
            let mut peak = 0.0f32;
            let plane = self.buses.plane(self.spat_bus, c);
            for s in plane {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
            self.shared.hoa_peak[c].store(peak.to_bits(), AtomicOrdering::Relaxed);
        }
    }

    fn publish_snapshot(&self) {
        let pose = self.shared.pose.load();
        let (kind, asset, t0, t1) = self.shared.scene.load();
        self.shared.snapshot.publish(ClockSnapshot {
            sample_time: self.sample_time,
            sample_rate: self.sample_rate,
            yaw: pose.yaw,
            pitch: pose.pitch,
            roll: pose.roll,
            x: pose.x,
            y: pose.y,
            z: pose.z,
            scene_kind: kind,
            scene_asset: asset,
            scene_t0: t0,
            scene_t1: t1,
        });
    }
}

fn hoa_channels(order: u8) -> usize {
    let n = order as usize + 1;
    n * n
}

fn seconds_to_samples(seconds: f64, sample_rate: u32) -> u64 {
    if seconds <= 0.0 {
        return 0;
    }
    let samples = seconds * sample_rate as f64;
    if samples >= u64::MAX as f64 {
        u64::MAX
    } else {
        libm::round(samples) as u64
    }
}

fn default_sine(size: usize) -> Vec<f32> {
    let size = size.max(4);
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let phase = (i as f32) / (size as f32);
        data.push(math::sin(phase * math::TAU));
    }
    data
}
