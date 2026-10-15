//! Preallocated audio buses. The callback only writes existing slots.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct Bus {
    pub name: String,
    pub channels: usize,
    pub data: Vec<f32>,
}

pub struct BusBank {
    buses: Vec<Bus>,
    names: BTreeMap<String, usize>,
    block_size: usize,
}

impl BusBank {
    pub fn new(block_size: usize) -> Self {
        Self {
            buses: Vec::new(),
            names: BTreeMap::new(),
            block_size: block_size.max(1),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn len(&self) -> usize {
        self.buses.len()
    }

    pub fn id(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    pub fn name(&self, id: usize) -> Option<&str> {
        self.buses.get(id).map(|b| b.name.as_str())
    }

    pub fn channels(&self, id: usize) -> usize {
        self.buses.get(id).map(|b| b.channels).unwrap_or(0)
    }

    /// Create or resize a bus. Control thread only: this allocates.
    pub fn ensure(&mut self, name: &str, channels: usize) -> usize {
        let channels = channels.max(1);
        let samples = channels.saturating_mul(self.block_size);
        if let Some(&id) = self.names.get(name) {
            let bus = &mut self.buses[id];
            bus.channels = channels;
            bus.data.clear();
            bus.data.resize(samples, 0.0);
            return id;
        }
        let id = self.buses.len();
        self.names.insert(String::from(name), id);
        self.buses.push(Bus {
            name: String::from(name),
            channels,
            data: alloc::vec![0.0; samples],
        });
        id
    }

    /// Create a bus, or widen an existing one. Never shrinks.
    pub fn id_or_create(&mut self, name: &str, channels: usize) -> usize {
        if let Some(id) = self.names.get(name).copied() {
            if self.buses[id].channels >= channels {
                return id;
            }
        }
        self.ensure(name, channels)
    }

    pub fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size.max(1);
        for bus in &mut self.buses {
            let samples = bus.channels.saturating_mul(self.block_size);
            bus.data.clear();
            bus.data.resize(samples, 0.0);
        }
    }

    #[inline]
    pub fn silence(&mut self) {
        for bus in &mut self.buses {
            bus.data.fill(0.0);
        }
    }

    #[inline]
    pub fn sample(&self, bus: usize, ch: usize, frame: usize) -> f32 {
        let Some(b) = self.buses.get(bus) else {
            return 0.0;
        };
        if ch >= b.channels || frame >= self.block_size {
            return 0.0;
        }
        b.data[ch * self.block_size + frame]
    }

    #[inline]
    pub fn add(&mut self, bus: usize, ch: usize, frame: usize, value: f32) {
        let block = self.block_size;
        let Some(b) = self.buses.get_mut(bus) else {
            return;
        };
        if ch >= b.channels || frame >= block {
            return;
        }
        b.data[ch * block + frame] += value;
    }

    #[inline]
    pub fn set(&mut self, bus: usize, ch: usize, frame: usize, value: f32) {
        let block = self.block_size;
        let Some(b) = self.buses.get_mut(bus) else {
            return;
        };
        if ch >= b.channels || frame >= block {
            return;
        }
        b.data[ch * block + frame] = value;
    }

    pub fn plane(&self, bus: usize, ch: usize) -> &[f32] {
        let block = self.block_size;
        let Some(b) = self.buses.get(bus) else {
            return &[];
        };
        if ch >= b.channels {
            return &[];
        }
        let start = ch * block;
        &b.data[start..start + block]
    }
}
