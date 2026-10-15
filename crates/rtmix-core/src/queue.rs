//! Bounded SPSC queue. Slots are allocated when the queue is created.
//! Push and pop move an existing note; they do not allocate.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::instrument::Instrument;

pub struct Note {
    pub start: u64,
    pub end: u64,
    pub seq: u64,
    pub in_bus: usize,
    pub out_bus: usize,
    pub inst: Box<dyn Instrument>,
}

pub struct ControlQueue {
    slots: Box<[UnsafeCell<Option<Note>>]>,
    cap: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl ControlQueue {
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(UnsafeCell::new(None));
        }
        Self {
            slots: slots.into_boxed_slice(),
            cap,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn push(&self, note: Note) -> Result<(), Note> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= self.cap {
            return Err(note);
        }
        let idx = tail % self.cap;
        unsafe {
            *self.slots[idx].get() = Some(note);
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<Note> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head % self.cap;
        let note = unsafe { (*self.slots[idx].get()).take() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        note
    }
}

unsafe impl Send for ControlQueue {}
unsafe impl Sync for ControlQueue {}
