//! Counting allocator around the audio callback.
//! This is its own test binary so other tests cannot bump the counters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rtmix_core::{Engine, Instrument, RunCtx, SetupCtx};

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.fetch_add(1, Ordering::SeqCst);
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct Hold;

impl Instrument for Hold {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {}
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            ctx.buses.add(ctx.out_bus, 0, frame, 0.25);
        }
    }
}

fn allocs() -> usize {
    ALLOCS.load(Ordering::SeqCst)
}

fn frees() -> usize {
    FREES.load(Ordering::SeqCst)
}

#[test]
fn audio_callback_does_not_allocate() {
    let mut engine = Engine::new(48_000, 2, 128);
    engine
        .schedule_samples(0, 48_000, 0, engine.out_bus(), Box::new(Hold))
        .unwrap();

    let before_a = allocs();
    let before_f = frees();
    engine.process_block();
    assert_eq!(allocs(), before_a, "process_block allocated");
    assert_eq!(frees(), before_f, "process_block freed");

    let before_a = allocs();
    let before_f = frees();
    engine.process_block();
    assert_eq!(allocs(), before_a, "second block allocated");
    assert_eq!(frees(), before_f, "second block freed");

    let mut short = Engine::new(48_000, 1, 64);
    short
        .schedule_samples(0, 64, 0, short.out_bus(), Box::new(Hold))
        .unwrap();
    let before_a = allocs();
    let before_f = frees();
    short.process_block();
    assert_eq!(allocs(), before_a, "retiring a note allocated");
    assert_eq!(frees(), before_f, "retiring a note freed");
    assert!(short.bus_plane(short.out_bus(), 0).iter().any(|s| *s > 0.0));
}
