//! Sample-accurate scheduling.

use rtmix_core::{Engine, Instrument, RunCtx, SetupCtx};

struct Impulse;

impl Instrument for Impulse {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {}
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let at = ctx.block_start + ctx.frame_offset as u64;
        if at == ctx.note_start && ctx.frame_count > 0 {
            ctx.buses.add(ctx.out_bus, 0, ctx.frame_offset, 1.0);
        }
    }
}

#[test]
fn schedules_on_exact_sample() {
    let mut engine = Engine::new(48_000, 1, 32);
    engine
        .schedule_samples(40, 41, 0, engine.out_bus(), Box::new(Impulse))
        .unwrap();

    let mut rendered = Vec::new();
    for _ in 0..3 {
        engine.process_block();
        rendered.extend_from_slice(engine.bus_plane(engine.out_bus(), 0));
    }

    assert!(rendered[..40].iter().all(|s| *s == 0.0));
    assert!((rendered[40] - 1.0).abs() < 1.0e-6);
    assert!(rendered[41..].iter().all(|s| *s == 0.0));
    assert_eq!(engine.sample_time(), 96);
}
