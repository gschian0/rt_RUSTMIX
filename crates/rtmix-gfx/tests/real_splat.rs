//! Checks the renderer against a real splat file when one is present.
//!
//! Drop a `.splat` into `assets/` to enable this; the test is skipped when no
//! asset is there so a fresh clone still passes.

use std::path::PathBuf;

use rtmix_gfx::{load_splat_subset, splat_count, Camera, Reactivity, Spectrum, SplatView};

fn assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

#[test]
fn renders_a_real_splat_scene() {
    let candidates = ["garden.splat", "truck.splat", "nike.splat"];
    let Some((name, bytes)) = candidates
        .iter()
        .find_map(|name| std::fs::read(assets().join(name)).ok().map(|b| (*name, b)))
    else {
        eprintln!("skipping: no splat asset in {}", assets().display());
        return;
    };

    let total = splat_count(&bytes).expect("header should validate");
    assert!(total > 1_000, "{name} only holds {total} gaussians");
    let gaussians = load_splat_subset(&bytes, 40_000).expect("{name} should parse");
    assert!(
        gaussians.len() <= 40_000,
        "{name}: budget ignored, got {}",
        gaussians.len()
    );
    assert!(
        gaussians.len() > 1_000,
        "{name}: subset too small, got {}",
        gaussians.len()
    );
    assert!(
        gaussians.iter().any(|g| g.opacity > 0.5),
        "every gaussian in {name} is transparent"
    );

    let mut view = SplatView::new(&gaussians, 40_000);
    assert!(!view.is_empty());
    let camera = Camera {
        target: view.center(),
        distance: view.radius() * 1.1,
        yaw: 0.6,
        pitch: 0.25,
        fov_y: 1.0,
    };
    let mut rgba = Vec::new();
    view.render(
        160,
        120,
        &camera,
        &Spectrum::default(),
        &Reactivity::default(),
        &mut rgba,
    );
    assert_eq!(rgba.len(), 160 * 120 * 4);
    let lit = rgba
        .chunks_exact(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 90)
        .count();
    assert!(lit > 400, "{name}: only {lit} lit pixels; the scene looks blank");
}
