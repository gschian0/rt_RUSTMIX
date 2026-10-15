//! Gaussian splat playback: loaders, a 4D time slice, and a 16x16 tile rasterizer.
//! The live audio callback never calls into this crate.

pub mod analysis;
pub mod drip;
pub mod ply;
pub mod preview;
pub mod raster;
pub mod slice;
pub mod splat;
pub mod spz;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use analysis::{fft_in_place, Analyzer, Spectrum, BANDS};
pub use drip::{DripMode, DripView};
pub use ply::{load_ply, PlyError};
pub use preview::{camera_eye, Camera, Reactivity, SplatView};
pub use raster::{blend_front_to_back, project_point, radix_sort_pairs, tile_bins, TILE_SIZE};
pub use slice::{file_time, slice_4d, Gaussian3, Gaussian4};
pub use splat::{load_splat, load_splat_subset, splat_count, SplatError, SPLAT_STRIDE};
pub use spz::{load_spz, store_spz, SpzError};

pub const RASTER_WGSL: &str = include_str!("../shaders/raster.wgsl");
pub const SLICE_WGSL: &str = include_str!("../shaders/slice4d.wgsl");
pub const RASTER_SLANG: &str = include_str!("../shaders/raster.slang");
pub const SLICE_SLANG: &str = include_str!("../shaders/slice4d.slang");

/// Score clock in seconds. Picture time uses the same sample counter as audio.
pub fn score_seconds(sample_time: u64, sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        0.0
    } else {
        sample_time as f32 / sample_rate as f32
    }
}
