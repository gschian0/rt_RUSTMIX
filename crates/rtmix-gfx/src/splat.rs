//! Loader for the widely used `.splat` interchange format.
//!
//! Each gaussian is 32 bytes: position `3 x f32`, scale `3 x f32`, colour
//! `4 x u8` (RGBA), rotation `4 x u8` holding a quaternion as `w, x, y, z`
//! packed with `b = q * 128 + 128`.

use crate::slice::Gaussian3;

pub const SPLAT_STRIDE: usize = 32;

#[derive(Debug)]
pub struct SplatError {
    pub message: String,
}

impl SplatError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Build the 3x3 covariance from a rotation quaternion and per-axis scales.
pub fn covariance_from(rotation: [f32; 4], scale: [f32; 3]) -> [f32; 9] {
    let [w, x, y, z] = rotation;
    let norm = (w * w + x * x + y * y + z * z).sqrt().max(1.0e-8);
    let (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm);
    let r = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ];
    // M = R * S, then Sigma = M * M^T.
    let mut m = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            m[row * 3 + col] = r[row * 3 + col] * scale[col];
        }
    }
    let mut cov = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += m[row * 3 + k] * m[col * 3 + k];
            }
            cov[row * 3 + col] = sum;
        }
    }
    cov
}

/// Number of gaussians in the file without decoding any of them.
pub fn splat_count(bytes: &[u8]) -> Result<usize, SplatError> {
    validate(bytes)?;
    Ok(bytes.len() / SPLAT_STRIDE)
}

fn validate(bytes: &[u8]) -> Result<(), SplatError> {
    if bytes.len() < SPLAT_STRIDE {
        return Err(SplatError::new("splat file is shorter than one gaussian"));
    }
    if bytes.len() % SPLAT_STRIDE != 0 {
        return Err(SplatError::new(format!(
            "splat file length {} is not a multiple of {SPLAT_STRIDE}",
            bytes.len()
        )));
    }
    Ok(())
}

fn decode(record: &[u8]) -> Gaussian3 {
    let mean = [
        read_f32(&record[0..4]),
        read_f32(&record[4..8]),
        read_f32(&record[8..12]),
    ];
    let scale = [
        read_f32(&record[12..16]),
        read_f32(&record[16..20]),
        read_f32(&record[20..24]),
    ];
    let color = [
        record[24] as f32 / 255.0,
        record[25] as f32 / 255.0,
        record[26] as f32 / 255.0,
    ];
    let opacity = record[27] as f32 / 255.0;
    let rotation = [
        (record[28] as f32 - 128.0) / 128.0,
        (record[29] as f32 - 128.0) / 128.0,
        (record[30] as f32 - 128.0) / 128.0,
        (record[31] as f32 - 128.0) / 128.0,
    ];
    Gaussian3 {
        mean,
        cov: covariance_from(rotation, scale),
        color,
        opacity,
    }
}

/// Decode at most `budget` gaussians, striding through the file so the subset
/// still covers the whole scene. Multi-million-gaussian scenes would otherwise
/// cost hundreds of megabytes just to throw most of it away.
pub fn load_splat_subset(bytes: &[u8], budget: usize) -> Result<Vec<Gaussian3>, SplatError> {
    validate(bytes)?;
    let count = bytes.len() / SPLAT_STRIDE;
    // Round the stride up so `budget` is a hard cap, not an approximation.
    let stride = if budget == 0 || budget >= count {
        1
    } else {
        count.div_ceil(budget)
    };
    let mut out = Vec::with_capacity(count / stride + 1);
    let mut index = 0usize;
    while index < count {
        out.push(decode(
            &bytes[index * SPLAT_STRIDE..(index + 1) * SPLAT_STRIDE],
        ));
        index += stride;
    }
    Ok(out)
}

pub fn load_splat(bytes: &[u8]) -> Result<Vec<Gaussian3>, SplatError> {
    load_splat_subset(bytes, usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(mean: [f32; 3], scale: [f32; 3], rgba: [u8; 4], quat: [f32; 4]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in mean.iter().chain(scale.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&rgba);
        for q in quat {
            bytes.push((q * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8);
        }
        bytes
    }

    #[test]
    fn round_trips_a_single_gaussian() {
        let bytes = encode(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [255, 128, 0, 200],
            [1.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(bytes.len(), SPLAT_STRIDE);
        let loaded = load_splat(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        let g = &loaded[0];
        assert!((g.mean[0] - 1.0).abs() < 1.0e-6);
        assert!((g.opacity - 200.0 / 255.0).abs() < 1.0e-3);
        // Identity rotation leaves a diagonal covariance of squared scales.
        assert!((g.cov[0] - 0.01).abs() < 1.0e-6, "cov {:?}", g.cov);
        assert!((g.cov[4] - 0.04).abs() < 1.0e-6);
        assert!((g.cov[8] - 0.09).abs() < 1.0e-6);
        assert!(g.cov[1].abs() < 1.0e-6);
    }

    #[test]
    fn subset_loading_strides_across_the_whole_file() {
        let mut bytes = Vec::new();
        for i in 0..100 {
            bytes.extend_from_slice(&encode(
                [i as f32, 0.0, 0.0],
                [1.0; 3],
                [10, 20, 30, 255],
                [1.0, 0.0, 0.0, 0.0],
            ));
        }
        assert_eq!(splat_count(&bytes).unwrap(), 100);
        let subset = load_splat_subset(&bytes, 10).unwrap();
        assert_eq!(subset.len(), 10);
        // Stride 10 keeps the first record and reaches the far end of the file.
        assert!((subset[0].mean[0] - 0.0).abs() < 1.0e-6);
        assert!((subset[9].mean[0] - 90.0).abs() < 1.0e-6);
        // Budgets never overshoot, even when they do not divide evenly.
        assert!(load_splat_subset(&bytes, 30).unwrap().len() <= 30);
        assert!(load_splat_subset(&bytes, 7).unwrap().len() <= 7);
        // A budget at or above the count keeps everything.
        assert_eq!(load_splat_subset(&bytes, 100).unwrap().len(), 100);
        assert_eq!(load_splat(&bytes).unwrap().len(), 100);
    }

    #[test]
    fn rejects_a_truncated_record() {
        let mut bytes = encode([0.0; 3], [1.0; 3], [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]);
        bytes.pop();
        assert!(load_splat(&bytes).is_err());
    }

    #[test]
    fn rotation_is_orthonormal_so_the_trace_is_preserved() {
        let axis = (0.5f32).sqrt();
        let cov = covariance_from([axis, axis, 0.0, 0.0], [0.3, 0.3, 0.3]);
        let trace = cov[0] + cov[4] + cov[8];
        assert!((trace - 3.0 * 0.09).abs() < 1.0e-5, "trace {trace}");
    }
}
