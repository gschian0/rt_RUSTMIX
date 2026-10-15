//! Niantic-style SPZ container: gzip around an NGSP header and packed gaussians.
//!
//! Layout after inflate:
//! magic `0x5053474E` ('NGSP'), version 2, count, sh degree, fractional bits, flags, reserved,
//! then int24 positions, uint8 log-scales, uint8 quaternion xyz, uint8 opacity, uint8 color.

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::ply::covariance_from_scale_rot;
use crate::slice::Gaussian3;

const MAGIC: u32 = 0x5053_474E;

#[derive(Debug)]
pub struct SpzError {
    pub message: String,
}

impl SpzError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub fn store_spz(gaussians: &[Gaussian3]) -> Result<Vec<u8>, SpzError> {
    let raw = pack(gaussians);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&raw)
        .map_err(|err| SpzError::new(err.to_string()))?;
    encoder
        .finish()
        .map_err(|err| SpzError::new(err.to_string()))
}

pub fn load_spz(bytes: &[u8]) -> Result<Vec<Gaussian3>, SpzError> {
    let raw = if bytes.len() >= 4 && u32::from_le_bytes(bytes[0..4].try_into().unwrap()) == MAGIC {
        bytes.to_vec()
    } else {
        let mut decoder = GzDecoder::new(bytes);
        let mut raw = Vec::new();
        decoder
            .read_to_end(&mut raw)
            .map_err(|err| SpzError::new(err.to_string()))?;
        raw
    };
    unpack(&raw)
}

fn pack(gaussians: &[Gaussian3]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(gaussians.len() as u32).to_le_bytes());
    out.push(0);
    out.push(8);
    out.push(0);
    out.push(0);
    for g in gaussians {
        for axis in 0..3 {
            write_i24(&mut out, (g.mean[axis] * 256.0).round() as i32);
        }
    }
    let poses: Vec<([f32; 4], [f32; 3])> =
        gaussians.iter().map(|g| pose_from_cov(&g.cov)).collect();
    for (_, scale) in &poses {
        for axis in 0..3 {
            let q = ((libm::logf(scale[axis].max(1.0e-8)) + 10.0) * 16.0).clamp(0.0, 255.0) as u8;
            out.push(q);
        }
    }
    for (quat, _) in &poses {
        out.push(quant_unit(quat[1]));
        out.push(quant_unit(quat[2]));
        out.push(quant_unit(quat[3]));
    }
    for g in gaussians {
        out.push((g.opacity.clamp(0.0, 1.0) * 255.0) as u8);
    }
    for g in gaussians {
        for channel in 0..3 {
            out.push((g.color[channel].clamp(0.0, 1.0) * 255.0) as u8);
        }
    }
    out
}

fn unpack(raw: &[u8]) -> Result<Vec<Gaussian3>, SpzError> {
    if raw.len() < 16 {
        return Err(SpzError::new("SPZ header is short"));
    }
    let magic = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(SpzError::new("SPZ magic is not NGSP"));
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    if version != 2 {
        return Err(SpzError::new("unsupported SPZ version"));
    }
    let count = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let fractional = raw[13];
    let mut cursor = 16;
    let mut means = vec![[0.0f32; 3]; count];
    let scale = (1u32 << fractional) as f32;
    for mean in &mut means {
        for axis in 0..3 {
            let v = read_i24(raw, &mut cursor)?;
            mean[axis] = v as f32 / scale;
        }
    }
    let mut scales = vec![[1.0f32; 3]; count];
    for scale_v in &mut scales {
        for axis in 0..3 {
            let b = *raw
                .get(cursor)
                .ok_or_else(|| SpzError::new("truncated scales"))?;
            cursor += 1;
            scale_v[axis] = libm::expf(b as f32 / 16.0 - 10.0);
        }
    }
    let mut rots = vec![[1.0f32, 0.0, 0.0, 0.0]; count];
    for rot in &mut rots {
        let x = dequant_unit(
            *raw.get(cursor)
                .ok_or_else(|| SpzError::new("truncated rot"))?,
        );
        cursor += 1;
        let y = dequant_unit(
            *raw.get(cursor)
                .ok_or_else(|| SpzError::new("truncated rot"))?,
        );
        cursor += 1;
        let z = dequant_unit(
            *raw.get(cursor)
                .ok_or_else(|| SpzError::new("truncated rot"))?,
        );
        cursor += 1;
        let w2 = (1.0 - x * x - y * y - z * z).max(0.0);
        rot[0] = libm::sqrtf(w2);
        rot[1] = x;
        rot[2] = y;
        rot[3] = z;
    }
    let mut opacity = vec![1.0f32; count];
    for value in &mut opacity {
        let b = *raw
            .get(cursor)
            .ok_or_else(|| SpzError::new("truncated opacity"))?;
        cursor += 1;
        *value = b as f32 / 255.0;
    }
    let mut color = vec![[1.0f32; 3]; count];
    for rgb in &mut color {
        for channel in 0..3 {
            let b = *raw
                .get(cursor)
                .ok_or_else(|| SpzError::new("truncated color"))?;
            cursor += 1;
            rgb[channel] = b as f32 / 255.0;
        }
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(Gaussian3 {
            mean: means[i],
            cov: covariance_from_scale_rot(scales[i], rots[i]),
            color: color[i],
            opacity: opacity[i],
        });
    }
    Ok(out)
}

fn write_i24(out: &mut Vec<u8>, value: i32) {
    let value = value.clamp(-8_388_608, 8_388_607) as u32;
    out.push((value & 0xff) as u8);
    out.push(((value >> 8) & 0xff) as u8);
    out.push(((value >> 16) & 0xff) as u8);
}

fn read_i24(raw: &[u8], cursor: &mut usize) -> Result<i32, SpzError> {
    if *cursor + 3 > raw.len() {
        return Err(SpzError::new("truncated position"));
    }
    let b0 = raw[*cursor] as u32;
    let b1 = raw[*cursor + 1] as u32;
    let b2 = raw[*cursor + 2] as u32;
    *cursor += 3;
    let mut value = b0 | (b1 << 8) | (b2 << 16);
    if value & 0x80_0000 != 0 {
        value |= 0xff00_0000;
    }
    Ok(value as i32)
}

fn quant_unit(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 127.0).round() as i32 + 128).clamp(0, 255) as u8
}

fn dequant_unit(v: u8) -> f32 {
    (v as f32 - 128.0) / 127.0
}

fn pose_from_cov(cov: &[f32; 9]) -> ([f32; 4], [f32; 3]) {
    let (evals, evecs) = jacobi_eigen(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut r = [0.0f32; 9];
    for col in 0..3 {
        let src = order[col];
        for row in 0..3 {
            r[row * 3 + col] = evecs[row * 3 + src];
        }
    }
    let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
        + r[2] * (r[3] * r[7] - r[4] * r[6]);
    if det < 0.0 {
        r[2] = -r[2];
        r[5] = -r[5];
        r[8] = -r[8];
    }
    let scale = [
        libm::sqrtf(evals[order[0]].max(0.0)),
        libm::sqrtf(evals[order[1]].max(0.0)),
        libm::sqrtf(evals[order[2]].max(0.0)),
    ];
    (mat_to_quat(&r), scale)
}

fn mat_to_quat(m: &[f32; 9]) -> [f32; 4] {
    let trace = m[0] + m[4] + m[8];
    let q = if trace > 0.0 {
        let s = libm::sqrtf(trace + 1.0) * 2.0;
        [
            0.25 * s,
            (m[7] - m[5]) / s,
            (m[2] - m[6]) / s,
            (m[3] - m[1]) / s,
        ]
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = libm::sqrtf(1.0 + m[0] - m[4] - m[8]) * 2.0;
        [
            (m[7] - m[5]) / s,
            0.25 * s,
            (m[1] + m[3]) / s,
            (m[2] + m[6]) / s,
        ]
    } else if m[4] > m[8] {
        let s = libm::sqrtf(1.0 + m[4] - m[0] - m[8]) * 2.0;
        [
            (m[2] - m[6]) / s,
            (m[1] + m[3]) / s,
            0.25 * s,
            (m[5] + m[7]) / s,
        ]
    } else {
        let s = libm::sqrtf((1.0 + m[8] - m[0] - m[4]).max(0.0)) * 2.0;
        let s = s.max(1.0e-8);
        [
            (m[3] - m[1]) / s,
            (m[2] + m[6]) / s,
            (m[5] + m[7]) / s,
            0.25 * s,
        ]
    };
    if q[0] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    }
}

fn jacobi_eigen(cov: &[f32; 9]) -> ([f32; 3], [f32; 9]) {
    let mut a = *cov;
    let mut v = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..8 {
        rotate_jacobi(&mut a, &mut v, 0, 1);
        rotate_jacobi(&mut a, &mut v, 0, 2);
        rotate_jacobi(&mut a, &mut v, 1, 2);
    }
    ([a[0], a[4], a[8]], v)
}

fn rotate_jacobi(a: &mut [f32; 9], v: &mut [f32; 9], p: usize, q: usize) {
    let apq = a[p * 3 + q];
    if apq.abs() < 1.0e-12 {
        return;
    }
    let app = a[p * 3 + p];
    let aqq = a[q * 3 + q];
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + libm::sqrtf(1.0 + tau * tau))
    } else {
        -1.0 / (-tau + libm::sqrtf(1.0 + tau * tau))
    };
    let c = 1.0 / libm::sqrtf(1.0 + t * t);
    let s = t * c;
    a[p * 3 + p] = app - t * apq;
    a[q * 3 + q] = aqq + t * apq;
    a[p * 3 + q] = 0.0;
    a[q * 3 + p] = 0.0;
    for k in 0..3 {
        if k == p || k == q {
            continue;
        }
        let aik = a[k * 3 + p];
        let ajk = a[k * 3 + q];
        let new_ik = c * aik - s * ajk;
        let new_jk = s * aik + c * ajk;
        a[k * 3 + p] = new_ik;
        a[p * 3 + k] = new_ik;
        a[k * 3 + q] = new_jk;
        a[q * 3 + k] = new_jk;
    }
    for k in 0..3 {
        let vip = v[k * 3 + p];
        let viq = v[k * 3 + q];
        v[k * 3 + p] = c * vip - s * viq;
        v[k * 3 + q] = s * vip + c * viq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spz_roundtrip_preserves_position_and_color() {
        let src = vec![Gaussian3 {
            mean: [0.5, -1.25, 2.0],
            cov: [0.04, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.09],
            color: [0.2, 0.4, 0.8],
            opacity: 0.75,
        }];
        let bytes = store_spz(&src).unwrap();
        let loaded = load_spz(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].mean[0] - 0.5).abs() < 1.0 / 256.0 + 1.0e-3);
        assert!((loaded[0].mean[1] + 1.25).abs() < 1.0 / 256.0 + 1.0e-3);
        assert!((loaded[0].color[2] - 0.8).abs() < 1.0 / 255.0 + 1.0e-3);
        assert!((loaded[0].opacity - 0.75).abs() < 1.0 / 255.0 + 1.0e-3);
        assert!(loaded[0].cov[1].abs() < 1.0e-3);
    }

    #[test]
    fn quaternion_puts_energy_off_the_diagonal() {
        let half = 0.35_f32;
        let mut q = [libm::cosf(half), 0.15, -0.2, libm::sinf(half)];
        let n = libm::sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for c in &mut q {
            *c /= n;
        }
        if q[0] < 0.0 {
            for c in &mut q {
                *c = -*c;
            }
        }
        let cov = covariance_from_scale_rot([2.0, 0.35, 0.7], q);
        let off = cov[1].abs() + cov[2].abs() + cov[5].abs();
        assert!(off > 0.05, "expected off-diagonal covariance, got {cov:?}");
        let src = vec![Gaussian3 {
            mean: [0.0, 0.0, 1.0],
            cov,
            color: [0.5, 0.5, 0.5],
            opacity: 1.0,
        }];
        let loaded = load_spz(&store_spz(&src).unwrap()).unwrap();
        let loaded_off = loaded[0].cov[1].abs() + loaded[0].cov[2].abs() + loaded[0].cov[5].abs();
        assert!(
            loaded_off > 0.05,
            "SPZ roundtrip dropped the rotation, cov {:?}",
            loaded[0].cov
        );
    }
}
