//! ACN / SN3D (AmbiX) Ambisonics.
//!
//! Order 1 is always available. Order 3 is the default desktop feature.
//! Orders above 3 require the `higher_order` feature.
//!
//! Listener cartesian coordinates used by [`encode_cartesian`]: +X right, +Y up, +Z front.
//! Azimuth is 0 at the front and positive toward the left. Elevation is 0 on the horizon
//! and positive upward.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use rtmix_core::{CoreError, Engine, Instrument, Pfield, RunCtx, SetupCtx};

pub const fn max_order() -> usize {
    if cfg!(feature = "higher_order") {
        5
    } else if cfg!(feature = "order3") {
        3
    } else {
        1
    }
}

pub const fn max_channels() -> usize {
    (max_order() + 1) * (max_order() + 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeakerLayout {
    Stereo,
    Square,
    Cube,
}

impl SpeakerLayout {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "stereo" => Some(Self::Stereo),
            "square" => Some(Self::Square),
            "cube" => Some(Self::Cube),
            _ => None,
        }
    }

    pub fn speakers(self) -> usize {
        match self {
            Self::Stereo => 2,
            Self::Square => 4,
            Self::Cube => 8,
        }
    }

    /// These compact arrays are mode-matched at first order.
    pub fn decode_order(self, available: usize) -> usize {
        let cap = match self {
            Self::Stereo | Self::Square | Self::Cube => 1,
        };
        available.min(cap).max(1)
    }
}

#[derive(Clone, Debug)]
pub struct SpatError {
    pub message: &'static str,
}

impl SpatError {
    fn new(message: &'static str) -> Self {
        Self { message }
    }
}

pub fn channel_count(order: usize) -> usize {
    (order + 1) * (order + 1)
}

pub fn acn_index(order_n: i32, m: i32) -> usize {
    (order_n * order_n + order_n + m) as usize
}

/// Encode a unit-distance point source. `out` must hold `(order+1)^2` channels.
pub fn encode_aed(
    order: usize,
    azimuth: f32,
    elevation: f32,
    distance: f32,
    out: &mut [f32],
) -> Result<(), SpatError> {
    let x_front = libm::cosf(elevation) * libm::cosf(azimuth);
    let y_left = libm::cosf(elevation) * libm::sinf(azimuth);
    let z_up = libm::sinf(elevation);
    encode_ambisonic(order, x_front, y_left, z_up, distance, out)
}

/// `x` is right, `y` is up, `z` is front.
pub fn encode_cartesian(
    order: usize,
    x_right: f32,
    y_up: f32,
    z_front: f32,
    distance: f32,
    out: &mut [f32],
) -> Result<(), SpatError> {
    encode_ambisonic(order, z_front, -x_right, y_up, distance, out)
}

fn encode_ambisonic(
    order: usize,
    x_front: f32,
    y_left: f32,
    z_up: f32,
    distance: f32,
    out: &mut [f32],
) -> Result<(), SpatError> {
    if order == 0 || order > max_order() {
        return Err(SpatError::new("ambisonic order is not enabled"));
    }
    let nch = channel_count(order);
    if out.len() < nch {
        return Err(SpatError::new("encode buffer is short"));
    }
    for slot in out.iter_mut().take(nch) {
        *slot = 0.0;
    }
    let len = libm::sqrtf(x_front * x_front + y_left * y_left + z_up * z_up).max(1.0e-8);
    let x = (x_front / len) as f64;
    let y = (y_left / len) as f64;
    let z = (z_up / len) as f64;
    let az = libm::atan2(y, x);
    let el = libm::asin(z.clamp(-1.0, 1.0));
    let gain = 1.0 / distance.max(1.0e-3) as f64;
    let mut idx = 0;
    for n in 0..=order as i32 {
        for m in -n..=n {
            out[idx] = (gain * real_sh_sn3d(n, m, az, el)) as f32;
            idx += 1;
        }
    }
    Ok(())
}

fn real_sh_sn3d(n: i32, m: i32, az: f64, el: f64) -> f64 {
    let z = libm::sin(el);
    let p = assoc_legendre(n, m.abs(), z);
    let norm = sn3d_norm(n, m.abs());
    let trig = if m > 0 {
        libm::cos(m as f64 * az)
    } else if m < 0 {
        libm::sin((-m) as f64 * az)
    } else {
        1.0
    };
    norm * p * trig
}

/// Associated Legendre P_n^|m| without the Condon-Shortley phase, so order-1
/// components match the positive AmbiX cartesian axes.
fn assoc_legendre(n: i32, m: i32, x: f64) -> f64 {
    let m = m.abs();
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = libm::sqrt((1.0 - x) * (1.0 + x));
        let mut fact = 1.0;
        for _ in 1..=m {
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }
    if n == m {
        return pmm;
    }
    let mut pmmp1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if n == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m + 2)..=n {
        pll = (x * (2.0 * ll as f64 - 1.0) * pmmp1 - (ll as f64 + m as f64 - 1.0) * pmm)
            / (ll as f64 - m as f64);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn sn3d_norm(n: i32, m: i32) -> f64 {
    let m = m.abs();
    let mut fact = 1.0;
    let mut i = n - m + 1;
    while i <= n + m {
        fact *= i as f64;
        i += 1;
    }
    let mut nrm = libm::sqrt(1.0 / fact);
    if m > 0 {
        nrm *= core::f64::consts::SQRT_2;
    }
    nrm
}

/// Rotate HOA coefficients by listener yaw, pitch, and roll (radians).
/// Yaw is about the up axis, pitch about the left axis, roll about the front axis.
pub fn rotate(order: usize, yaw: f32, pitch: f32, roll: f32, coeffs: &mut [f32]) -> Result<(), SpatError> {
    if order == 0 || order > max_order() {
        return Err(SpatError::new("ambisonic order is not enabled"));
    }
    if coeffs.len() < channel_count(order) {
        return Err(SpatError::new("rotate buffer is short"));
    }
    if yaw != 0.0 {
        rotate_z(order, yaw, coeffs);
    }
    if pitch != 0.0 {
        rotate_y(order, pitch, coeffs);
    }
    if roll != 0.0 {
        rotate_x(order, roll, coeffs);
    }
    Ok(())
}

fn rotate_z(order: usize, angle: f32, coeffs: &mut [f32]) {
    for n in 1..=order as i32 {
        for m in 1..=n {
            let ia = acn_index(n, m);
            let ib = acn_index(n, -m);
            let cs = libm::cosf(m as f32 * angle);
            let sn = libm::sinf(m as f32 * angle);
            let pos = coeffs[ia];
            let neg = coeffs[ib];
            coeffs[ia] = pos * cs - neg * sn;
            coeffs[ib] = neg * cs + pos * sn;
        }
    }
}

fn rotate_y(order: usize, beta: f32, coeffs: &mut [f32]) {
    for n in 1..=order as i32 {
        let mut re = [0.0f64; 16];
        let mut im = [0.0f64; 16];
        let s2 = core::f64::consts::SQRT_2;
        for m in 0..=n {
            if m == 0 {
                re[n as usize] = coeffs[acn_index(n, 0)] as f64;
            } else {
                let rp = coeffs[acn_index(n, m)] as f64;
                let rn = coeffs[acn_index(n, -m)] as f64;
                re[(n + m) as usize] = rp / s2;
                im[(n + m) as usize] = -rn / s2;
                re[(n - m) as usize] = rp / s2;
                im[(n - m) as usize] = rn / s2;
            }
        }
        let mut ore = [0.0f64; 16];
        let mut oim = [0.0f64; 16];
        for mp in -n..=n {
            let mut ar = 0.0;
            let mut ai = 0.0;
            for m in -n..=n {
                let d = wigner_d(n, mp, m, beta as f64);
                let br = re[(n + m) as usize];
                let bi = im[(n + m) as usize];
                ar += d * br;
                ai += d * bi;
            }
            ore[(n + mp) as usize] = ar;
            oim[(n + mp) as usize] = ai;
        }
        coeffs[acn_index(n, 0)] = ore[n as usize] as f32;
        for m in 1..=n {
            let cr = ore[(n + m) as usize];
            let ci = oim[(n + m) as usize];
            coeffs[acn_index(n, m)] = (s2 * cr) as f32;
            coeffs[acn_index(n, -m)] = (-s2 * ci) as f32;
        }
    }
}

fn rotate_x(order: usize, angle: f32, coeffs: &mut [f32]) {
    let half = core::f32::consts::FRAC_PI_2;
    rotate_z(order, half, coeffs);
    rotate_y(order, angle, coeffs);
    rotate_z(order, -half, coeffs);
}

fn wigner_d(j: i32, mp: i32, m: i32, beta: f64) -> f64 {
    let s_min = 0.max(m - mp);
    let s_max = (j + m).min(j - mp);
    if s_min > s_max {
        return 0.0;
    }
    let half = beta * 0.5;
    let c = libm::cos(half);
    let s = libm::sin(half);
    let pref = libm::sqrt(factorial(j + mp) * factorial(j - mp) * factorial(j + m) * factorial(j - m));
    let mut sum = 0.0;
    for k in s_min..=s_max {
        let den = factorial(k)
            * factorial(j + m - k)
            * factorial(mp - m + k)
            * factorial(j - mp - k);
        if den == 0.0 {
            continue;
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let p_cos = 2 * j + m - mp - 2 * k;
        let p_sin = mp - m + 2 * k;
        sum += sign / den * powi_f64(c, p_cos) * powi_f64(s, p_sin);
    }
    pref * sum
}

fn factorial(n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    let mut f = 1.0;
    for i in 2..=n {
        f *= i as f64;
    }
    f
}

fn powi_f64(base: f64, exp: i32) -> f64 {
    if exp == 0 {
        return 1.0;
    }
    if exp < 0 {
        let p = powi_f64(base, -exp);
        if p == 0.0 {
            0.0
        } else {
            1.0 / p
        }
    } else {
        let mut a = 1.0;
        for _ in 0..exp {
            a *= base;
        }
        a
    }
}

#[derive(Clone, Debug)]
pub struct Decoder {
    matrix: Vec<f32>,
    speakers: usize,
    channels: usize,
    layout: SpeakerLayout,
}

impl Decoder {
    pub fn new(layout: SpeakerLayout, order: usize) -> Result<Self, SpatError> {
        let order = layout.decode_order(order.min(max_order()));
        let speakers = speaker_dirs(layout);
        let channels = channel_count(order);
        let nspk = speakers.len();
        let mut encoding = vec![0.0f32; nspk * channels];
        for (s, (az, el)) in speakers.iter().enumerate() {
            encode_aed(order, *az, *el, 1.0, &mut encoding[s * channels..(s + 1) * channels])?;
        }
        let mut gram = vec![0.0f32; nspk * nspk];
        for r in 0..nspk {
            for c in 0..nspk {
                let mut dot = 0.0;
                for k in 0..channels {
                    dot += encoding[r * channels + k] * encoding[c * channels + k];
                }
                gram[r * nspk + c] = dot;
            }
            gram[r * nspk + r] += 1.0e-4;
        }
        if !invert_matrix(nspk, &mut gram) {
            return Err(SpatError::new("speaker matrix is singular"));
        }
        let mut matrix = vec![0.0f32; nspk * channels];
        for r in 0..nspk {
            for c in 0..channels {
                let mut acc = 0.0;
                for k in 0..nspk {
                    acc += gram[r * nspk + k] * encoding[k * channels + c];
                }
                matrix[r * channels + c] = acc;
            }
        }
        Ok(Self {
            matrix,
            speakers: nspk,
            channels,
            layout,
        })
    }

    pub fn layout(&self) -> SpeakerLayout {
        self.layout
    }

    pub fn speakers(&self) -> usize {
        self.speakers
    }

    pub fn process(&self, hoa: &[f32], out: &mut [f32]) {
        for s in 0..self.speakers {
            if s >= out.len() {
                break;
            }
            let mut acc = 0.0;
            for c in 0..self.channels {
                let h = if c < hoa.len() { hoa[c] } else { 0.0 };
                acc += self.matrix[s * self.channels + c] * h;
            }
            out[s] = acc;
        }
    }
}

fn speaker_dirs(layout: SpeakerLayout) -> Vec<(f32, f32)> {
    let d30 = core::f32::consts::FRAC_PI_6;
    let d45 = core::f32::consts::FRAC_PI_4;
    let d135 = 3.0 * d45;
    match layout {
        SpeakerLayout::Stereo => vec![(d30, 0.0), (-d30, 0.0)],
        SpeakerLayout::Square => vec![(d45, 0.0), (-d45, 0.0), (-d135, 0.0), (d135, 0.0)],
        SpeakerLayout::Cube => {
            let el = libm::atanf(1.0 / libm::sqrtf(2.0));
            vec![
                (d45, -el),
                (-d45, -el),
                (-d135, -el),
                (d135, -el),
                (d45, el),
                (-d45, el),
                (-d135, el),
                (d135, el),
            ]
        }
    }
}

fn invert_matrix(n: usize, a: &mut [f32]) -> bool {
    let mut aug = vec![0.0f32; n * n * 2];
    for r in 0..n {
        for c in 0..n {
            aug[r * (2 * n) + c] = a[r * n + c];
        }
        aug[r * (2 * n) + n + r] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col * (2 * n) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (2 * n) + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1.0e-8 {
            return false;
        }
        if pivot != col {
            for c in 0..(2 * n) {
                let i0 = col * (2 * n) + c;
                let i1 = pivot * (2 * n) + c;
                aug.swap(i0, i1);
            }
        }
        let div = aug[col * (2 * n) + col];
        for c in 0..(2 * n) {
            aug[col * (2 * n) + c] /= div;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * (2 * n) + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                let v = aug[col * (2 * n) + c];
                aug[r * (2 * n) + c] -= factor * v;
            }
        }
    }
    for r in 0..n {
        for c in 0..n {
            a[r * n + c] = aug[r * (2 * n) + n + c];
        }
    }
    true
}

const HRIR_LEN: usize = 32;
const FFT_N: usize = 64;
const HOP: usize = 32;

#[derive(Clone, Copy)]
struct Cx {
    re: f32,
    im: f32,
}

fn fft(buf: &mut [Cx], inverse: bool) {
    let n = buf.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = if inverse {
            core::f32::consts::TAU / len as f32
        } else {
            -core::f32::consts::TAU / len as f32
        };
        let wlen_re = libm::cosf(ang);
        let wlen_im = libm::sinf(ang);
        let mut i = 0;
        while i < n {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for k in 0..(len / 2) {
                let u_re = buf[i + k].re;
                let u_im = buf[i + k].im;
                let v_re = buf[i + k + len / 2].re;
                let v_im = buf[i + k + len / 2].im;
                let tr = wr * v_re - wi * v_im;
                let ti = wr * v_im + wi * v_re;
                buf[i + k].re = u_re + tr;
                buf[i + k].im = u_im + ti;
                buf[i + k + len / 2].re = u_re - tr;
                buf[i + k + len / 2].im = u_im - ti;
                let nwr = wr * wlen_re - wi * wlen_im;
                wi = wr * wlen_im + wi * wlen_re;
                wr = nwr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f32;
        for c in buf.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }
}

/// Synthetic short HRIR. Positive azimuth is left; `ear` +1 is the left ear.
pub fn synth_hrir(azimuth: f32, elevation: f32, ear: f32, sample_rate: f32) -> [f32; HRIR_LEN] {
    let head = 0.0875_f32;
    let speed = 343.0_f32;
    let itd = (head / speed) * (azimuth + libm::sinf(azimuth));
    let shift = ear * 0.5 * itd * sample_rate;
    let delay = (12.0 - shift).clamp(4.0, (HRIR_LEN as f32) - 5.0);
    let ild = (0.55 + 0.45 * libm::sinf(azimuth * ear)).clamp(0.15, 1.2);
    let el_gain = 1.0 - 0.12 * elevation.abs();
    let mut ir = [0.0f32; HRIR_LEN];
    for n in 0..HRIR_LEN {
        let x = n as f32 - delay;
        let sinc = if x.abs() < 1.0e-5 {
            1.0
        } else {
            libm::sinf(core::f32::consts::PI * x) / (core::f32::consts::PI * x)
        };
        let win = 0.5 - 0.5 * libm::cosf(core::f32::consts::TAU * n as f32 / (HRIR_LEN as f32 - 1.0));
        ir[n] = sinc * win * ild * el_gain;
    }
    ir
}

struct Binaural {
    decoder: Decoder,
    /// Packed spectra: speaker-major, then ear, then FFT bin. Allocated in configure.
    hrir_fft: Vec<Cx>,
    scratch: Vec<Cx>,
    hist: Vec<f32>,
    hist_pos: usize,
    filled: usize,
    overlap_l: Vec<f32>,
    overlap_r: Vec<f32>,
    spk_frame: Vec<f32>,
    /// One hop of stereo samples waiting to be emitted, allocated in `prepare`.
    pending_l: Vec<f32>,
    pending_r: Vec<f32>,
    pending_read: usize,
    pending_count: usize,
}

impl Binaural {
    fn prepare(&mut self, sample_rate: u32, order: usize) -> Result<(), SpatError> {
        self.decoder = Decoder::new(SpeakerLayout::Cube, order)?;
        let nspk = self.decoder.speakers();
        let dirs = speaker_dirs(SpeakerLayout::Cube);
        self.hrir_fft = Vec::with_capacity(nspk * 2 * FFT_N);
        let sr = sample_rate as f32;
        for (az, el) in &dirs {
            for ear in [1.0_f32, -1.0] {
                let ir = synth_hrir(*az, *el, ear, sr);
                let mut spec = Vec::with_capacity(FFT_N);
                for bin in 0..FFT_N {
                    if bin < HRIR_LEN {
                        spec.push(Cx { re: ir[bin], im: 0.0 });
                    } else {
                        spec.push(Cx { re: 0.0, im: 0.0 });
                    }
                }
                fft(&mut spec, false);
                self.hrir_fft.extend(spec);
            }
        }
        self.scratch = alloc::vec![Cx { re: 0.0, im: 0.0 }; FFT_N];
        self.hist = alloc::vec![0.0; nspk * FFT_N];
        self.overlap_l = alloc::vec![0.0; nspk * HOP];
        self.overlap_r = alloc::vec![0.0; nspk * HOP];
        self.spk_frame = alloc::vec![0.0; nspk];
        self.pending_l = alloc::vec![0.0; HOP];
        self.pending_r = alloc::vec![0.0; HOP];
        self.pending_read = 0;
        self.pending_count = 0;
        self.hist_pos = 0;
        self.filled = 0;
        Ok(())
    }

    fn push_hop(&mut self, ear_l: &[f32; HOP], ear_r: &[f32; HOP]) {
        for k in 0..HOP {
            if self.pending_count >= HOP {
                break;
            }
            let idx = (self.pending_read + self.pending_count) % HOP;
            self.pending_l[idx] = ear_l[k];
            self.pending_r[idx] = ear_r[k];
            self.pending_count += 1;
        }
    }

    fn pop_ear(&mut self) -> (f32, f32) {
        if self.pending_count == 0 {
            return (0.0, 0.0);
        }
        let idx = self.pending_read;
        let sample = (self.pending_l[idx], self.pending_r[idx]);
        self.pending_read = (self.pending_read + 1) % HOP;
        self.pending_count -= 1;
        sample
    }
}

pub struct SpatNote {
    pub start: f64,
    pub dur: f64,
    pub azimuth: Pfield,
    pub elevation: Pfield,
    pub distance: Pfield,
}

pub struct RotateNote {
    pub start: f64,
    pub dur: f64,
    pub yaw: Pfield,
    pub pitch: Pfield,
    pub roll: Pfield,
}

pub struct DecodeNote {
    pub start: f64,
    pub dur: f64,
    pub layout: SpeakerLayout,
}

pub struct BinauralNote {
    pub start: f64,
    pub dur: f64,
}

struct SpatInst {
    azimuth: Pfield,
    elevation: Pfield,
    distance: Pfield,
}

struct RotateInst {
    yaw: Pfield,
    pitch: Pfield,
    roll: Pfield,
    published: bool,
    /// Output channel is the row, input channel is the column. Built once per pose.
    matrix: Vec<f32>,
    stride: usize,
    cached_yaw: f32,
    cached_pitch: f32,
    cached_roll: f32,
    ready: bool,
}

struct DecodeInst {
    decoder: Option<Decoder>,
    layout: SpeakerLayout,
}

struct BinauralInst {
    state: Option<Binaural>,
}

pub fn schedule_spat(engine: &mut Engine, note: SpatNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "SPAT",
        note.start,
        note.dur,
        Box::new(SpatInst {
            azimuth: note.azimuth,
            elevation: note.elevation,
            distance: note.distance,
        }),
    )
}

pub fn schedule_rotate(engine: &mut Engine, note: RotateNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "ROTATE",
        note.start,
        note.dur,
        Box::new(RotateInst {
            yaw: note.yaw,
            pitch: note.pitch,
            roll: note.roll,
            published: false,
            matrix: Vec::new(),
            stride: 0,
            cached_yaw: 0.0,
            cached_pitch: 0.0,
            cached_roll: 0.0,
            ready: false,
        }),
    )
}

pub fn schedule_decode(engine: &mut Engine, note: DecodeNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "DECODE",
        note.start,
        note.dur,
        Box::new(DecodeInst {
            decoder: None,
            layout: note.layout,
        }),
    )
}

pub fn schedule_binaural(engine: &mut Engine, note: BinauralNote) -> Result<(), CoreError> {
    engine.schedule_named(
        "BINAURAL",
        note.start,
        note.dur,
        Box::new(BinauralInst { state: None }),
    )
}

impl Instrument for SpatInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {}
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let order = (ctx.order as usize).min(max_order()).max(1);
        let nch = channel_count(order).min(ctx.buses.channels(ctx.out_bus));
        let phase = ctx.note_phase(ctx.frame_offset);
        let mut gains = [0.0f32; 36];
        if encode_aed(
            order,
            self.azimuth.at(phase),
            self.elevation.at(phase),
            self.distance.at(phase).max(0.05),
            &mut gains[..channel_count(order)],
        )
        .is_err()
        {
            return;
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            let x = ctx.buses.sample(ctx.in_bus, 0, frame);
            for ch in 0..nch {
                ctx.buses.add(ctx.out_bus, ch, frame, x * gains[ch]);
            }
        }
    }
}

impl RotateInst {
    fn rebuild(&mut self, order: usize, yaw: f32, pitch: f32, roll: f32) {
        let n = channel_count(order);
        if self.matrix.len() < n * n || n > 36 {
            return;
        }
        self.stride = n;
        for basis in 0..n {
            let mut impulse = [0.0f32; 36];
            impulse[basis] = 1.0;
            if rotate(order, yaw, pitch, roll, &mut impulse[..n]).is_err() {
                return;
            }
            for row in 0..n {
                self.matrix[row * n + basis] = impulse[row];
            }
        }
        self.cached_yaw = yaw;
        self.cached_pitch = pitch;
        self.cached_roll = roll;
        self.ready = true;
    }
}

impl Instrument for RotateInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, _: &mut SetupCtx<'_>) {
        let n = channel_count(max_order());
        self.matrix = alloc::vec![0.0; n * n];
        self.ready = false;
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let order = (ctx.order as usize).min(max_order()).max(1);
        let nch = channel_count(order).min(ctx.buses.channels(ctx.in_bus));
        let phase = ctx.note_phase(ctx.frame_offset);
        let dynamic = matches!(self.yaw, Pfield::Ramp { .. })
            || matches!(self.pitch, Pfield::Ramp { .. })
            || matches!(self.roll, Pfield::Ramp { .. });
        if !self.published || dynamic {
            ctx.pose.set_angles(
                self.yaw.at(phase),
                self.pitch.at(phase),
                self.roll.at(phase),
            );
            self.published = true;
        }
        let pose = ctx.pose.load();
        if !self.ready
            || pose.yaw != self.cached_yaw
            || pose.pitch != self.cached_pitch
            || pose.roll != self.cached_roll
            || self.stride != nch
        {
            self.rebuild(order, pose.yaw, pose.pitch, pose.roll);
        }
        if !self.ready || self.stride != nch {
            return;
        }
        let mut sample = [0.0f32; 36];
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            for ch in 0..nch {
                sample[ch] = ctx.buses.sample(ctx.in_bus, ch, frame);
            }
            for row in 0..nch {
                let mut acc = 0.0;
                let base = row * self.stride;
                for col in 0..nch {
                    acc += self.matrix[base + col] * sample[col];
                }
                ctx.buses.set(ctx.out_bus, row, frame, acc);
            }
        }
    }
}

impl Instrument for DecodeInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        self.decoder = Decoder::new(self.layout, ctx.order as usize).ok();
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let Some(decoder) = self.decoder.as_ref() else {
            return;
        };
        let mut hoa = [0.0f32; 36];
        let mut spk = [0.0f32; 8];
        let nch = decoder.channels.min(36);
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            for ch in 0..nch {
                hoa[ch] = ctx.buses.sample(ctx.in_bus, ch, frame);
            }
            decoder.process(&hoa[..nch], &mut spk);
            let outs = ctx.buses.channels(ctx.out_bus).min(decoder.speakers());
            for s in 0..outs {
                ctx.buses.add(ctx.out_bus, s, frame, spk[s]);
            }
        }
    }
}

impl Instrument for BinauralInst {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}
    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        let mut state = Binaural {
            decoder: Decoder {
                matrix: Vec::new(),
                speakers: 0,
                channels: 0,
                layout: SpeakerLayout::Cube,
            },
            hrir_fft: Vec::new(),
            scratch: Vec::new(),
            hist: Vec::new(),
            hist_pos: 0,
            filled: 0,
            overlap_l: Vec::new(),
            overlap_r: Vec::new(),
            spk_frame: Vec::new(),
            pending_l: Vec::new(),
            pending_r: Vec::new(),
            pending_read: 0,
            pending_count: 0,
        };
        if state.prepare(ctx.sample_rate, ctx.order as usize).is_ok() {
            self.state = Some(state);
        }
    }
    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let nspk = state.decoder.speakers;
        let hoa_n = state.decoder.channels;
        if nspk == 0 || state.scratch.len() != FFT_N {
            return;
        }
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            for s in 0..nspk {
                let mut acc = 0.0;
                let row = s * hoa_n;
                for ch in 0..hoa_n {
                    acc += state.decoder.matrix[row + ch] * ctx.buses.sample(ctx.in_bus, ch, frame);
                }
                let slot = s * FFT_N + state.hist_pos;
                state.hist[slot] = acc;
            }
            state.hist_pos = (state.hist_pos + 1) % FFT_N;
            state.filled += 1;
            if state.filled >= HOP {
                state.filled = 0;
                let mut ear_l = [0.0f32; HOP];
                let mut ear_r = [0.0f32; HOP];
                for s in 0..nspk {
                    for ear in 0..2 {
                        for bin in 0..FFT_N {
                            let sample = if bin < HOP {
                                let idx = (state.hist_pos + FFT_N - HOP + bin) % FFT_N;
                                state.hist[s * FFT_N + idx]
                            } else {
                                0.0
                            };
                            state.scratch[bin].re = sample;
                            state.scratch[bin].im = 0.0;
                        }
                        fft(&mut state.scratch, false);
                        let base = (s * 2 + ear) * FFT_N;
                        for bin in 0..FFT_N {
                            let a = &state.scratch[bin];
                            let b = &state.hrir_fft[base + bin];
                            let re = a.re * b.re - a.im * b.im;
                            let im = a.re * b.im + a.im * b.re;
                            state.scratch[bin].re = re;
                            state.scratch[bin].im = im;
                        }
                        fft(&mut state.scratch, true);
                        let overlap = if ear == 0 {
                            &mut state.overlap_l[s * HOP..(s + 1) * HOP]
                        } else {
                            &mut state.overlap_r[s * HOP..(s + 1) * HOP]
                        };
                        for k in 0..HOP {
                            let y = state.scratch[k].re + overlap[k];
                            overlap[k] = state.scratch[k + HOP].re;
                            if ear == 0 {
                                ear_l[k] += y;
                            } else {
                                ear_r[k] += y;
                            }
                        }
                    }
                }
                state.push_hop(&ear_l, &ear_r);
            }
            let (left, right) = state.pop_ear();
            if ctx.buses.channels(ctx.out_bus) >= 1 {
                ctx.buses.add(ctx.out_bus, 0, frame, left);
            }
            if ctx.buses.channels(ctx.out_bus) >= 2 {
                ctx.buses.add(ctx.out_bus, 1, frame, right);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_x_decodes_to_the_right() {
        let mut hoa = [0.0f32; 4];
        encode_cartesian(1, 1.0, 0.0, 0.0, 1.0, &mut hoa).unwrap();
        assert!(hoa[1] < -0.9, "Y (left) should be negative for +X, got {}", hoa[1]);
        let decoder = Decoder::new(SpeakerLayout::Stereo, 1).unwrap();
        let mut speakers = [0.0f32; 2];
        decoder.process(&hoa, &mut speakers);
        assert!(
            speakers[1] > speakers[0],
            "right {} should exceed left {}",
            speakers[1],
            speakers[0]
        );
    }

    #[test]
    fn rotation_by_360_degrees_returns_to_start() {
        let order = max_order().min(3);
        let n = channel_count(order);
        let mut coeffs = vec![0.0f32; n];
        encode_aed(order, 0.4, 0.25, 1.0, &mut coeffs).unwrap();
        let original = coeffs.clone();
        let full = core::f32::consts::TAU;
        rotate(order, full, 0.0, 0.0, &mut coeffs).unwrap();
        for (a, b) in original.iter().zip(coeffs.iter()) {
            assert!((a - b).abs() < 1.0e-4, "{a} vs {b}");
        }
        let mut coeffs = original.clone();
        rotate(order, 0.0, full, 0.0, &mut coeffs).unwrap();
        for (a, b) in original.iter().zip(coeffs.iter()) {
            assert!((a - b).abs() < 1.0e-3, "pitch {a} vs {b}");
        }
        let mut coeffs = original.clone();
        rotate(order, 0.0, 0.0, full, &mut coeffs).unwrap();
        for (a, b) in original.iter().zip(coeffs.iter()) {
            assert!((a - b).abs() < 1.0e-3, "roll {a} vs {b}");
        }
    }

    #[test]
    fn order1_encode_energy_is_stable() {
        let mut energies = [0.0f32; 24];
        let mut n = 0;
        for step in 0..8 {
            for el_step in 0..3 {
                let az = step as f32 * 0.7;
                let el = el_step as f32 * 0.4 - 0.4;
                let mut c = [0.0f32; 4];
                encode_aed(1, az, el, 1.0, &mut c).unwrap();
                energies[n] = c.iter().map(|v| v * v).sum();
                n += 1;
            }
        }
        let reference = energies[0];
        assert!((reference - 2.0).abs() < 1.0e-3, "expected energy 2, got {reference}");
        for e in energies.iter().take(n) {
            assert!((e - reference).abs() < 1.0e-4, "{e} vs {reference}");
        }
    }

    #[test]
    fn wigner_d_matches_order1_table() {
        let beta = 0.35_f64;
        assert!((wigner_d(1, 0, 0, beta) - beta.cos()).abs() < 1.0e-5);
        assert!((wigner_d(1, 1, 1, beta) - (1.0 + beta.cos()) / 2.0).abs() < 1.0e-5);
        let off = wigner_d(1, 1, 0, beta);
        let mag = beta.sin() / core::f64::consts::SQRT_2;
        assert!((off.abs() - mag.abs()).abs() < 1.0e-5);
    }

    #[test]
    fn yaw_matches_reencoding() {
        let order = 1;
        let az = 0.3_f32;
        let el = -0.2_f32;
        let mut a = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        encode_aed(order, az, el, 1.0, &mut a).unwrap();
        let delta = 0.55_f32;
        rotate(order, delta, 0.0, 0.0, &mut a).unwrap();
        encode_aed(order, az + delta, el, 1.0, &mut b).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1.0e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn fft_overlap_add_delays_an_impulse() {
        let mut ir = [0.0f32; HRIR_LEN];
        ir[5] = 1.0;
        let mut spec = vec![Cx { re: 0.0, im: 0.0 }; FFT_N];
        for i in 0..HRIR_LEN {
            spec[i].re = ir[i];
        }
        fft(&mut spec, false);
        let mut time = vec![Cx { re: 0.0, im: 0.0 }; FFT_N];
        time[0].re = 1.0;
        fft(&mut time, false);
        for i in 0..FFT_N {
            let re = time[i].re * spec[i].re - time[i].im * spec[i].im;
            let im = time[i].re * spec[i].im + time[i].im * spec[i].re;
            time[i].re = re;
            time[i].im = im;
        }
        fft(&mut time, true);
        let mut peak = 0usize;
        let mut best = 0.0f32;
        for (i, c) in time.iter().enumerate() {
            if c.re.abs() > best {
                best = c.re.abs();
                peak = i;
            }
        }
        assert_eq!(peak, 5);
        assert!((time[5].re - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn binaural_emits_every_sample_of_the_hop() {
        struct ConstHoa;
        impl Instrument for ConstHoa {
            fn init(&mut self, _: &mut SetupCtx<'_>) {}
            fn configure(&mut self, _: &mut SetupCtx<'_>) {}
            fn run(&mut self, ctx: &mut RunCtx<'_>) {
                for i in 0..ctx.frame_count {
                    ctx.buses.add(ctx.out_bus, 0, ctx.frame_offset + i, 0.5);
                }
            }
        }

        let mut engine = Engine::new(48_000, 2, 256);
        engine.set_order(1).unwrap();
        engine
            .schedule_named("SPAT", 0.0, 1.0, Box::new(ConstHoa))
            .unwrap();
        schedule_binaural(
            &mut engine,
            BinauralNote {
                start: 0.0,
                dur: 1.0,
            },
        )
        .unwrap();
        engine.process_block();
        let out = engine.bus_id("out").unwrap();
        let left = engine.bus_plane(out, 0);
        let nonzero = left.iter().filter(|s| s.abs() > 1.0e-6).count();
        assert!(
            nonzero > HOP,
            "expected the whole hop, got {nonzero} samples in a block of {}",
            left.len()
        );
    }

    #[test]
    fn hrir_is_louder_on_the_near_ear() {
        let sr = 48_000.0;
        let left = synth_hrir(1.0, 0.0, 1.0, sr);
        let right = synth_hrir(1.0, 0.0, -1.0, sr);
        let el: f32 = left.iter().map(|s| s * s).sum();
        let er: f32 = right.iter().map(|s| s * s).sum();
        assert!(el > er);
    }
}
