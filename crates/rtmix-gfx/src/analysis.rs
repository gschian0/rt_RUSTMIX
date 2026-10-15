//! Spectrum analysis that drives the splat view.
//!
//! This runs on the GUI thread over a copy of the audio scope, so it is free to
//! allocate in `new` and is never called from the audio callback.

pub const BANDS: usize = 12;

/// Lowest and highest analysis frequency, split into `BANDS` log-spaced bins.
const LOW_HZ: f32 = 40.0;
const HIGH_HZ: f32 = 12_000.0;

#[derive(Clone, Copy, Debug)]
pub struct Spectrum {
    /// Smoothed, normalized magnitude per band.
    pub bands: [f32; BANDS],
    /// Broadband RMS level.
    pub level: f32,
    /// Spectral centroid in Hz: how bright the sound is right now.
    pub centroid: f32,
    /// Positive spectral flux: rises on transients and new syllables.
    pub flux: f32,
}

impl Default for Spectrum {
    fn default() -> Self {
        Self {
            bands: [0.0; BANDS],
            level: 0.0,
            centroid: 0.0,
            flux: 0.0,
        }
    }
}

pub struct Analyzer {
    size: usize,
    window: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    magnitude: Vec<f32>,
    previous: Vec<f32>,
    smoothed: [f32; BANDS],
    level: f32,
    centroid: f32,
    flux: f32,
}

impl Analyzer {
    /// `size` must be a power of two.
    pub fn new(size: usize) -> Self {
        let size = size.next_power_of_two().max(64);
        let window = (0..size)
            .map(|i| {
                // Hann window.
                let phase = std::f32::consts::TAU * i as f32 / size as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            size,
            window,
            re: vec![0.0; size],
            im: vec![0.0; size],
            magnitude: vec![0.0; size / 2],
            previous: vec![0.0; size / 2],
            smoothed: [0.0; BANDS],
            level: 0.0,
            centroid: 0.0,
            flux: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn analyze(&mut self, samples: &[f32], sample_rate: f32) -> Spectrum {
        let n = self.size;
        let mut sum_squares = 0.0f32;
        for i in 0..n {
            let s = samples.get(i).copied().unwrap_or(0.0);
            sum_squares += s * s;
            self.re[i] = s * self.window[i];
            self.im[i] = 0.0;
        }
        fft_in_place(&mut self.re, &mut self.im);

        let bins = n / 2;
        let mut flux = 0.0f32;
        let mut weighted = 0.0f32;
        let mut total = 0.0f32;
        for bin in 0..bins {
            let magnitude = (self.re[bin] * self.re[bin] + self.im[bin] * self.im[bin]).sqrt()
                / (n as f32 * 0.25);
            let rise = (magnitude - self.previous[bin]).max(0.0);
            flux += rise;
            self.previous[bin] = magnitude;
            self.magnitude[bin] = magnitude;
            let hz = bin as f32 * sample_rate / n as f32;
            weighted += hz * magnitude;
            total += magnitude;
        }

        let hz_per_bin = sample_rate / n as f32;
        let ratio = (HIGH_HZ / LOW_HZ).powf(1.0 / BANDS as f32);
        let mut edge = LOW_HZ;
        for band in 0..BANDS {
            let next = edge * ratio;
            let first = ((edge / hz_per_bin) as usize).min(bins.saturating_sub(1));
            let last = (((next / hz_per_bin) as usize).max(first + 1)).min(bins);
            let mut peak = 0.0f32;
            for bin in first..last {
                peak = peak.max(self.magnitude[bin]);
            }
            // Perceptual compression, then asymmetric smoothing for a lively
            // attack and a graceful release.
            let shaped = (peak * 14.0).sqrt().min(1.5);
            let rate = if shaped > self.smoothed[band] {
                0.55
            } else {
                0.12
            };
            self.smoothed[band] += (shaped - self.smoothed[band]) * rate;
            edge = next;
        }

        let rms = (sum_squares / n as f32).sqrt();
        let target_level = (rms * 6.0).min(1.5);
        let rate = if target_level > self.level { 0.5 } else { 0.1 };
        self.level += (target_level - self.level) * rate;
        let target_centroid = if total > 1.0e-9 {
            weighted / total
        } else {
            0.0
        };
        self.centroid += (target_centroid - self.centroid) * 0.2;
        let target_flux = (flux * 2.5).min(1.5);
        let flux_rate = if target_flux > self.flux { 0.7 } else { 0.08 };
        self.flux += (target_flux - self.flux) * flux_rate;

        Spectrum {
            bands: self.smoothed,
            level: self.level,
            centroid: self.centroid,
            flux: self.flux,
        }
    }
}

/// Iterative in-place radix-2 Cooley-Tukey FFT.
pub fn fft_in_place(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert_eq!(n, im.len());
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n < 2 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let angle = -std::f32::consts::TAU / len as f32;
        let (wr, wi) = (angle.cos(), angle.sin());
        let mut start = 0usize;
        while start < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for offset in 0..len / 2 {
                let a = start + offset;
                let b = a + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let next_cr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next_cr;
            }
            start += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_finds_a_pure_tone_bin() {
        let n = 1024;
        let sample_rate = 48_000.0f32;
        let freq = 1_500.0f32;
        let mut re: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sample_rate).sin())
            .collect();
        let mut im = vec![0.0f32; n];
        fft_in_place(&mut re, &mut im);
        let mut best = 0usize;
        let mut best_magnitude = 0.0f32;
        for bin in 0..n / 2 {
            let m = (re[bin] * re[bin] + im[bin] * im[bin]).sqrt();
            if m > best_magnitude {
                best_magnitude = m;
                best = bin;
            }
        }
        let measured = best as f32 * sample_rate / n as f32;
        assert!(
            (measured - freq).abs() < sample_rate / n as f32,
            "measured {measured}"
        );
    }

    #[test]
    fn analyzer_puts_energy_in_the_matching_band() {
        let mut analyzer = Analyzer::new(2048);
        let sample_rate = 48_000.0f32;
        let samples: Vec<f32> = (0..2048)
            .map(|i| 0.5 * (std::f32::consts::TAU * 220.0 * i as f32 / sample_rate).sin())
            .collect();
        let mut spectrum = Spectrum::default();
        for _ in 0..24 {
            spectrum = analyzer.analyze(&samples, sample_rate);
        }
        let loudest = spectrum
            .bands
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();
        // 220 Hz sits in the lower third of a 40 Hz..12 kHz log split.
        assert!(loudest <= 4, "loudest band {loudest}");
        assert!(spectrum.level > 0.1, "level {}", spectrum.level);
        assert!(
            spectrum.centroid > 100.0 && spectrum.centroid < 1500.0,
            "centroid {}",
            spectrum.centroid
        );
    }

    #[test]
    fn silence_decays_to_nothing() {
        let mut analyzer = Analyzer::new(512);
        let loud = vec![0.7f32; 512];
        analyzer.analyze(&loud, 48_000.0);
        let quiet = vec![0.0f32; 512];
        let mut spectrum = Spectrum::default();
        for _ in 0..200 {
            spectrum = analyzer.analyze(&quiet, 48_000.0);
        }
        assert!(spectrum.level < 0.01, "level {}", spectrum.level);
        assert!(spectrum.bands.iter().all(|b| *b < 0.01));
    }
}
