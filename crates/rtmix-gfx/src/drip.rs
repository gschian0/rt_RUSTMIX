//! Audio-reactive Gray-Scott reaction-diffusion fields.

use crate::analysis::Spectrum;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DripMode {
    Acid,
    Ink,
    Lava,
}

pub struct DripView {
    width: usize,
    height: usize,
    u: Vec<f32>,
    v: Vec<f32>,
    next_u: Vec<f32>,
    next_v: Vec<f32>,
    phase: f32,
}

impl DripView {
    pub fn new(width: usize, height: usize) -> Self {
        let width = width.max(32);
        let height = height.max(32);
        let len = width * height;
        let mut view = Self {
            width,
            height,
            u: vec![1.0; len],
            v: vec![0.0; len],
            next_u: vec![0.0; len],
            next_v: vec![0.0; len],
            phase: 0.0,
        };
        view.seed();
        view
    }

    pub fn size(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    pub fn render(&mut self, mode: DripMode, spectrum: &Spectrum, delta: f32, rgba: &mut Vec<u8>) {
        self.phase += delta.clamp(0.0, 0.1) * (0.4 + spectrum.level * 2.0);
        self.inject(spectrum);
        let steps = if spectrum.flux > 0.35 { 4 } else { 3 };
        for _ in 0..steps {
            self.step(mode, spectrum);
        }

        rgba.resize(self.width * self.height * 4, 255);
        for (index, out) in rgba.chunks_exact_mut(4).enumerate() {
            let x = index % self.width;
            let y = index / self.width;
            let (sample_x, sample_y, dimensional_warp) =
                self.warp_coordinates(x, y, mode, spectrum);
            let sample = self.index(sample_x, sample_y);
            let chemical = (self.v[sample] * 1.8 - self.u[sample] * 0.28).clamp(0.0, 1.0);
            let edge = (self.v[sample] - self.v[self.index(sample_x + 1, sample_y)]).abs()
                + (self.v[sample] - self.v[self.index(sample_x, sample_y + 1)]).abs();
            let glow = (chemical + edge * 2.5 + spectrum.flux * 0.08 + dimensional_warp * 0.12)
                .clamp(0.0, 1.0);
            let color = warp_color(
                palette(mode, glow, spectrum.centroid),
                dimensional_warp + self.phase * 0.7,
            );
            out[0] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
            out[1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
            out[2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
            out[3] = 255;
        }
    }

    /// Fold a 2D field through a moving pseudo-3D coordinate transform before
    /// sampling it. The z-like term affects both depth glow and hue rotation.
    fn warp_coordinates(
        &self,
        x: usize,
        y: usize,
        mode: DripMode,
        spectrum: &Spectrum,
    ) -> (usize, usize, f32) {
        let nx = x as f32 / (self.width - 1) as f32 * 2.0 - 1.0;
        let ny = y as f32 / (self.height - 1) as f32 * 2.0 - 1.0;
        let radius = (nx * nx + ny * ny).sqrt();
        let base_angle = ny.atan2(nx);
        let twist = match mode {
            DripMode::Acid => self.phase * 0.8 + spectrum.bands[3] * 2.2,
            DripMode::Ink => (radius * 5.0 + self.phase).sin() * 0.7,
            DripMode::Lava => self.phase * 0.35 + spectrum.bands[9] * 3.0,
        };
        let angle = base_angle + twist * (0.35 + radius * 1.8);
        let fold = match mode {
            DripMode::Acid => 1.0 + spectrum.level * 0.18,
            DripMode::Ink => 0.72 + (self.phase + radius * 4.0).sin().abs() * 0.65,
            DripMode::Lava => 1.0 + (self.phase * 0.5 + radius * 6.0).sin() * 0.12,
        };
        let warped_radius = (radius * fold + (self.phase + angle * 3.0).sin() * 0.08).min(1.45);
        let warped_x = angle.cos() * warped_radius;
        let warped_y = angle.sin() * warped_radius;
        let sample_x = (((warped_x * 0.5 + 0.5) * self.width as f32) as isize)
            .rem_euclid(self.width as isize) as usize;
        let sample_y = (((warped_y * 0.5 + 0.5) * self.height as f32) as isize)
            .rem_euclid(self.height as isize) as usize;
        let z_like = ((angle * 2.0 + self.phase).sin() * 0.5 + 0.5) * (1.0 - radius.min(1.0));
        (sample_x, sample_y, z_like)
    }

    fn seed(&mut self) {
        for seed in 0..18 {
            let cx = (seed * 73 + 31) % self.width;
            let cy = (seed * 47 + 19) % self.height;
            self.paint(cx, cy, 4 + seed % 7, 0.9);
        }
    }

    fn inject(&mut self, spectrum: &Spectrum) {
        let width = self.width;
        let height = self.height;
        for (band, energy) in spectrum.bands.iter().copied().enumerate() {
            if energy < 0.025 {
                continue;
            }
            let drift =
                ((self.phase * (0.7 + band as f32 * 0.09)).sin() * 0.15 + 0.5) * width as f32;
            let x = ((band * width / spectrum.bands.len()) + drift as usize) % width;
            let y = ((band * 29) + (self.phase * 17.0) as usize) % height;
            self.paint(x, y, (2.0 + energy * 7.0) as usize, 0.45 + energy * 0.4);
        }
        if spectrum.flux > 0.12 {
            let x = ((self.phase * 97.0).sin().abs() * (width - 1) as f32) as usize;
            let y = ((self.phase * 61.0).cos().abs() * (height - 1) as f32) as usize;
            self.paint(x, y, (4.0 + spectrum.flux * 10.0) as usize, 1.0);
        }
    }

    fn paint(&mut self, cx: usize, cy: usize, radius: usize, amount: f32) {
        let radius = radius.max(1);
        for oy in 0..=radius * 2 {
            for ox in 0..=radius * 2 {
                let dx = ox as isize - radius as isize;
                let dy = oy as isize - radius as isize;
                if dx * dx + dy * dy > (radius * radius) as isize {
                    continue;
                }
                let x = (cx as isize + dx).rem_euclid(self.width as isize) as usize;
                let y = (cy as isize + dy).rem_euclid(self.height as isize) as usize;
                let index = y * self.width + x;
                self.v[index] = self.v[index].max(amount);
                self.u[index] = (1.0 - amount * 0.6).max(0.0);
            }
        }
    }

    fn step(&mut self, mode: DripMode, spectrum: &Spectrum) {
        let (feed, kill) = match mode {
            DripMode::Acid => (0.032, 0.061),
            DripMode::Ink => (0.022, 0.052),
            DripMode::Lava => (0.046, 0.064),
        };
        let feed = feed + spectrum.bands[2] * 0.004;
        let kill = kill + spectrum.bands[8] * 0.003;
        for y in 0..self.height {
            for x in 0..self.width {
                let i = y * self.width + x;
                let u = self.u[i];
                let v = self.v[i];
                let lap_u = self.laplacian(&self.u, x, y);
                let lap_v = self.laplacian(&self.v, x, y);
                let reaction = u * v * v;
                self.next_u[i] = (u + 0.16 * lap_u - reaction + feed * (1.0 - u)).clamp(0.0, 1.0);
                self.next_v[i] = (v + 0.08 * lap_v + reaction - (feed + kill) * v).clamp(0.0, 1.0);
            }
        }
        core::mem::swap(&mut self.u, &mut self.next_u);
        core::mem::swap(&mut self.v, &mut self.next_v);
    }

    fn laplacian(&self, field: &[f32], x: usize, y: usize) -> f32 {
        field[self.index(x + 1, y)]
            + field[self.index(x.wrapping_sub(1), y)]
            + field[self.index(x, y + 1)]
            + field[self.index(x, y.wrapping_sub(1))]
            - 4.0 * field[self.index(x, y)]
    }

    fn index(&self, x: usize, y: usize) -> usize {
        (y % self.height) * self.width + x % self.width
    }
}

fn palette(mode: DripMode, value: f32, centroid: f32) -> [f32; 3] {
    let heat = (centroid / 5_000.0).clamp(0.0, 1.0);
    match mode {
        DripMode::Acid => [
            value * (0.3 + heat),
            value.sqrt(),
            (1.0 - value) * 0.2 + value * 0.75,
        ],
        DripMode::Ink => [
            0.03 + value * 0.3,
            0.02 + value * (0.25 + heat * 0.35),
            0.06 + value.powf(0.7),
        ],
        DripMode::Lava => [
            value.sqrt(),
            value * value * (0.45 + heat * 0.4),
            value.powi(4) * 0.25,
        ],
    }
}

fn warp_color(mut color: [f32; 3], phase: f32) -> [f32; 3] {
    let amount = phase.sin() * 0.5 + 0.5;
    let rotated = [color[1], color[2], color[0]];
    for channel in 0..3 {
        color[channel] = color[channel] * (1.0 - amount) + rotated[channel] * amount;
    }
    let contrast = 0.82 + phase.cos().abs() * 0.42;
    for channel in &mut color {
        *channel = (*channel * contrast).powf(0.86);
    }
    color
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaction_field_renders_and_changes() {
        let mut view = DripView::new(64, 48);
        let mut first = Vec::new();
        let mut second = Vec::new();
        view.render(DripMode::Acid, &Spectrum::default(), 1.0 / 60.0, &mut first);
        let mut spectrum = Spectrum::default();
        spectrum.bands = [1.0; 12];
        spectrum.flux = 1.0;
        view.render(DripMode::Acid, &spectrum, 1.0 / 60.0, &mut second);
        assert_eq!(view.size(), [64, 48]);
        assert_eq!(second.len(), 64 * 48 * 4);
        assert_ne!(first, second);
    }
}
