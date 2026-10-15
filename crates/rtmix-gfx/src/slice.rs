//! Conditional slice of a 4D Gaussian at score time t.

#[derive(Clone, Debug)]
pub struct Gaussian3 {
    pub mean: [f32; 3],
    pub cov: [f32; 9],
    pub color: [f32; 3],
    pub opacity: f32,
}

#[derive(Clone, Debug)]
pub struct Gaussian4 {
    pub mean: [f32; 4],
    pub cov: [f32; 16],
    pub color: [f32; 3],
    pub opacity: f32,
}

/// Map a note's local time across the file's `[t0, t1]` range.
pub fn file_time(score_t: f32, note_start: f32, note_dur: f32, t0: f32, t1: f32) -> f32 {
    let u = if note_dur.abs() < 1.0e-8 {
        0.0
    } else {
        ((score_t - note_start) / note_dur).clamp(0.0, 1.0)
    };
    t0 + (t1 - t0) * u
}

/// Condition `(x, y, z)` on `T = t`. Opacity is scaled by the temporal gaussian.
pub fn slice_4d(g: &Gaussian4, t: f32) -> Gaussian3 {
    let dt = t - g.mean[3];
    let var_t = g.cov[15].max(1.0e-8);
    let mut mean = [0.0f32; 3];
    let mut cross = [0.0f32; 3];
    for i in 0..3 {
        cross[i] = g.cov[i * 4 + 3];
        mean[i] = g.mean[i] + cross[i] / var_t * dt;
    }
    let mut cov = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            cov[i * 3 + j] = g.cov[i * 4 + j] - cross[i] * cross[j] / var_t;
        }
    }
    let weight = libm::expf(-0.5 * dt * dt / var_t);
    Gaussian3 {
        mean,
        cov,
        color: g.color,
        opacity: (g.opacity * weight).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(x: f32, y: f32, z: f32, t: f32) -> [f32; 16] {
        let mut cov = [0.0; 16];
        cov[0] = x;
        cov[5] = y;
        cov[10] = z;
        cov[15] = t;
        cov
    }

    #[test]
    fn slice_at_mean_keeps_spatial_mean_and_opacity() {
        let g = Gaussian4 {
            mean: [1.0, 2.0, 3.0, 0.5],
            cov: diagonal(0.2, 0.3, 0.4, 0.25),
            color: [0.2, 0.4, 0.6],
            opacity: 0.8,
        };
        let sliced = slice_4d(&g, 0.5);
        assert!((sliced.mean[0] - 1.0).abs() < 1.0e-5);
        assert!((sliced.mean[1] - 2.0).abs() < 1.0e-5);
        assert!((sliced.mean[2] - 3.0).abs() < 1.0e-5);
        assert!((sliced.opacity - 0.8).abs() < 1.0e-5);
        assert!((sliced.cov[0] - 0.2).abs() < 1.0e-5);
        assert!((sliced.color[2] - 0.6).abs() < 1.0e-6);
    }

    #[test]
    fn slice_shifts_mean_when_time_is_correlated() {
        let mut cov = diagonal(1.0, 1.0, 1.0, 0.5);
        cov[3] = 0.4;
        cov[12] = 0.4;
        let g = Gaussian4 {
            mean: [0.0, 0.0, 0.0, 0.0],
            cov,
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
        };
        let later = slice_4d(&g, 1.0);
        assert!(later.mean[0] > 0.5, "x mean {}", later.mean[0]);
        assert!(later.cov[0] < 1.0);
        assert!(later.opacity < 0.5);
    }

    #[test]
    fn file_time_maps_the_note_across_the_asset_range() {
        let t = file_time(1.5, 1.0, 2.0, 0.0, 10.0);
        assert!((t - 2.5).abs() < 1.0e-5);
    }
}
