//! CPU EWA splat renderer with audio-reactive deformation.
//!
//! Runs on the GUI thread. Scratch buffers are kept on the struct so a steady
//! frame rate does not churn the allocator.

use crate::analysis::{Spectrum, BANDS};
use crate::raster::radix_sort_pairs;
use crate::slice::Gaussian3;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: [0.0; 3],
            distance: 4.0,
            yaw: 0.0,
            pitch: 0.2,
            fov_y: 1.0,
        }
    }
}

/// How strongly the spectrum deforms the cloud.
#[derive(Clone, Copy, Debug)]
pub struct Reactivity {
    pub displace: f32,
    pub swell: f32,
    pub brightness: f32,
    pub spin: f32,
    pub point_size: f32,
    pub exposure: f32,
    pub zoom: f32,
    pub twist: f32,
}

impl Default for Reactivity {
    fn default() -> Self {
        Self {
            displace: 0.35,
            swell: 0.8,
            brightness: 0.9,
            spin: 0.6,
            point_size: 1.35,
            exposure: 1.4,
            zoom: 0.32,
            twist: 0.45,
        }
    }
}

const FAST_RENDER_CUTOFF: usize = 140_000;

struct Prepared {
    gaussian: Gaussian3,
    /// Which spectrum band drives this splat.
    band: usize,
    /// Unit direction from the cloud centre, used for radial displacement.
    direction: [f32; 3],
}

pub struct SplatView {
    prepared: Vec<Prepared>,
    center: [f32; 3],
    radius: f32,
    density_compensation: f32,
    keys: Vec<u32>,
    values: Vec<u32>,
    accum: Vec<[f32; 4]>,
}

impl SplatView {
    /// `budget` caps how many gaussians are drawn; the source is strided so the
    /// subset still covers the whole scene.
    pub fn new(gaussians: &[Gaussian3], budget: usize) -> Self {
        Self::with_source_count(gaussians, budget, gaussians.len())
    }

    /// Builds a view from a subset while preserving the visual density of the
    /// original capture. `source_count` is the number of splats before loading
    /// was budgeted.
    pub fn with_source_count(gaussians: &[Gaussian3], budget: usize, source_count: usize) -> Self {
        let budget = budget.max(1);
        let stride = if gaussians.len() <= budget {
            1
        } else {
            gaussians.len().div_ceil(budget).max(1)
        };
        let mut kept: Vec<Gaussian3> = gaussians.iter().step_by(stride).cloned().collect();

        // Splat files are authored y-down/z-forward; flip to a y-up view.
        for g in &mut kept {
            g.mean[1] = -g.mean[1];
            g.mean[2] = -g.mean[2];
            g.cov[1] = -g.cov[1];
            g.cov[2] = -g.cov[2];
            g.cov[3] = -g.cov[3];
            g.cov[6] = -g.cov[6];
        }

        let center = robust_center(&kept);
        let radius = robust_radius(&kept, center).max(1.0e-3);
        let density_compensation =
            ((source_count.max(kept.len()) as f32 / kept.len().max(1) as f32).sqrt())
                .clamp(1.0, 2.6);

        let prepared = kept
            .into_iter()
            .map(|g| {
                let mut direction = [
                    g.mean[0] - center[0],
                    g.mean[1] - center[1],
                    g.mean[2] - center[2],
                ];
                let length = (direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2])
                    .sqrt()
                    .max(1.0e-6);
                for axis in &mut direction {
                    *axis /= length;
                }
                // Height maps to frequency, so bass moves the floor and treble
                // moves the top of the scene.
                let height = ((g.mean[1] - center[1]) / radius * 0.5 + 0.5).clamp(0.0, 1.0);
                let band = ((height * BANDS as f32) as usize).min(BANDS - 1);
                Prepared {
                    gaussian: g,
                    band,
                    direction,
                }
            })
            .collect::<Vec<_>>();

        Self {
            prepared,
            center,
            radius,
            density_compensation,
            keys: Vec::new(),
            values: Vec::new(),
            accum: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.prepared.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Render into `rgba`, resizing it to `width * height * 4`.
    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        camera: &Camera,
        spectrum: &Spectrum,
        reactivity: &Reactivity,
        rgba: &mut Vec<u8>,
    ) {
        let width = width.max(1);
        let height = height.max(1);
        let pixels = (width * height) as usize;
        self.accum.clear();
        self.accum.resize(pixels, [0.0, 0.0, 0.0, 1.0]);
        rgba.clear();
        rgba.resize(pixels * 4, 0);

        let basis = view_basis(camera);
        let eye = camera_eye(camera);
        let focal = (height as f32 * 0.5) / (camera.fov_y * 0.5).tan().max(1.0e-4);
        let half_w = width as f32 * 0.5;
        let half_h = height as f32 * 0.5;
        let render_stride = if self.prepared.len() > FAST_RENDER_CUTOFF {
            self.prepared.len().div_ceil(FAST_RENDER_CUTOFF)
        } else {
            1
        };

        let displace = reactivity.displace * self.radius;
        // Centroid drives a cool-to-warm tint so brighter sound looks hotter.
        let warmth = (spectrum.centroid / 2500.0).clamp(0.0, 1.0);

        self.keys.clear();
        self.values.clear();
        for (index, item) in self.prepared.iter().enumerate().step_by(render_stride) {
            let energy = spectrum.bands[item.band];
            let push = energy * displace + spectrum.flux * displace * 0.25;
            let world = [
                item.gaussian.mean[0] + item.direction[0] * push,
                item.gaussian.mean[1] + item.direction[1] * push,
                item.gaussian.mean[2] + item.direction[2] * push,
            ];
            let cam = transform(
                &basis,
                [world[0] - eye[0], world[1] - eye[1], world[2] - eye[2]],
            );
            if cam[2] < 0.05 {
                continue;
            }
            let sx = half_w + focal * cam[0] / cam[2];
            let sy = half_h - focal * cam[1] / cam[2];
            if sx < -64.0 || sy < -64.0 || sx > width as f32 + 64.0 || sy > height as f32 + 64.0 {
                continue;
            }
            let depth = cam[2];
            let key = ((depth * 512.0).clamp(0.0, 4_194_303.0)) as u32;
            self.keys.push(key);
            self.values.push(index as u32);
            let _ = sx;
            let _ = sy;
        }
        radix_sort_pairs(&mut self.keys, &mut self.values);

        // Back to front so a plain "over" blend is correct.
        for slot in (0..self.values.len()).rev() {
            let item = &self.prepared[self.values[slot] as usize];
            let energy = spectrum.bands[item.band];
            let push = energy * displace + spectrum.flux * displace * 0.25;
            let world = [
                item.gaussian.mean[0] + item.direction[0] * push,
                item.gaussian.mean[1] + item.direction[1] * push,
                item.gaussian.mean[2] + item.direction[2] * push,
            ];
            let cam = transform(
                &basis,
                [world[0] - eye[0], world[1] - eye[1], world[2] - eye[2]],
            );
            if cam[2] < 0.05 {
                continue;
            }
            let sx = half_w + focal * cam[0] / cam[2];
            let sy = half_h - focal * cam[1] / cam[2];

            let swell = (1.0 + energy * reactivity.swell + spectrum.level * 0.3)
                * reactivity.point_size
                * self.density_compensation
                * (render_stride as f32).sqrt();
            let cov = project_covariance(&item.gaussian.cov, &basis, cam, focal, swell);
            let det = cov[0] * cov[2] - cov[1] * cov[1];
            if det <= 1.0e-9 {
                continue;
            }
            let inv_det = 1.0 / det;
            let conic = [cov[2] * inv_det, -cov[1] * inv_det, cov[0] * inv_det];
            let mid = 0.5 * (cov[0] + cov[2]);
            let spread = (mid * mid - det).max(0.0).sqrt();
            let extent = (3.0 * (mid + spread).max(0.0).sqrt()).clamp(0.6, 128.0);

            let boost = 1.0 + energy * reactivity.brightness + spectrum.flux * 0.5;
            let color = [
                (item.gaussian.color[0] * boost * (0.75 + 0.55 * warmth)).min(4.0),
                (item.gaussian.color[1] * boost * 0.92).min(4.0),
                (item.gaussian.color[2] * boost * (1.25 - 0.5 * warmth)).min(4.0),
            ];

            let min_x = ((sx - extent).floor().max(0.0)) as u32;
            let max_x = ((sx + extent).ceil().min(width as f32 - 1.0)).max(0.0) as u32;
            let min_y = ((sy - extent).floor().max(0.0)) as u32;
            let max_y = ((sy + extent).ceil().min(height as f32 - 1.0)).max(0.0) as u32;
            if min_x > max_x || min_y > max_y {
                continue;
            }
            for py in min_y..=max_y {
                let dy = py as f32 - sy;
                for px in min_x..=max_x {
                    let dx = px as f32 - sx;
                    let power =
                        -0.5 * (conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy);
                    if power < -6.0 {
                        continue;
                    }
                    let alpha = (item.gaussian.opacity * power.exp()).clamp(0.0, 0.99);
                    if alpha < 0.004 {
                        continue;
                    }
                    let pixel = &mut self.accum[(py * width + px) as usize];
                    for channel in 0..3 {
                        pixel[channel] = pixel[channel] * (1.0 - alpha) + color[channel] * alpha;
                    }
                    pixel[3] *= 1.0 - alpha;
                }
            }
        }

        for (pixel, out) in self.accum.iter().zip(rgba.chunks_exact_mut(4)) {
            let background = 0.025 + spectrum.level * 0.035;
            for channel in 0..3 {
                let value =
                    (pixel[channel] + pixel[3] * background) * reactivity.exposure.max(0.05);
                let mapped = value / (1.0 + value);
                out[channel] = (mapped.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8;
            }
            out[3] = 255;
        }
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Median per axis, so a few stray gaussians cannot drag the framing away.
fn robust_center(gaussians: &[Gaussian3]) -> [f32; 3] {
    if gaussians.is_empty() {
        return [0.0; 3];
    }
    let mut center = [0.0f32; 3];
    let mut axis_values: Vec<f32> = Vec::with_capacity(gaussians.len());
    for axis in 0..3 {
        axis_values.clear();
        axis_values.extend(gaussians.iter().map(|g| g.mean[axis]));
        axis_values.sort_by(f32::total_cmp);
        center[axis] = axis_values[axis_values.len() / 2];
    }
    center
}

/// A percentile radius keeps isolated reconstruction debris from shrinking the
/// entire scene to a tiny object in the middle of the viewport.
fn robust_radius(gaussians: &[Gaussian3], center: [f32; 3]) -> f32 {
    if gaussians.is_empty() {
        return 0.0;
    }
    let mut distances = gaussians
        .iter()
        .map(|g| distance(g.mean, center))
        .collect::<Vec<_>>();
    distances.sort_by(f32::total_cmp);
    let index = ((distances.len() - 1) as f32 * 0.975).round() as usize;
    distances[index]
}

pub fn camera_eye(camera: &Camera) -> [f32; 3] {
    let cp = camera.pitch.cos();
    [
        camera.target[0] + camera.distance * cp * camera.yaw.sin(),
        camera.target[1] + camera.distance * camera.pitch.sin(),
        camera.target[2] + camera.distance * cp * camera.yaw.cos(),
    ]
}

/// Rows are the camera right, up and forward axes in world space.
fn view_basis(camera: &Camera) -> [f32; 9] {
    let eye = camera_eye(camera);
    let mut forward = [
        camera.target[0] - eye[0],
        camera.target[1] - eye[1],
        camera.target[2] - eye[2],
    ];
    normalize(&mut forward);
    let world_up = [0.0f32, 1.0, 0.0];
    let mut right = cross(forward, world_up);
    if right[0].abs() + right[1].abs() + right[2].abs() < 1.0e-5 {
        right = [1.0, 0.0, 0.0];
    }
    normalize(&mut right);
    let up = cross(right, forward);
    [
        right[0], right[1], right[2], up[0], up[1], up[2], forward[0], forward[1], forward[2],
    ]
}

fn transform(basis: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        basis[0] * v[0] + basis[1] * v[1] + basis[2] * v[2],
        basis[3] * v[0] + basis[4] * v[1] + basis[5] * v[2],
        basis[6] * v[0] + basis[7] * v[1] + basis[8] * v[2],
    ]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: &mut [f32; 3]) {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1.0e-8);
    for axis in v.iter_mut() {
        *axis /= length;
    }
}

/// EWA projection of a 3D covariance into the 2D screen covariance
/// `[xx, xy, yy]`, with the usual low-pass term so tiny splats stay visible.
pub fn project_covariance(
    cov: &[f32; 9],
    basis: &[f32; 9],
    cam: [f32; 3],
    focal: f32,
    swell: f32,
) -> [f32; 3] {
    let z = cam[2].max(1.0e-4);
    let jacobian = [
        focal / z,
        0.0,
        -focal * cam[0] / (z * z),
        0.0,
        -focal / z,
        focal * cam[1] / (z * z),
    ];
    // T = J * R, a 2x3 matrix.
    let mut t = [0.0f32; 6];
    for row in 0..2 {
        for col in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += jacobian[row * 3 + k] * basis[k * 3 + col];
            }
            t[row * 3 + col] = sum;
        }
    }
    let scale = (swell * swell).max(1.0e-6);
    // Sigma2d = T * Sigma3d * T^T.
    let mut out = [0.0f32; 3];
    let mut temp = [0.0f32; 6];
    for row in 0..2 {
        for col in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += t[row * 3 + k] * cov[k * 3 + col] * scale;
            }
            temp[row * 3 + col] = sum;
        }
    }
    let mut full = [0.0f32; 4];
    for row in 0..2 {
        for col in 0..2 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += temp[row * 3 + k] * t[col * 3 + k];
            }
            full[row * 2 + col] = sum;
        }
    }
    out[0] = full[0] + 0.3;
    out[1] = full[1];
    out[2] = full[3] + 0.3;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(mean: [f32; 3], scale: f32) -> Gaussian3 {
        let mut cov = [0.0f32; 9];
        cov[0] = scale * scale;
        cov[4] = scale * scale;
        cov[8] = scale * scale;
        Gaussian3 {
            mean,
            cov,
            color: [1.0, 1.0, 1.0],
            opacity: 0.95,
        }
    }

    fn silent() -> Spectrum {
        Spectrum::default()
    }

    #[test]
    fn a_splat_in_front_of_the_camera_lights_pixels() {
        let mut view = SplatView::new(&[point([0.0, 0.0, 0.0], 0.25)], 64);
        let camera = Camera {
            target: [0.0, 0.0, 0.0],
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
        };
        let mut rgba = Vec::new();
        view.render(
            64,
            64,
            &camera,
            &silent(),
            &Reactivity::default(),
            &mut rgba,
        );
        assert_eq!(rgba.len(), 64 * 64 * 4);
        let center = ((32 * 64 + 32) * 4) as usize;
        assert!(rgba[center] > 100, "centre pixel {} too dark", rgba[center]);
        let corner = 0usize;
        assert!(
            rgba[corner] < 60,
            "corner pixel {} too bright",
            rgba[corner]
        );
    }

    #[test]
    fn spectrum_energy_brightens_and_moves_the_cloud() {
        let cloud: Vec<Gaussian3> = (0..64)
            .map(|i| {
                let a = i as f32 * 0.1;
                point([a.sin(), a.cos() * 0.5, a.cos()], 0.1)
            })
            .collect();
        let camera = Camera {
            target: [0.0, 0.0, 0.0],
            distance: 4.0,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
        };
        let mut quiet_view = SplatView::new(&cloud, 256);
        let mut loud_view = SplatView::new(&cloud, 256);
        let mut quiet = Vec::new();
        let mut loud = Vec::new();
        quiet_view.render(
            48,
            48,
            &camera,
            &silent(),
            &Reactivity::default(),
            &mut quiet,
        );
        let energetic = Spectrum {
            bands: [1.0; BANDS],
            level: 1.0,
            centroid: 2000.0,
            flux: 0.8,
        };
        loud_view.render(
            48,
            48,
            &camera,
            &energetic,
            &Reactivity::default(),
            &mut loud,
        );
        let sum = |buffer: &[u8]| buffer.iter().map(|v| *v as u64).sum::<u64>();
        assert!(sum(&loud) > sum(&quiet), "loud frame should be brighter");
        assert_ne!(quiet, loud, "spectrum did not change the image");
    }

    #[test]
    fn an_empty_cloud_renders_a_blank_frame() {
        let mut view = SplatView::new(&[], 16);
        assert!(view.is_empty());
        let mut rgba = Vec::new();
        view.render(
            8,
            8,
            &Camera::default(),
            &silent(),
            &Reactivity::default(),
            &mut rgba,
        );
        assert_eq!(rgba.len(), 8 * 8 * 4);
        assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
    }

    #[test]
    fn isolated_outliers_do_not_shrink_scene_framing() {
        let mut cloud = (0..200)
            .map(|i| {
                let angle = i as f32 * 0.1;
                point([angle.sin(), angle.cos(), 0.0], 0.05)
            })
            .collect::<Vec<_>>();
        cloud.push(point([1_000.0, 0.0, 0.0], 0.05));

        let view = SplatView::new(&cloud, cloud.len());
        assert!(
            view.radius() < 2.0,
            "outlier inflated radius to {}",
            view.radius()
        );
    }

    #[test]
    fn projected_covariance_shrinks_with_distance() {
        let mut cov = [0.0f32; 9];
        cov[0] = 0.04;
        cov[4] = 0.04;
        cov[8] = 0.04;
        let basis = view_basis(&Camera::default());
        let near = project_covariance(&cov, &basis, [0.0, 0.0, 1.0], 100.0, 1.0);
        let far = project_covariance(&cov, &basis, [0.0, 0.0, 8.0], 100.0, 1.0);
        assert!(near[0] > far[0] * 4.0, "near {:?} far {:?}", near, far);
    }
}
