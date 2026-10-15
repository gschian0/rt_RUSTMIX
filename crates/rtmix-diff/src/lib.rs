//! Offline forward/backward fit of splat color and opacity.
//! This crate is not linked from the live audio app.

use rtmix_gfx::Gaussian3;

#[derive(Clone, Debug)]
pub struct AppearanceGrad {
    pub color: Vec<[f32; 3]>,
    pub opacity: Vec<f32>,
}

pub fn render(gaussians: &[Gaussian3], width: usize, height: usize) -> Vec<f32> {
    let mut image = vec![0.0f32; width * height * 3];
    let mut order: Vec<usize> = (0..gaussians.len()).collect();
    order.sort_by(|a, b| gaussians[*a].mean[2].total_cmp(&gaussians[*b].mean[2]));
    for y in 0..height {
        for x in 0..width {
            let mut color = [0.0f32; 3];
            let mut transmittance = 1.0f32;
            for index in &order {
                let g = &gaussians[*index];
                let dx = x as f32 - g.mean[0];
                let dy = y as f32 - g.mean[1];
                let alpha = (g.opacity * libm::expf(-(dx * dx + dy * dy) / 8.0)).clamp(0.0, 0.99);
                for channel in 0..3 {
                    color[channel] += transmittance * alpha * g.color[channel];
                }
                transmittance *= 1.0 - alpha;
                if transmittance < 1.0e-3 {
                    break;
                }
            }
            let pixel = (y * width + x) * 3;
            image[pixel..pixel + 3].copy_from_slice(&color);
        }
    }
    image
}

pub fn backward(gaussians: &[Gaussian3], image: &[f32], target: &[f32], width: usize, height: usize) -> AppearanceGrad {
    let mut color_grad = vec![[0.0f32; 3]; gaussians.len()];
    let mut opacity_grad = vec![0.0f32; gaussians.len()];
    let mut order: Vec<usize> = (0..gaussians.len()).collect();
    order.sort_by(|a, b| gaussians[*a].mean[2].total_cmp(&gaussians[*b].mean[2]));
    for y in 0..height {
        for x in 0..width {
            let pixel = (y * width + x) * 3;
            let mut dl = [0.0f32; 3];
            for channel in 0..3 {
                dl[channel] = image[pixel + channel] - target[pixel + channel];
            }
            let mut transmittance = 1.0f32;
            for index in &order {
                let g = &gaussians[*index];
                let dx = x as f32 - g.mean[0];
                let dy = y as f32 - g.mean[1];
                let spatial = libm::expf(-(dx * dx + dy * dy) / 8.0);
                let alpha = (g.opacity * spatial).clamp(0.0, 0.99);
                let weight = transmittance * alpha;
                for channel in 0..3 {
                    color_grad[*index][channel] += dl[channel] * weight;
                }
                let mut d_alpha = 0.0f32;
                for channel in 0..3 {
                    d_alpha += dl[channel] * transmittance * g.color[channel];
                }
                opacity_grad[*index] += d_alpha * spatial;
                transmittance *= 1.0 - alpha;
                if transmittance < 1.0e-3 {
                    break;
                }
            }
        }
    }
    AppearanceGrad {
        color: color_grad,
        opacity: opacity_grad,
    }
}

pub fn loss(image: &[f32], target: &[f32]) -> f32 {
    image.iter().zip(target.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// One gradient step on color and opacity. Returns the loss before the step.
pub fn fit_step(gaussians: &mut [Gaussian3], target: &[f32], width: usize, height: usize, lr: f32) -> f32 {
    let image = render(gaussians, width, height);
    let value = loss(&image, target);
    let grad = backward(gaussians, &image, target, width, height);
    for (g, (color, opacity)) in gaussians.iter_mut().zip(grad.color.iter().zip(grad.opacity.iter())) {
        for channel in 0..3 {
            g.color[channel] = (g.color[channel] - lr * color[channel]).clamp(0.0, 1.0);
        }
        g.opacity = (g.opacity - lr * opacity).clamp(0.0, 1.0);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appearance_step_reduces_loss() {
        let mut gaussians = vec![Gaussian3 {
            mean: [1.0, 1.0, 0.2],
            cov: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            color: [0.0, 0.0, 1.0],
            opacity: 0.8,
        }];
        let mut target_src = gaussians.clone();
        target_src[0].color = [1.0, 0.0, 0.0];
        let target = render(&target_src, 3, 3);
        let before = loss(&render(&gaussians, 3, 3), &target);
        for _ in 0..40 {
            fit_step(&mut gaussians, &target, 3, 3, 0.15);
        }
        let after = loss(&render(&gaussians, 3, 3), &target);
        assert!(after < before, "before {before} after {after}");
        assert!(
            gaussians[0].color[0] > gaussians[0].color[2],
            "color {:?}",
            gaussians[0].color
        );
    }
}
