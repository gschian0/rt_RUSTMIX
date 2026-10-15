//! CPU reference for the 16x16 tile rasterizer: project, bin, radix sort, blend.

use crate::slice::Gaussian3;

pub const TILE_SIZE: u32 = 16;

pub fn project_point(
    mean: [f32; 3],
    view_proj: &[f32; 16],
    width: f32,
    height: f32,
) -> Option<(f32, f32, f32)> {
    let x = mean[0];
    let y = mean[1];
    let z = mean[2];
    let cx = view_proj[0] * x + view_proj[1] * y + view_proj[2] * z + view_proj[3];
    let cy = view_proj[4] * x + view_proj[5] * y + view_proj[6] * z + view_proj[7];
    let cz = view_proj[8] * x + view_proj[9] * y + view_proj[10] * z + view_proj[11];
    let cw = view_proj[12] * x + view_proj[13] * y + view_proj[14] * z + view_proj[15];
    if cw.abs() < 1.0e-6 || cw <= 0.0 {
        return None;
    }
    let ndc_x = cx / cw;
    let ndc_y = cy / cw;
    let sx = (ndc_x * 0.5 + 0.5) * width;
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height;
    Some((sx, sy, cz / cw))
}

pub fn tile_bins(x: f32, y: f32, radius: f32, width: u32, height: u32) -> Vec<(u32, u32)> {
    let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    if tiles_x == 0 || tiles_y == 0 {
        return Vec::new();
    }
    let min_tx = ((x - radius) / TILE_SIZE as f32).floor().max(0.0) as u32;
    let max_tx = ((x + radius) / TILE_SIZE as f32).floor().max(0.0) as u32;
    let min_ty = ((y - radius) / TILE_SIZE as f32).floor().max(0.0) as u32;
    let max_ty = ((y + radius) / TILE_SIZE as f32).floor().max(0.0) as u32;
    let mut out = Vec::new();
    for ty in min_ty..=max_ty.min(tiles_y - 1) {
        for tx in min_tx..=max_tx.min(tiles_x - 1) {
            out.push((tx, ty));
        }
    }
    out
}

pub fn radix_sort_pairs(keys: &mut [u32], values: &mut [u32]) {
    assert_eq!(keys.len(), values.len());
    let n = keys.len();
    if n == 0 {
        return;
    }
    let mut key_b = vec![0u32; n];
    let mut val_b = vec![0u32; n];
    for pass in 0..4 {
        let shift = pass * 8;
        let mut count = [0u32; 256];
        for key in keys.iter() {
            count[((key >> shift) & 255) as usize] += 1;
        }
        let mut sum = 0u32;
        for bin in &mut count {
            let value = *bin;
            *bin = sum;
            sum += value;
        }
        for i in 0..n {
            let digit = ((keys[i] >> shift) & 255) as usize;
            let dest = count[digit] as usize;
            count[digit] += 1;
            key_b[dest] = keys[i];
            val_b[dest] = values[i];
        }
        keys.copy_from_slice(&key_b);
        values.copy_from_slice(&val_b);
    }
}

pub fn blend_front_to_back(fragments: &[(f32, [f32; 3], f32)]) -> [f32; 3] {
    let mut order: Vec<usize> = (0..fragments.len()).collect();
    order.sort_by(|a, b| fragments[*a].0.total_cmp(&fragments[*b].0));
    let mut color = [0.0f32; 3];
    let mut transmittance = 1.0f32;
    for index in order {
        let (_, rgb, alpha) = fragments[index];
        let alpha = alpha.clamp(0.0, 0.99);
        for channel in 0..3 {
            color[channel] += transmittance * alpha * rgb[channel];
        }
        transmittance *= 1.0 - alpha;
        if transmittance < 1.0e-3 {
            break;
        }
    }
    color
}

pub fn identity_view() -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

pub fn raster_preview(
    gaussians: &[Gaussian3],
    width: u32,
    height: u32,
    view_proj: &[f32; 16],
) -> Vec<[f32; 3]> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (index, g) in gaussians.iter().enumerate() {
        let Some((x, y, depth)) = project_point(g.mean, view_proj, width as f32, height as f32)
        else {
            continue;
        };
        if !(0.0..=1.5).contains(&depth) {
            continue;
        }
        let tiles = tile_bins(x, y, 8.0, width, height);
        for (tx, ty) in tiles {
            let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
            let tile_id = ty * tiles_x + tx;
            let key = (tile_id << 16) | ((depth.clamp(0.0, 1.0) * 65535.0) as u32);
            keys.push(key);
            values.push(index as u32);
        }
    }
    radix_sort_pairs(&mut keys, &mut values);
    let mut image = vec![[0.0f32; 3]; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let mut frags = Vec::new();
            for (key, value) in keys.iter().zip(values.iter()) {
                let g = &gaussians[*value as usize];
                let Some((sx, sy, depth)) =
                    project_point(g.mean, view_proj, width as f32, height as f32)
                else {
                    continue;
                };
                let dx = x as f32 - sx;
                let dy = y as f32 - sy;
                if dx * dx + dy * dy > 64.0 {
                    continue;
                }
                let _ = key;
                frags.push((depth, g.color, g.opacity));
            }
            image[(y * width + x) as usize] = blend_front_to_back(&frags);
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_x_projects_to_the_right() {
        let view = identity_view();
        let (x, _y, _z) = project_point([1.0, 0.0, 0.5], &view, 200.0, 100.0).unwrap();
        assert!(x > 100.0, "screen x {x}");
    }

    #[test]
    fn bins_use_sixteen_pixel_tiles() {
        let tiles = tile_bins(20.0, 20.0, 4.0, 64, 64);
        assert!(tiles.contains(&(1, 1)));
        assert_eq!(TILE_SIZE, 16);
    }

    #[test]
    fn radix_sort_orders_depth_keys() {
        let mut keys = [5u32, 1, 9, 1, 3];
        let mut values = [10u32, 11, 12, 13, 14];
        radix_sort_pairs(&mut keys, &mut values);
        assert_eq!(keys, [1, 1, 3, 5, 9]);
        assert_eq!(values, [11, 13, 14, 10, 12]);
    }

    #[test]
    fn front_to_back_hides_the_far_color() {
        let color =
            blend_front_to_back(&[(0.2, [1.0, 0.0, 0.0], 0.99), (0.8, [0.0, 0.0, 1.0], 0.99)]);
        assert!(color[0] > 0.9);
        assert!(color[2] < 0.05);
    }
}
