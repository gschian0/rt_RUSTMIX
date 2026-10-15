//! 16x16 tile Gaussian rasterizer.
//! project -> bin -> radix sort by depth -> front-to-back blend.
//! Desktop prefers SPIR-V from slangc when that compiler is on PATH.
//! The same algorithm is always available as embedded WGSL.

struct Frame {
    view_proj: mat4x4<f32>,
    params: vec4<u32>,
    time: vec4<f32>,
}

struct Gaussian3D {
    mean_opacity: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    color_c8: vec4<f32>,
}

struct KeyValue {
    key: u32,
    value: u32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read_write> records: array<KeyValue>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> image: array<vec4<f32>>;

fn clip_point(mean: vec3<f32>) -> vec4<f32> {
    return frame.view_proj * vec4<f32>(mean, 1.0);
}

fn screen_of(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(abs(clip.w), 1e-6);
    let width = f32(frame.params.x);
    let height = f32(frame.params.y);
    let sx = (ndc.x * 0.5 + 0.5) * width;
    let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * height;
    return vec2<f32>(sx, sy);
}

@compute @workgroup_size(64)
fn project(tid: vec3<u32>) {
    let index = tid.x;
    if (index >= frame.params.z) {
        return;
    }
    let g = gaussians[index];
    let clip = clip_point(g.mean_opacity.xyz);
    if (clip.w <= 1e-4) {
        records[index].key = 0xffffffffu;
        records[index].value = index;
        return;
    }
    let screen = screen_of(clip);
    let tile = frame.params.w;
    let tiles_x = (frame.params.x + tile - 1u) / tile;
    let tx = u32(clamp(screen.x / f32(tile), 0.0, f32(tiles_x - 1u)));
    let depth = clamp(clip.z / clip.w, 0.0, 1.0);
    let key = (tx << 16u) | u32(depth * 65535.0);
    records[index].key = key;
    records[index].value = index;
}

@compute @workgroup_size(64)
fn radix_histogram(tid: vec3<u32>) {
    let index = tid.x;
    let count = frame.params.z;
    if (index >= count) {
        return;
    }
    let shift = frame.time.y;
    let key = records[index].key;
    let digit = (key >> u32(shift)) & 255u;
    atomicAdd(&hist[digit], 1u);
}

@compute @workgroup_size(1)
fn radix_prefix(_tid: vec3<u32>) {
    var sum = 0u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let value = atomicLoad(&hist[i]);
        atomicStore(&hist[i], sum);
        sum = sum + value;
    }
}

@compute @workgroup_size(64)
fn radix_scatter(tid: vec3<u32>) {
    let index = tid.x;
    if (index >= frame.params.z) {
        return;
    }
    let shift = frame.time.y;
    let key = records[index].key;
    let digit = (key >> u32(shift)) & 255u;
    let dest = atomicAdd(&hist[digit], 1u);
    records[dest].key = key;
    records[dest].value = records[index].value;
}

@compute @workgroup_size(16, 16)
fn blend_tile(tid: vec3<u32>) {
    let tile = frame.params.w;
    let width = frame.params.x;
    let height = frame.params.y;
    let pix_x = tid.x + tid.z * tile;
    let pix_y = tid.y;
    if (pix_x >= width || pix_y >= height) {
        return;
    }
    var color = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance = 1.0;
    let count = frame.params.z;
    for (var i = 0u; i < count; i = i + 1u) {
        let id = records[i].value;
        if (records[i].key == 0xffffffffu) {
            continue;
        }
        let g = gaussians[id];
        let clip = clip_point(g.mean_opacity.xyz);
        if (clip.w <= 1e-4) {
            continue;
        }
        let screen = screen_of(clip);
        let dx = f32(pix_x) - screen.x;
        let dy = f32(pix_y) - screen.y;
        let radius2 = max(g.c0.x, 1.0);
        let alpha = g.mean_opacity.w * exp(-(dx * dx + dy * dy) / radius2);
        let a = clamp(alpha, 0.0, 0.99);
        color = color + transmittance * a * g.color_c8.xyz;
        transmittance = transmittance * (1.0 - a);
        if (transmittance < 0.01) {
            break;
        }
    }
    let pixel = pix_y * width + pix_x;
    image[pixel] = vec4<f32>(color, 1.0 - transmittance);
}
