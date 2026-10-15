//! Temporal slice of an XYZT Gaussian. Same conditional used by the CPU path.
//! t = sample_time / sample_rate, mapped across the file range by the score.

struct Frame {
    view_proj: mat4x4<f32>,
    params: vec4<u32>,
    time: vec4<f32>,
}

struct Gaussian4D {
    mean: vec4<f32>,
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    row3: vec4<f32>,
    color_opacity: vec4<f32>,
}

struct Gaussian3D {
    mean_opacity: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    color_c8: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> src: array<Gaussian4D>;
@group(0) @binding(2) var<storage, read_write> dst: array<Gaussian3D>;

@compute @workgroup_size(64)
fn slice4d(tid: vec3<u32>) {
    let index = tid.x;
    if (index >= frame.params.z) {
        return;
    }
    let g = src[index];
    let var_t = max(g.row3.w, 1e-8);
    let dt = frame.time.x - g.mean.w;
    let weight = exp(-0.5 * dt * dt / var_t);
    let cx0 = g.row0.w / var_t;
    let cx1 = g.row1.w / var_t;
    let cx2 = g.row2.w / var_t;
    let mean = vec3<f32>(
        g.mean.x + cx0 * dt,
        g.mean.y + cx1 * dt,
        g.mean.z + cx2 * dt,
    );
    let s0 = g.row0.w;
    let s1 = g.row1.w;
    let s2 = g.row2.w;
    let m00 = g.row0.x - s0 * s0 / var_t;
    let m01 = g.row0.y - s0 * s1 / var_t;
    let m02 = g.row0.z - s0 * s2 / var_t;
    let m11 = g.row1.y - s1 * s1 / var_t;
    let m12 = g.row1.z - s1 * s2 / var_t;
    let m22 = g.row2.z - s2 * s2 / var_t;
    dst[index].mean_opacity = vec4<f32>(mean, g.color_opacity.w * weight);
    dst[index].c0 = vec4<f32>(m00, m01, m02, m11);
    dst[index].c1 = vec4<f32>(m12, m22, 0.0, 0.0);
    dst[index].color_c8 = vec4<f32>(g.color_opacity.xyz, 0.0);
}
