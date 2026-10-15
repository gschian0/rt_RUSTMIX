struct Uniforms {
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    eye: vec4<f32>,
    viewport: vec4<f32>,
    audio: vec4<f32>,
    visual: vec4<f32>,
    bands0: vec4<f32>,
    bands1: vec4<f32>,
    bands2: vec4<f32>,
}

struct Splat {
    mean_opacity: vec4<f32>,
    cov0_band: vec4<f32>,
    cov1_pad: vec4<f32>,
    cov2_density: vec4<f32>,
    color: vec4<f32>,
    direction: vec4<f32>,
}

struct State {
    chemical: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> splat_state: array<State>;

fn band_energy(index: u32) -> f32 {
    if (index < 4u) {
        return uniforms.bands0[index];
    }
    if (index < 8u) {
        return uniforms.bands1[index - 4u];
    }
    return uniforms.bands2[index - 8u];
}

struct SplatOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) chemical: vec2<f32>,
}

@vertex
fn vs_splat(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> SplatOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let splat = splats[instance];
    let chemical = splat_state[u32(splat.cov1_pad.w)].chemical;
    let energy = band_energy(u32(splat.cov0_band.w));
    let push = (energy + uniforms.audio.y * 0.25) * uniforms.visual.x;
    let world = splat.mean_opacity.xyz + splat.direction.xyz * push;
    let relative = world - uniforms.eye.xyz;
    let cam = vec3<f32>(
        dot(relative, uniforms.right.xyz),
        dot(relative, uniforms.up.xyz),
        dot(relative, uniforms.forward.xyz),
    );
    let corner = corners[vertex];
    var out: SplatOut;
    if (cam.z <= 0.02) {
        out.position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
        out.local = corner;
        out.color = vec3<f32>(0.0);
        out.opacity = 0.0;
        out.chemical = chemical;
        return out;
    }
    let reaction_scale = 1.0 + chemical.y * (1.15 + uniforms.visual.y * 0.8);
    let energy_scale = (1.0 + energy * uniforms.visual.y + uniforms.audio.x * 0.25)
* reaction_scale
* uniforms.visual.w * splat.cov2_density.w;
    let cov_world = mat3x3<f32>(
        splat.cov0_band.xyz,
        splat.cov1_pad.xyz,
        splat.cov2_density.xyz,
    );
    let camera_rotation = mat3x3<f32>(
        uniforms.right.xyz,
        uniforms.up.xyz,
        uniforms.forward.xyz,
    );
    let cov_camera = transpose(camera_rotation) * cov_world * camera_rotation;
    let inv_z = 1.0 / cam.z;
    let jacobian_x = vec3<f32>(
        uniforms.viewport.z * inv_z,
        0.0,
        -uniforms.viewport.z * cam.x * inv_z * inv_z,
    );
    let jacobian_y = vec3<f32>(
        0.0,
        uniforms.viewport.z * inv_z,
        -uniforms.viewport.z * cam.y * inv_z * inv_z,
    );
    let covariance_scale = energy_scale * energy_scale;
    let covariance_xx = dot(jacobian_x, cov_camera * jacobian_x)
        * covariance_scale + 0.35;
    let covariance_xy = dot(jacobian_x, cov_camera * jacobian_y)
        * covariance_scale;
    let covariance_yy = dot(jacobian_y, cov_camera * jacobian_y)
        * covariance_scale + 0.35;
    let trace = covariance_xx + covariance_yy;
    let delta = sqrt(max(
        (covariance_xx - covariance_yy)
            * (covariance_xx - covariance_yy)
            + 4.0 * covariance_xy * covariance_xy,
        0.0,
    ));
    let major = clamp(sqrt(max((trace + delta) * 0.5, 0.35)) * 3.0, 0.65, 128.0);
    let minor = clamp(sqrt(max((trace - delta) * 0.5, 0.35)) * 3.0, 0.65, 128.0);
    let angle = 0.5 * atan2(2.0 * covariance_xy, covariance_xx - covariance_yy);
    let ellipse = vec2<f32>(
        corner.x * major * cos(angle) - corner.y * minor * sin(angle),
        corner.x * major * sin(angle) + corner.y * minor * cos(angle),
    );
    let center = vec2<f32>(
        cam.x * uniforms.viewport.z / cam.z / (uniforms.viewport.x * 0.5),
        cam.y * uniforms.viewport.z / cam.z / (uniforms.viewport.y * 0.5),
    );
    let offset = vec2<f32>(
        ellipse.x / (uniforms.viewport.x * 0.5),
        ellipse.y / (uniforms.viewport.y * 0.5),
    );
    out.position = vec4<f32>(center + offset, 0.5, 1.0);
    out.local = corner;
    let warmth = clamp(uniforms.audio.z / 5000.0, 0.0, 1.0);
    let glow = 1.0 + energy * uniforms.visual.z + uniforms.audio.y * 0.4
        + chemical.y * (1.6 + uniforms.visual.z * 0.9);
    let rainbow_phase = world.y * 0.17 + world.x * 0.09 + uniforms.viewport.w * 0.08
        + energy * 1.7;
    let spectral_color = 0.62 + 0.38 * cos(
        vec3<f32>(0.0, 2.094, 4.188) + rainbow_phase
    );
    let reaction_phase = chemical.y * 8.0 - chemical.x * 2.0
        + world.x * 0.11 + world.z * 0.07 + uniforms.viewport.w * 0.05;
    let reaction_color = 0.5 + 0.5 * cos(
        vec3<f32>(0.2, 2.3, 4.4) + reaction_phase
    );
    let base_color = mix(splat.color.rgb, splat.color.rgb * spectral_color * 1.6, 0.28);
    out.color = mix(base_color, reaction_color * (0.8 + base_color * 1.4),
        0.42 + chemical.y * 0.56)
        * glow
        * vec3<f32>(0.8 + warmth * 0.45, 0.95, 1.2 - warmth * 0.4);
    out.opacity = splat.mean_opacity.w * (0.78 + chemical.y * 0.82);
    out.chemical = chemical;
    return out;
}

@fragment
fn fs_splat(input: SplatOut) -> @location(0) vec4<f32> {
    let radius2 = dot(input.local, input.local);
    if (radius2 > 1.0) {
        discard;
    }
    let reaction_contrast = 0.78 + smoothstep(0.05, 0.65, input.chemical.y) * 0.72;
    let alpha = clamp(input.opacity * reaction_contrast * exp(-3.2 * radius2), 0.0, 0.98);
    let exposed = input.color * uniforms.audio.w * (1.0 + input.chemical.y * 0.9);
    let mapped = exposed / (vec3<f32>(1.0) + exposed);
    return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), alpha);
}

@group(1) @binding(0) var<uniform> reaction_uniforms: Uniforms;
@group(1) @binding(1) var<storage, read> state_in: array<State>;
@group(1) @binding(2) var<storage, read_write> state_out: array<State>;

const FIELD_W: u32 = 512u;
const FIELD_H: u32 = 288u;

fn field_index(x: i32, y: i32) -> u32 {
    let wrapped_x = (x % i32(FIELD_W) + i32(FIELD_W)) % i32(FIELD_W);
    let wrapped_y = (y % i32(FIELD_H) + i32(FIELD_H)) % i32(FIELD_H);
    return u32(wrapped_y) * FIELD_W + u32(wrapped_x);
}

fn load_state(x: i32, y: i32) -> vec2<f32> {
    return state_in[field_index(x, y)].chemical;
}

@compute @workgroup_size(8, 8)
fn reaction(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= FIELD_W || id.y >= FIELD_H) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let center = load_state(x, y);
    let lap = load_state(x - 1, y) + load_state(x + 1, y)
        + load_state(x, y - 1) + load_state(x, y + 1) - center * 4.0;
    let mode = reaction_uniforms.viewport.w;
    var feed = select(0.032, 0.022, mode > 0.5);
    feed = select(feed, 0.046, mode > 1.5);
    var kill = select(0.061, 0.052, mode > 0.5);
    kill = select(kill, 0.064, mode > 1.5);
    feed += reaction_uniforms.bands0.z * 0.004;
    kill += reaction_uniforms.bands2.x * 0.003;
    let uvv = center.x * center.y * center.y;
    var next = vec2<f32>(
        center.x + 0.16 * lap.x - uvv + feed * (1.0 - center.x),
        center.y + 0.08 * lap.y + uvv - (feed + kill) * center.y,
    );

    let phase = reaction_uniforms.viewport.z;
    for (var band = 0u; band < 12u; band += 1u) {
        let energy = band_energy_reaction(band);
        let px = f32(band) / 12.0 * f32(FIELD_W)
            + sin(phase * (0.7 + f32(band) * 0.09)) * 70.0;
        let py = f32((band * 29u) % FIELD_H) + sin(phase * 1.7 + f32(band)) * 35.0;
        let delta = vec2<f32>(f32(id.x) - px, f32(id.y) - py);
        let injection = exp(-dot(delta, delta) / (18.0 + energy * 180.0)) * energy;
        next.y = max(next.y, injection * 0.95);
        next.x = min(next.x, 1.0 - injection * 0.55);
    }
    state_out[id.y * FIELD_W + id.x].chemical = clamp(next, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn band_energy_reaction(index: u32) -> f32 {
    if (index < 4u) {
        return reaction_uniforms.bands0[index];
    }
    if (index < 8u) {
        return reaction_uniforms.bands1[index - 4u];
    }
    return reaction_uniforms.bands2[index - 8u];
}

fn splat_band(index: u32) -> f32 {
    let band = index % 12u;
    if (band < 4u) {
        return reaction_uniforms.bands0[band];
    }
    if (band < 8u) {
        return reaction_uniforms.bands1[band - 4u];
    }
    return reaction_uniforms.bands2[band - 8u];
}

fn splat_neighbor(index: i32, count: u32) -> u32 {
    return u32((index % i32(count) + i32(count)) % i32(count));
}

@compute @workgroup_size(64)
fn splat_reaction(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = arrayLength(&state_in);
    if (id.x >= count || count == 0u) {
        return;
    }
    let center = state_in[id.x].chemical;
    let offsets = array<i32, 8>(-64, -16, -4, -1, 1, 4, 16, 64);
    var average = vec2<f32>(0.0);
    for (var i = 0u; i < 8u; i += 1u) {
        average += state_in[splat_neighbor(i32(id.x) + offsets[i], count)].chemical;
    }
    average *= 0.125;
    let laplacian = average - center;
    let feed = 0.027 + reaction_uniforms.bands0.z * 0.008;
    let kill = 0.056 + reaction_uniforms.bands2.x * 0.006;
    let uvv = center.x * center.y * center.y;
    var next = vec2<f32>(
        center.x + 0.19 * laplacian.x - uvv + feed * (1.0 - center.x),
        center.y + 0.095 * laplacian.y + uvv - (feed + kill) * center.y,
    );
    let energy = splat_band(id.x / 257u);
    let pulse = fract(sin(f32(id.x) * 12.9898
        + reaction_uniforms.viewport.w * 0.71) * 43758.5453);
    let injection = smoothstep(0.985, 1.0, pulse)
        * (energy * 0.55 + reaction_uniforms.audio.y * 0.8);
    next.y = max(next.y, injection);
    next.x = min(next.x, 1.0 - injection * 0.6);
    state_out[id.x].chemical = clamp(next, vec2<f32>(0.0), vec2<f32>(1.0));
}

@group(2) @binding(0) var<uniform> domain_uniforms: Uniforms;
@group(2) @binding(1) var<storage, read> domain_state: array<State>;
@group(2) @binding(2) var domain_image: texture_storage_2d<rgba8unorm, write>;

fn domain_load(x: i32, y: i32) -> vec2<f32> {
    return domain_state[field_index(x, y)].chemical;
}

@compute @workgroup_size(8, 8)
fn domain_transform(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= FIELD_W || id.y >= FIELD_H) {
        return;
    }
    let phase = domain_uniforms.viewport.z;
    let normalized = vec2<f32>(f32(id.x) / f32(FIELD_W), f32(id.y) / f32(FIELD_H))
        - vec2<f32>(0.5);
    let zoom = 1.0 + sin(phase * 0.37 + domain_uniforms.audio.x) * domain_uniforms.visual.x;
    let angle = sin(phase * 0.23) * domain_uniforms.visual.y
        + domain_uniforms.audio.y * 0.25;
    let rotated = vec2<f32>(
        normalized.x * cos(angle) - normalized.y * sin(angle),
        normalized.x * sin(angle) + normalized.y * cos(angle),
    ) / max(zoom, 0.25);
    let transformed = (rotated + vec2<f32>(0.5))
        * vec2<f32>(f32(FIELD_W), f32(FIELD_H));
    let x = i32(transformed.x);
    let y = i32(transformed.y);
    let grad = vec2<f32>(
        domain_load(x + 1, y).y - domain_load(x - 1, y).y,
        domain_load(x, y + 1).y - domain_load(x, y - 1).y,
    );
    let warp = grad * (9.0 + domain_uniforms.audio.x * 20.0);
    let warped = domain_load(x + i32(warp.x), y + i32(warp.y));
    let edge = length(grad) * 5.0;
    let value = clamp(
        warped.y * (1.5 + domain_uniforms.visual.z * 0.25)
            - warped.x * 0.28 + edge,
        0.0,
        1.0,
    );
    let mode = domain_uniforms.viewport.w;
    let heat = clamp(domain_uniforms.audio.z / 5000.0, 0.0, 1.0);
    let gradient_phase = value * 6.283 + normalized.x * 3.0
        + normalized.y * 2.0 + phase * 0.18 + heat;
    var color = 0.5 + 0.5 * cos(
        vec3<f32>(0.0, 2.094, 4.188) + gradient_phase
    );
    color *= 0.22 + value * 1.15;
    if (mode > 0.5) {
        let ink_phase = gradient_phase * 0.72 + edge * 4.0;
        color = (0.5 + 0.5 * cos(vec3<f32>(4.2, 2.0, 0.1) + ink_phase))
            * (0.08 + pow(value, 0.72));
    }
    if (mode > 1.5) {
        let lava_phase = gradient_phase * 1.3 + domain_uniforms.bands1.y * 3.0;
        color = (0.5 + 0.5 * cos(vec3<f32>(0.0, 1.3, 3.7) + lava_phase))
            * (0.16 + sqrt(value));
    }
    textureStore(
        domain_image,
        vec2<i32>(i32(id.x), i32(id.y)),
        vec4<f32>(color, 1.0),
    );
}

@group(3) @binding(0) var display_texture: texture_2d<f32>;
@group(3) @binding(1) var display_sampler: sampler;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> FullscreenOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: FullscreenOut;
    out.position = vec4<f32>(positions[vertex], 0.0, 1.0);
    out.uv = positions[vertex] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_fullscreen(input: FullscreenOut) -> @location(0) vec4<f32> {
    return textureSample(display_texture, display_sampler, input.uv);
}

@fragment
fn fs_background(_input: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.015, 0.02, 0.03, 1.0);
}
