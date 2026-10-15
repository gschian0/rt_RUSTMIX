// Forward 16x16 tile rasterizer in Slang.
// build.rs compiles this with slangc when it is on PATH. The binary also
// embeds the matching WGSL so a machine without slangc still builds.

struct Frame {
    float4x4 view_proj;
    uint4 params;
    float4 time;
};

struct Gaussian3D {
    float4 mean_opacity;
    float4 c0;
    float4 c1;
    float4 color_c8;
};

struct KeyValue {
    uint key;
    uint value;
};

[[vk::binding(0, 0)]]
cbuffer FrameBlock { Frame frame; };

[[vk::binding(1, 0)]]
StructuredBuffer<Gaussian3D> gaussians;

[[vk::binding(2, 0)]]
RWStructuredBuffer<KeyValue> records;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> hist;

[[vk::binding(4, 0)]]
RWStructuredBuffer<float4> image;

float4 clip_point(float3 mean) {
    return mul(frame.view_proj, float4(mean, 1.0));
}

float2 screen_of(float4 clip) {
    float2 ndc = clip.xy / max(abs(clip.w), 1e-6);
    float width = float(frame.params.x);
    float height = float(frame.params.y);
    float sx = (ndc.x * 0.5 + 0.5) * width;
    float sy = (1.0 - (ndc.y * 0.5 + 0.5)) * height;
    return float2(sx, sy);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= frame.params.z) return;
    Gaussian3D g = gaussians[index];
    float4 clip = clip_point(g.mean_opacity.xyz);
    if (clip.w <= 1e-4) {
        records[index].key = 0xffffffff;
        records[index].value = index;
        return;
    }
    float2 screen = screen_of(clip);
    uint tile = frame.params.w;
    uint tiles_x = (frame.params.x + tile - 1) / tile;
    uint tx = uint(clamp(screen.x / float(tile), 0.0, float(tiles_x - 1)));
    float depth = clamp(clip.z / clip.w, 0.0, 1.0);
    records[index].key = (tx << 16) | uint(depth * 65535.0);
    records[index].value = index;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void radix_histogram(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= frame.params.z) return;
    uint shift = uint(frame.time.y);
    uint digit = (records[index].key >> shift) & 255;
    InterlockedAdd(hist[digit], 1);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void radix_prefix(uint3 tid : SV_DispatchThreadID) {
    uint sum = 0;
    for (uint i = 0; i < 256; i++) {
        uint value = hist[i];
        hist[i] = sum;
        sum += value;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void radix_scatter(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= frame.params.z) return;
    uint shift = uint(frame.time.y);
    uint digit = (records[index].key >> shift) & 255;
    uint dest;
    InterlockedAdd(hist[digit], 1, dest);
    KeyValue rec = records[index];
    records[dest].key = rec.key;
    records[dest].value = rec.value;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blend_tile(uint3 tid : SV_DispatchThreadID) {
    uint tile = frame.params.w;
    uint width = frame.params.x;
    uint height = frame.params.y;
    uint pix_x = tid.x;
    uint pix_y = tid.y;
    if (pix_x >= width || pix_y >= height) return;
    float3 color = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    uint count = frame.params.z;
    for (uint i = 0; i < count; i++) {
        uint id = records[i].value;
        if (records[i].key == 0xffffffff) continue;
        Gaussian3D g = gaussians[id];
        float4 clip = clip_point(g.mean_opacity.xyz);
        if (clip.w <= 1e-4) continue;
        float2 screen = screen_of(clip);
        float dx = float(pix_x) - screen.x;
        float dy = float(pix_y) - screen.y;
        float radius2 = max(g.c0.x, 1.0);
        float alpha = g.mean_opacity.w * exp(-(dx * dx + dy * dy) / radius2);
        float a = clamp(alpha, 0.0, 0.99);
        color += transmittance * a * g.color_c8.xyz;
        transmittance *= (1.0 - a);
        if (transmittance < 0.01) break;
    }
    image[pix_y * width + pix_x] = float4(color, 1.0 - transmittance);
}
