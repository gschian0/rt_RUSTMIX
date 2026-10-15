// 4D-Rotor style slice: condition an XYZT Gaussian on score time t.

struct Frame {
    float4x4 view_proj;
    uint4 params;
    float4 time;
};

struct Gaussian4D {
    float4 mean;
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
    float4 color_opacity;
};

struct Gaussian3D {
    float4 mean_opacity;
    float4 c0;
    float4 c1;
    float4 color_c8;
};

[[vk::binding(0, 0)]]
cbuffer FrameBlock { Frame frame; };

[[vk::binding(1, 0)]]
StructuredBuffer<Gaussian4D> src;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gaussian3D> dst;

[shader("compute")]
[numthreads(64, 1, 1)]
void slice4d(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= frame.params.z) return;
    Gaussian4D g = src[index];
    float var_t = max(g.row3.w, 1e-8);
    float dt = frame.time.x - g.mean.w;
    float weight = exp(-0.5 * dt * dt / var_t);
    float s0 = g.row0.w;
    float s1 = g.row1.w;
    float s2 = g.row2.w;
    float3 mean = float3(
        g.mean.x + s0 / var_t * dt,
        g.mean.y + s1 / var_t * dt,
        g.mean.z + s2 / var_t * dt
    );
    float m00 = g.row0.x - s0 * s0 / var_t;
    float m01 = g.row0.y - s0 * s1 / var_t;
    float m02 = g.row0.z - s0 * s2 / var_t;
    float m11 = g.row1.y - s1 * s1 / var_t;
    float m12 = g.row1.z - s1 * s2 / var_t;
    float m22 = g.row2.z - s2 * s2 / var_t;
    dst[index].mean_opacity = float4(mean, g.color_opacity.w * weight);
    dst[index].c0 = float4(m00, m01, m02, m11);
    dst[index].c1 = float4(m12, m22, 0.0, 0.0);
    dst[index].color_c8 = float4(g.color_opacity.xyz, 0.0);
}
