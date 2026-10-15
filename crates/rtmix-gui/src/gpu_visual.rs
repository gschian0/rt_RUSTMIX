use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use eframe::{
    egui_wgpu,
    egui_wgpu::wgpu::{self, util::DeviceExt},
};
use rtmix_gfx::{camera_eye, Camera, Gaussian3, Reactivity, Spectrum};

const FIELD_W: u32 = 512;
const FIELD_H: u32 = 288;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUniforms {
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    eye: [f32; 4],
    viewport: [f32; 4],
    audio: [f32; 4],
    visual: [f32; 4],
    bands0: [f32; 4],
    bands1: [f32; 4],
    bands2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplat {
    mean_opacity: [f32; 4],
    cov0_band: [f32; 4],
    cov1_pad: [f32; 4],
    cov2_density: [f32; 4],
    color: [f32; 4],
    direction: [f32; 4],
}

pub fn initialize(cc: &eframe::CreationContext<'_>) -> Option<egui_wgpu::RenderState> {
    let state = cc.wgpu_render_state.as_ref()?.clone();
    let resources = GpuVisualResources::new(&state.device, state.target_format);
    state.renderer.write().callback_resources.insert(resources);
    Some(state)
}

pub fn upload_splats(
    state: &egui_wgpu::RenderState,
    gaussians: &[Gaussian3],
    center: [f32; 3],
    radius: f32,
    source_count: usize,
) {
    let density = ((source_count.max(gaussians.len()) as f32 / gaussians.len().max(1) as f32)
        .sqrt())
    .clamp(1.0, 2.6);
    let radius = radius.max(1.0e-5);
    let mut splats = gaussians
        .iter()
        .map(|gaussian| {
            let mean = [gaussian.mean[0], -gaussian.mean[1], -gaussian.mean[2]];
            let mut direction = [
                mean[0] - center[0],
                mean[1] - center[1],
                mean[2] - center[2],
            ];
            normalize(&mut direction);
            let height = ((mean[1] - center[1]) / radius * 0.5 + 0.5).clamp(0.0, 1.0);
            let band = (height * 12.0).floor().min(11.0);
            let variance =
                ((gaussian.cov[0] + gaussian.cov[4] + gaussian.cov[8]) / 3.0).max(1.0e-10);
            let signs = [1.0, -1.0, -1.0];
            let mut covariance = gaussian.cov;
            for row in 0..3 {
                for column in 0..3 {
                    covariance[row * 3 + column] *= signs[row] * signs[column];
                }
            }
            GpuSplat {
                mean_opacity: [mean[0], mean[1], mean[2], gaussian.opacity],
                cov0_band: [covariance[0], covariance[1], covariance[2], band],
                cov1_pad: [covariance[3], covariance[4], covariance[5], variance.sqrt()],
                cov2_density: [covariance[6], covariance[7], covariance[8], density],
                color: [gaussian.color[0], gaussian.color[1], gaussian.color[2], 1.0],
                direction: [direction[0], direction[1], direction[2], 0.0],
            }
        })
        .collect::<Vec<_>>();
    splats.sort_by_key(|splat| morton_key(splat.mean_opacity, center, radius));
    for (index, splat) in splats.iter_mut().enumerate() {
        splat.cov1_pad[3] = index as f32;
    }

    let mut renderer = state.renderer.write();
    let resources: &mut GpuVisualResources = renderer.callback_resources.get_mut().unwrap();
    resources.upload_splats(&state.device, &splats);
}

pub fn scene_bounds(gaussians: &[Gaussian3]) -> ([f32; 3], f32) {
    if gaussians.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let mut center = [0.0f32; 3];
    let mut values = Vec::with_capacity(gaussians.len());
    for axis in 0..3 {
        values.clear();
        values.extend(gaussians.iter().map(|g| match axis {
            0 => g.mean[0],
            1 => -g.mean[1],
            _ => -g.mean[2],
        }));
        values.sort_by(f32::total_cmp);
        center[axis] = values[values.len() / 2];
    }
    values.clear();
    values.extend(gaussians.iter().map(|g| {
        let point = [g.mean[0], -g.mean[1], -g.mean[2]];
        let delta = [
            point[0] - center[0],
            point[1] - center[1],
            point[2] - center[2],
        ];
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
    }));
    values.sort_by(f32::total_cmp);
    let index = ((values.len() - 1) as f32 * 0.975).round() as usize;
    (center, values[index].max(1.0e-3))
}

pub fn splat_callback(
    rect: egui::Rect,
    mut camera: Camera,
    spectrum: Spectrum,
    reactivity: Reactivity,
    time: f32,
) -> egui::PaintCallback {
    let width = rect.width().max(1.0);
    let height = rect.height().max(1.0);
    camera.distance *=
        (0.82 + (time * 0.31 + spectrum.bands[1]).sin() * reactivity.zoom * 0.22).clamp(0.42, 1.25);
    let mut basis = camera_basis(camera);
    let roll = (time * 0.27 + spectrum.bands[7]).sin() * reactivity.twist
        + spectrum.flux * reactivity.twist * 0.18;
    let (sin_roll, cos_roll) = roll.sin_cos();
    let right = basis[0];
    let up = basis[1];
    basis[0] = [
        right[0] * cos_roll + up[0] * sin_roll,
        right[1] * cos_roll + up[1] * sin_roll,
        right[2] * cos_roll + up[2] * sin_roll,
    ];
    basis[1] = [
        up[0] * cos_roll - right[0] * sin_roll,
        up[1] * cos_roll - right[1] * sin_roll,
        up[2] * cos_roll - right[2] * sin_roll,
    ];
    let focal = (height * 0.5) / (camera.fov_y * 0.5).tan().max(1.0e-4);
    let uniforms = uniforms(
        basis,
        camera_eye(&camera),
        [width, height, focal, time],
        spectrum,
        [
            reactivity.displace * camera.distance * 0.12,
            reactivity.swell,
            reactivity.brightness,
            reactivity.point_size,
        ],
        reactivity.exposure,
    );
    egui_wgpu::Callback::new_paint_callback(
        rect,
        GpuCallback {
            kind: CallbackKind::Splat,
            uniforms,
        },
    )
}

pub fn drip_callback(
    rect: egui::Rect,
    mode: u32,
    spectrum: Spectrum,
    reactivity: Reactivity,
    time: f32,
) -> egui::PaintCallback {
    let uniforms = uniforms(
        [[1.0, 0.0, 0.0]; 3],
        [0.0; 3],
        [rect.width(), rect.height(), time, mode as f32],
        spectrum,
        [
            reactivity.zoom,
            reactivity.twist,
            reactivity.swell,
            reactivity.point_size,
        ],
        1.0,
    );
    egui_wgpu::Callback::new_paint_callback(
        rect,
        GpuCallback {
            kind: CallbackKind::Drip,
            uniforms,
        },
    )
}

fn uniforms(
    basis: [[f32; 3]; 3],
    eye: [f32; 3],
    viewport: [f32; 4],
    spectrum: Spectrum,
    visual: [f32; 4],
    exposure: f32,
) -> GpuUniforms {
    GpuUniforms {
        right: [basis[0][0], basis[0][1], basis[0][2], 0.0],
        up: [basis[1][0], basis[1][1], basis[1][2], 0.0],
        forward: [basis[2][0], basis[2][1], basis[2][2], 0.0],
        eye: [eye[0], eye[1], eye[2], 0.0],
        viewport,
        audio: [spectrum.level, spectrum.flux, spectrum.centroid, exposure],
        visual,
        bands0: spectrum.bands[0..4].try_into().unwrap(),
        bands1: spectrum.bands[4..8].try_into().unwrap(),
        bands2: spectrum.bands[8..12].try_into().unwrap(),
    }
}

#[derive(Clone, Copy)]
enum CallbackKind {
    Splat,
    Drip,
}

struct GpuCallback {
    kind: CallbackKind,
    uniforms: GpuUniforms,
}

impl egui_wgpu::CallbackTrait for GpuCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &GpuVisualResources = callback_resources.get().unwrap();
        queue.write_buffer(
            &resources.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms),
        );
        match self.kind {
            CallbackKind::Splat => resources.run_splat_reaction(encoder),
            CallbackKind::Drip => resources.run_differential_passes(encoder),
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &GpuVisualResources = callback_resources.get().unwrap();
        match self.kind {
            CallbackKind::Splat => resources.paint_splats(render_pass),
            CallbackKind::Drip => resources.paint_drip(render_pass),
        }
    }
}

struct GpuVisualResources {
    uniform_buffer: wgpu::Buffer,
    empty_bind_group: wgpu::BindGroup,
    splat_layout: wgpu::BindGroupLayout,
    splat_buffers: Vec<wgpu::Buffer>,
    splat_bind_groups: Vec<wgpu::BindGroup>,
    splat_counts: Vec<u32>,
    splat_state_a: wgpu::Buffer,
    splat_state_b: wgpu::Buffer,
    splat_reaction_layout: wgpu::BindGroupLayout,
    splat_reaction_pipeline: wgpu::ComputePipeline,
    splat_reaction_ab: wgpu::BindGroup,
    splat_reaction_ba: wgpu::BindGroup,
    splat_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    reaction_pipeline: wgpu::ComputePipeline,
    domain_pipeline: wgpu::ComputePipeline,
    reaction_ab: wgpu::BindGroup,
    reaction_ba: wgpu::BindGroup,
    domain_bind_group: wgpu::BindGroup,
    display_bind_group: wgpu::BindGroup,
    display_pipeline: wgpu::RenderPipeline,
}

impl GpuVisualResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtmix-gpu-visuals"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_visual.wgsl").into()),
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtmix-visual-uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtmix-empty-layout"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rtmix-empty-bind-group"),
            layout: &empty_layout,
            entries: &[],
        });

        let splat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtmix-splat-layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtmix-empty-splats"),
            contents: bytemuck::bytes_of(&GpuSplat::zeroed()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let initial_splat_state = [[1.0f32, 0.0f32]];
        let splat_state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtmix-splat-reaction-a"),
            contents: bytemuck::cast_slice(&initial_splat_state),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let splat_state_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtmix-splat-reaction-b"),
            contents: bytemuck::cast_slice(&initial_splat_state),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let splat_reaction_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rtmix-splat-reaction-layout"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let splat_reaction_ab = reaction_bind_group(
            device,
            &splat_reaction_layout,
            &uniform_buffer,
            &splat_state_a,
            &splat_state_b,
        );
        let splat_reaction_ba = reaction_bind_group(
            device,
            &splat_reaction_layout,
            &uniform_buffer,
            &splat_state_b,
            &splat_state_a,
        );
        let splat_reaction_pipeline = compute_pipeline(
            device,
            &shader,
            "splat_reaction",
            &[&empty_layout, &splat_reaction_layout],
        );
        let splat_bind_group = splat_bind_group(
            device,
            &splat_layout,
            &uniform_buffer,
            &splat_buffer,
            &splat_state_a,
        );
        let splat_pipeline = render_pipeline(
            device,
            &shader,
            "vs_splat",
            "fs_splat",
            target_format,
            &[&splat_layout],
            true,
        );
        let background_pipeline = render_pipeline(
            device,
            &shader,
            "vs_fullscreen",
            "fs_background",
            target_format,
            &[],
            false,
        );

        let state_len = (FIELD_W * FIELD_H) as usize;
        let mut initial = vec![[1.0f32, 0.0f32]; state_len];
        for seed in 0..32 {
            let cx = (seed * 73 + 31) % FIELD_W as usize;
            let cy = (seed * 47 + 19) % FIELD_H as usize;
            for oy in 0..12 {
                for ox in 0..12 {
                    let x = (cx + ox) % FIELD_W as usize;
                    let y = (cy + oy) % FIELD_H as usize;
                    initial[y * FIELD_W as usize + x] = [0.25, 0.9];
                }
            }
        }
        let state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtmix-reaction-a"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let state_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtmix-reaction-b"),
            size: (state_len * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let reaction_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtmix-reaction-layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let reaction_ab = reaction_bind_group(
            device,
            &reaction_layout,
            &uniform_buffer,
            &state_a,
            &state_b,
        );
        let reaction_ba = reaction_bind_group(
            device,
            &reaction_layout,
            &uniform_buffer,
            &state_b,
            &state_a,
        );
        let reaction_pipeline = compute_pipeline(
            device,
            &shader,
            "reaction",
            &[&empty_layout, &reaction_layout],
        );

        let display_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rtmix-domain-output"),
            size: wgpu::Extent3d {
                width: FIELD_W,
                height: FIELD_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let display_view = display_texture.create_view(&Default::default());
        let domain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtmix-domain-layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let domain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rtmix-domain-bind-group"),
            layout: &domain_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&display_view),
                },
            ],
        });
        let domain_pipeline = compute_pipeline(
            device,
            &shader,
            "domain_transform",
            &[&empty_layout, &empty_layout, &domain_layout],
        );

        let display_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtmix-display-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rtmix-display-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rtmix-display-bind-group"),
            layout: &display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let display_pipeline = render_pipeline(
            device,
            &shader,
            "vs_fullscreen",
            "fs_fullscreen",
            target_format,
            &[&empty_layout, &empty_layout, &empty_layout, &display_layout],
            false,
        );

        Self {
            uniform_buffer,
            empty_bind_group,
            splat_layout,
            splat_buffers: vec![splat_buffer],
            splat_bind_groups: vec![splat_bind_group],
            splat_counts: vec![0],
            splat_state_a,
            splat_state_b,
            splat_reaction_layout,
            splat_reaction_pipeline,
            splat_reaction_ab,
            splat_reaction_ba,
            splat_pipeline,
            background_pipeline,
            reaction_pipeline,
            domain_pipeline,
            reaction_ab,
            reaction_ba,
            domain_bind_group,
            display_bind_group,
            display_pipeline,
        }
    }

    fn upload_splats(&mut self, device: &wgpu::Device, splats: &[GpuSplat]) {
        self.splat_buffers.clear();
        self.splat_bind_groups.clear();
        self.splat_counts.clear();
        if splats.is_empty() {
            return;
        }
        let mut initial_state = vec![[1.0f32, 0.0f32]; splats.len()];
        for (index, state) in initial_state.iter_mut().enumerate() {
            let cluster = (index / 384) % 97;
            let speck = index.wrapping_mul(2_654_435_761) & 0x3fff;
            if cluster < 9 || speck < 48 {
                *state = [0.08, 0.96];
            }
        }
        self.splat_state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtmix-splat-reaction-a"),
            contents: bytemuck::cast_slice(&initial_state),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.splat_state_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtmix-splat-reaction-b"),
            size: (initial_state.len() * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.splat_reaction_ab = reaction_bind_group(
            device,
            &self.splat_reaction_layout,
            &self.uniform_buffer,
            &self.splat_state_a,
            &self.splat_state_b,
        );
        self.splat_reaction_ba = reaction_bind_group(
            device,
            &self.splat_reaction_layout,
            &self.uniform_buffer,
            &self.splat_state_b,
            &self.splat_state_a,
        );

        let max_per_binding = (device.limits().max_storage_buffer_binding_size as usize
            / std::mem::size_of::<GpuSplat>())
        .max(1);
        for chunk in splats.chunks(max_per_binding) {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rtmix-gpu-splat-chunk"),
                contents: bytemuck::cast_slice(chunk),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let bind_group = splat_bind_group(
                device,
                &self.splat_layout,
                &self.uniform_buffer,
                &buffer,
                &self.splat_state_a,
            );
            self.splat_counts.push(chunk.len() as u32);
            self.splat_bind_groups.push(bind_group);
            self.splat_buffers.push(buffer);
        }
    }

    fn run_splat_reaction(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.splat_counts.iter().copied().sum::<u32>();
        if count == 0 {
            return;
        }
        let workgroups = count.div_ceil(64);
        for (label, bind_group) in [
            ("rtmix-splat-reaction-a", &self.splat_reaction_ab),
            ("rtmix-splat-reaction-b", &self.splat_reaction_ba),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.splat_reaction_pipeline);
            pass.set_bind_group(0, &self.empty_bind_group, &[]);
            pass.set_bind_group(1, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    fn run_differential_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (FIELD_W.div_ceil(8), FIELD_H.div_ceil(8));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rtmix-reaction-pass-a"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reaction_pipeline);
            pass.set_bind_group(0, &self.empty_bind_group, &[]);
            pass.set_bind_group(1, &self.reaction_ab, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rtmix-reaction-pass-b"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reaction_pipeline);
            pass.set_bind_group(0, &self.empty_bind_group, &[]);
            pass.set_bind_group(1, &self.reaction_ba, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rtmix-domain-transform-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.domain_pipeline);
            pass.set_bind_group(0, &self.empty_bind_group, &[]);
            pass.set_bind_group(1, &self.empty_bind_group, &[]);
            pass.set_bind_group(2, &self.domain_bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
    }

    fn paint_splats(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.background_pipeline);
        render_pass.draw(0..3, 0..1);
        render_pass.set_pipeline(&self.splat_pipeline);
        for (bind_group, count) in self.splat_bind_groups.iter().zip(&self.splat_counts) {
            if *count == 0 {
                continue;
            }
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..6, 0..*count);
        }
    }

    fn paint_drip(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.display_pipeline);
        render_pass.set_bind_group(0, &self.empty_bind_group, &[]);
        render_pass.set_bind_group(1, &self.empty_bind_group, &[]);
        render_pass.set_bind_group(2, &self.empty_bind_group, &[]);
        render_pass.set_bind_group(3, &self.display_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

fn render_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    format: wgpu::TextureFormat,
    layouts: &[&wgpu::BindGroupLayout],
    blend: bool,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rtmix-render-pipeline-layout"),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rtmix-compute-pipeline-layout"),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuUniforms>() as u64),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn splat_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    splats: &wgpu::Buffer,
    state: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rtmix-splat-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: splats.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.as_entire_binding(),
            },
        ],
    })
}

fn reaction_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rtmix-reaction-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    })
}

fn camera_basis(camera: Camera) -> [[f32; 3]; 3] {
    let eye = camera_eye(&camera);
    let mut forward = [
        camera.target[0] - eye[0],
        camera.target[1] - eye[1],
        camera.target[2] - eye[2],
    ];
    normalize(&mut forward);
    let mut right = cross(forward, [0.0, 1.0, 0.0]);
    normalize(&mut right);
    let up = cross(right, forward);
    [right, up, forward]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn morton_key(mean: [f32; 4], center: [f32; 3], radius: f32) -> u32 {
    let quantize = |value: f32, axis: usize| {
        (((value - center[axis]) / radius * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0) as u32
    };
    let x = spread_bits(quantize(mean[0], 0));
    let y = spread_bits(quantize(mean[1], 1)) << 1;
    let z = spread_bits(quantize(mean[2], 2)) << 2;
    x | y | z
}

fn spread_bits(mut value: u32) -> u32 {
    value &= 0x3ff;
    value = (value | value << 16) & 0x0300_00ff;
    value = (value | value << 8) & 0x0300_f00f;
    value = (value | value << 4) & 0x030c_30c3;
    value = (value | value << 2) & 0x0924_9249;
    value
}

fn normalize(value: &mut [f32; 3]) {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
        .sqrt()
        .max(1.0e-8);
    for axis in value {
        *axis /= length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_bounds_ignore_isolated_reconstruction_debris() {
        let point = |x: f32| Gaussian3 {
            mean: [x, 0.0, 0.0],
            cov: [0.0; 9],
            color: [1.0; 3],
            opacity: 1.0,
        };
        let mut cloud = (0..200)
            .map(|index| point(index as f32 / 100.0 - 1.0))
            .collect::<Vec<_>>();
        cloud.push(point(1_000.0));
        let (_, radius) = scene_bounds(&cloud);
        assert!(radius < 2.0, "outlier inflated radius to {radius}");
    }
}
