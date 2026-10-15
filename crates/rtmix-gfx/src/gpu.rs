//! wgpu pipelines. SPIR-V is used when `slangc` produced it at build time.
//! Otherwise the embedded WGSL module is compiled by wgpu/naga.

use std::borrow::Cow;

use crate::{RASTER_WGSL, SLICE_WGSL};

pub fn raster_shader_source() -> wgpu::ShaderSource<'static> {
    if let Some(words) = spirv_words(raster_spirv_bytes()) {
        wgpu::ShaderSource::SpirV(Cow::Owned(words))
    } else {
        wgpu::ShaderSource::Wgsl(Cow::Borrowed(RASTER_WGSL))
    }
}

pub fn slice_shader_source() -> wgpu::ShaderSource<'static> {
    if let Some(words) = spirv_words(slice_spirv_bytes()) {
        wgpu::ShaderSource::SpirV(Cow::Owned(words))
    } else {
        wgpu::ShaderSource::Wgsl(Cow::Borrowed(SLICE_WGSL))
    }
}

#[cfg(has_spirv)]
fn raster_spirv_bytes() -> Option<&'static [u8]> {
    Some(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/raster_project.spv"
    )))
}

#[cfg(not(has_spirv))]
fn raster_spirv_bytes() -> Option<&'static [u8]> {
    None
}

#[cfg(has_spirv)]
fn slice_spirv_bytes() -> Option<&'static [u8]> {
    Some(include_bytes!(concat!(env!("OUT_DIR"), "/slice4d.spv")))
}

#[cfg(not(has_spirv))]
fn slice_spirv_bytes() -> Option<&'static [u8]> {
    None
}

fn spirv_words(bytes: Option<&[u8]>) -> Option<Vec<u32>> {
    let bytes = bytes?;
    if bytes.len() < 4 {
        return None;
    }
    Some(
        bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(word)
            })
            .collect(),
    )
}

pub struct SplatRenderer {
    pub project: wgpu::ComputePipeline,
    pub radix_histogram: wgpu::ComputePipeline,
    pub radix_prefix: wgpu::ComputePipeline,
    pub radix_scatter: wgpu::ComputePipeline,
    pub blend: wgpu::ComputePipeline,
    pub slice: wgpu::ComputePipeline,
}

impl SplatRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let raster = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtmix-raster"),
            source: raster_shader_source(),
        });
        let slice = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtmix-slice4d"),
            source: slice_shader_source(),
        });
        let make = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            project: make(&raster, "project"),
            radix_histogram: make(&raster, "radix_histogram"),
            radix_prefix: make(&raster, "radix_prefix"),
            radix_scatter: make(&raster, "radix_scatter"),
            blend: make(&raster, "blend_tile"),
            slice: make(&slice, "slice4d"),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_modules_parse() {
        naga::front::wgsl::parse_str(crate::RASTER_WGSL).expect("raster wgsl");
        naga::front::wgsl::parse_str(crate::SLICE_WGSL).expect("slice wgsl");
    }
}
