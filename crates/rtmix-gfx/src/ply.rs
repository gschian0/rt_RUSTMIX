//! 3D Gaussian splat PLY loader (ASCII and binary little-endian).

use crate::slice::Gaussian3;

#[derive(Debug)]
pub struct PlyError {
    pub message: String,
}

impl PlyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Float,
    Double,
    Uchar,
    Int,
}

struct Prop {
    name: String,
    kind: Kind,
}

pub fn load_ply(bytes: &[u8]) -> Result<Vec<Gaussian3>, PlyError> {
    let header_end =
        find_header_end(bytes).ok_or_else(|| PlyError::new("PLY header is truncated"))?;
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| PlyError::new("PLY header is not utf-8"))?;
    let mut format_binary = false;
    let mut count = 0usize;
    let mut props = Vec::new();
    let mut in_vertex = false;
    for line in header.lines() {
        let line = line.trim();
        if line.starts_with("format ") {
            format_binary = line.contains("binary_little_endian");
            if line.contains("binary_big_endian") {
                return Err(PlyError::new("big-endian PLY is not supported"));
            }
        } else if line.starts_with("element ") {
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            let name = parts.next().unwrap_or("");
            let n = parts.next().unwrap_or("0").parse::<usize>().unwrap_or(0);
            in_vertex = name == "vertex";
            if in_vertex {
                count = n;
            }
        } else if line.starts_with("property ") && in_vertex {
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            let kind = match parts.next().unwrap_or("") {
                "float" | "float32" => Kind::Float,
                "double" | "float64" => Kind::Double,
                "uchar" | "uint8" => Kind::Uchar,
                "int" | "int32" => Kind::Int,
                other => return Err(PlyError::new(format!("unsupported property {other}"))),
            };
            let name = parts.next().unwrap_or("").to_string();
            props.push(Prop { name, kind });
        }
    }
    let body = &bytes[header_end..];
    if format_binary {
        read_binary(body, count, &props)
    } else {
        let text =
            std::str::from_utf8(body).map_err(|_| PlyError::new("ASCII PLY body is not utf-8"))?;
        read_ascii(text, count, &props)
    }
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    let marker = b"end_header";
    let pos = bytes.windows(marker.len()).position(|w| w == marker)?;
    let mut i = pos + marker.len();
    if bytes.get(i) == Some(&b'\r') {
        i += 1;
    }
    if bytes.get(i) == Some(&b'\n') {
        i += 1;
    }
    Some(i)
}

fn read_ascii(text: &str, count: usize, props: &[Prop]) -> Result<Vec<Gaussian3>, PlyError> {
    let mut numbers = text.split_whitespace();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut values = Vec::with_capacity(props.len());
        for prop in props {
            let token = numbers
                .next()
                .ok_or_else(|| PlyError::new("truncated ASCII vertex"))?;
            values.push(parse_prop(token, prop.kind)?);
        }
        out.push(assemble(props, &values));
    }
    Ok(out)
}

fn read_binary(mut body: &[u8], count: usize, props: &[Prop]) -> Result<Vec<Gaussian3>, PlyError> {
    let stride: usize = props.iter().map(|p| prop_size(p.kind)).sum();
    if body.len() < stride * count {
        return Err(PlyError::new("truncated binary PLY"));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut values = Vec::with_capacity(props.len());
        for prop in props {
            let size = prop_size(prop.kind);
            let (chunk, rest) = body.split_at(size);
            body = rest;
            values.push(read_prop(chunk, prop.kind)?);
        }
        out.push(assemble(props, &values));
    }
    Ok(out)
}

fn prop_size(kind: Kind) -> usize {
    match kind {
        Kind::Float | Kind::Int => 4,
        Kind::Double => 8,
        Kind::Uchar => 1,
    }
}

fn parse_prop(token: &str, kind: Kind) -> Result<f32, PlyError> {
    let value = token
        .parse::<f64>()
        .map_err(|_| PlyError::new("bad numeric property"))?;
    let _ = kind;
    Ok(value as f32)
}

fn read_prop(chunk: &[u8], kind: Kind) -> Result<f32, PlyError> {
    match kind {
        Kind::Float => {
            let bytes: [u8; 4] = chunk.try_into().map_err(|_| PlyError::new("short float"))?;
            Ok(f32::from_le_bytes(bytes))
        }
        Kind::Double => {
            let bytes: [u8; 8] = chunk
                .try_into()
                .map_err(|_| PlyError::new("short double"))?;
            Ok(f64::from_le_bytes(bytes) as f32)
        }
        Kind::Int => {
            let bytes: [u8; 4] = chunk.try_into().map_err(|_| PlyError::new("short int"))?;
            Ok(i32::from_le_bytes(bytes) as f32)
        }
        Kind::Uchar => Ok(chunk.first().copied().unwrap_or(0) as f32),
    }
}

fn assemble(props: &[Prop], values: &[f32]) -> Gaussian3 {
    let get = |name: &str| -> f32 {
        props
            .iter()
            .position(|p| p.name == name)
            .map(|i| values[i])
            .unwrap_or(0.0)
    };
    let sh_c0 = 0.28209479177387814;
    let opacity = sigmoid(get("opacity"));
    Gaussian3 {
        mean: [get("x"), get("y"), get("z")],
        cov: covariance_from_scale_rot(
            [
                libm::expf(get("scale_0")),
                libm::expf(get("scale_1")),
                libm::expf(get("scale_2")),
            ],
            [get("rot_0"), get("rot_1"), get("rot_2"), get("rot_3")],
        ),
        color: [
            (0.5 + sh_c0 * get("f_dc_0")).clamp(0.0, 1.0),
            (0.5 + sh_c0 * get("f_dc_1")).clamp(0.0, 1.0),
            (0.5 + sh_c0 * get("f_dc_2")).clamp(0.0, 1.0),
        ],
        opacity,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

pub(crate) fn covariance_from_scale_rot(scale: [f32; 3], rot: [f32; 4]) -> [f32; 9] {
    let mut q = rot;
    let n = libm::sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).max(1.0e-8);
    for v in &mut q {
        *v /= n;
    }
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ];
    let mut rs = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            rs[row * 3 + col] = r[row * 3 + col] * scale[col];
        }
    }
    let mut cov = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += rs[row * 3 + k] * rs[col * 3 + k];
            }
            cov[row * 3 + col] = acc;
        }
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ascii_ply() {
        let text = b"ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0.25 -0.5 1.5 0.0 0.0 0.0 0.0 1 0 0 0 0 0 0
";
        let splats = load_ply(text).unwrap();
        assert_eq!(splats.len(), 1);
        assert!((splats[0].mean[0] - 0.25).abs() < 1.0e-6);
        assert!((splats[0].mean[1] + 0.5).abs() < 1.0e-6);
        assert!((splats[0].mean[2] - 1.5).abs() < 1.0e-6);
        assert!((splats[0].opacity - 0.5).abs() < 1.0e-4);
        assert!((splats[0].color[0] - 0.5).abs() < 1.0e-4);
    }

    #[test]
    fn parses_binary_little_endian_ply() {
        let mut bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float opacity\nend_header\n".to_vec();
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-1.0f32).to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let splats = load_ply(&bytes).unwrap();
        assert!((splats[0].mean[0] - 0.5).abs() < 1.0e-6);
        assert!((splats[0].mean[1] + 1.0).abs() < 1.0e-6);
        assert!((splats[0].opacity - 0.5).abs() < 1.0e-4);
    }
}
