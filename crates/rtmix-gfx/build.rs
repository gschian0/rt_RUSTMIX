fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_spirv)");
    println!("cargo:rerun-if-changed=shaders/raster.slang");
    println!("cargo:rerun-if-changed=shaders/slice4d.slang");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let Some(slangc) = find_slangc() else {
        return;
    };
    let jobs = [
        ("shaders/raster.slang", "project", "raster_project.spv"),
        ("shaders/raster.slang", "radix_histogram", "raster_radix.spv"),
        ("shaders/raster.slang", "blend_tile", "raster_blend.spv"),
        ("shaders/slice4d.slang", "slice4d", "slice4d.spv"),
    ];
    let mut ok = true;
    for (src, entry, dst) in jobs {
        let status = std::process::Command::new(&slangc)
            .current_dir(&manifest)
            .arg(src)
            .args(["-target", "spirv", "-stage", "compute", "-entry", entry, "-o"])
            .arg(format!("{out_dir}/{dst}"))
            .status();
        if !status.map(|code| code.success()).unwrap_or(false) {
            ok = false;
        }
    }
    if ok {
        println!("cargo:rustc-cfg=has_spirv");
    }
}

fn find_slangc() -> Option<String> {
    let output = std::process::Command::new("where").arg("slangc").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}
