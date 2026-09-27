fn main() {
    println!("cargo:rerun-if-changed=native/x264_encoder.c");
    println!("cargo:rerun-if-changed=native/x264_encoder.h");
    println!("cargo:rerun-if-env-changed=VCLASP_PATCHED_X264");
    println!("cargo:rustc-check-cfg=cfg(vclasp_patched_x264)");

    // Use pkg-config CLI to get x264 include path.
    let cflags = std::process::Command::new("pkg-config")
        .args(["--cflags", "x264"])
        .output()
        .expect("pkg-config not found; install pkg-config")
        .stdout;
    let cflags = String::from_utf8_lossy(&cflags).trim().to_string();

    let mut build = cc::Build::new();
    build.file("native/x264_encoder.c");
    if std::env::var_os("VCLASP_PATCHED_X264").is_some() {
        build.define("VCLASP_PATCHED_X264", None);
        println!("cargo:rustc-cfg=vclasp_patched_x264");
    }

    // Pass -I and -D flags from pkg-config.
    for flag in cflags.split_whitespace() {
        if let Some(val) = flag.strip_prefix("-I") {
            build.include(val);
        } else if let Some(val) = flag.strip_prefix("-D") {
            if let Some(eq_pos) = val.find('=') {
                build.define(&val[..eq_pos], Some(&val[eq_pos + 1..]));
            } else {
                build.define(val, None);
            }
        }
    }

    build.compile("x264_encoder");

    // Link libx264.
    let libs_stdout = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", "x264"])
        .output()
        .expect("pkg-config failed")
        .stdout;
    for flag in String::from_utf8_lossy(&libs_stdout).split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search={}", path);
        }
    }

    println!("cargo:rustc-link-lib=x264");
}
