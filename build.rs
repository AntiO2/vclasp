fn main() {
    println!("cargo:rerun-if-changed=native/x264_encoder.c");
    println!("cargo:rerun-if-changed=native/x264_encoder.h");
    println!("cargo:rerun-if-env-changed=VCLASP_PATCHED_X264");
    println!("cargo:rustc-check-cfg=cfg(vclasp_patched_x264)");

    // The default reader/planner build has no native codec dependency.
    if std::env::var_os("CARGO_FEATURE_FFMPEG").is_none() {
        return;
    }

    for (library, expected_major) in [
        ("libavcodec", 61),
        ("libavformat", 61),
        ("libavutil", 59),
        ("libswscale", 8),
    ] {
        let found = pkg_config::Config::new()
            .probe(library)
            .unwrap_or_else(|error| panic!("{library} development files are required: {error}"));
        let actual_major = found.version.split('.').next().unwrap_or_default();
        assert_eq!(
            actual_major,
            expected_major.to_string(),
            "{library} {} is incompatible with ffmpeg-next 7.1; expected major {expected_major}. Use one consistent FFmpeg 7.1 development/runtime installation (check PKG_CONFIG_PATH).",
            found.version
        );
    }

    let x264 = pkg_config::Config::new()
        .probe("x264")
        .unwrap_or_else(|error| panic!("x264 development files are required: {error}"));
    let mut build = cc::Build::new();
    build.file("native/x264_encoder.c");
    for path in &x264.include_paths {
        build.include(path);
    }
    for (key, value) in &x264.defines {
        build.define(key, value.as_deref());
    }
    if std::env::var_os("VCLASP_PATCHED_X264").is_some() {
        build.define("VCLASP_PATCHED_X264", None);
        println!("cargo:rustc-cfg=vclasp_patched_x264");
    }
    build.compile("x264_encoder");
}
