use std::env;
use std::fs::{self, File};
use std::io::{Write, copy};
use std::path::{Path, PathBuf};

const LITERT_VERSION: &str = "2.2.0";
const LITERT_BASE_URL: &str = "https://storage.googleapis.com/litert/binaries/";

#[derive(Debug)]
struct PlatformInfo {
    dir_name: &'static str,
    file_name: &'static str,
}

fn get_platform_info(os: &str, arch: &str) -> Option<PlatformInfo> {
    match (os, arch) {
        ("windows", "x86_64") => Some(PlatformInfo {
            dir_name: "windows_x86_64",
            file_name: "libLiteRt.dll",
        }),
        ("linux", "x86_64") => Some(PlatformInfo {
            dir_name: "linux_x86_64",
            file_name: "libLiteRt.so",
        }),
        ("linux", "aarch64") => Some(PlatformInfo {
            dir_name: "linux_arm64",
            file_name: "libLiteRt.so",
        }),
        ("macos", "aarch64") => Some(PlatformInfo {
            dir_name: "macos_arm64",
            file_name: "libLiteRt.dylib",
        }),
        ("android", "aarch64") => Some(PlatformInfo {
            dir_name: "android_arm64",
            file_name: "libLiteRt.so",
        }),
        ("android", "x86_64") => Some(PlatformInfo {
            dir_name: "android_x86_64",
            file_name: "libLiteRt.so",
        }),
        _ => None,
    }
}

fn get_cache_dir() -> PathBuf {
    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        PathBuf::from(cargo_home).join("litert_cache")
    } else if let Ok(user_home) = env::var("USERPROFILE").or_else(|_| env::var("HOME")) {
        PathBuf::from(user_home).join(".cargo").join("litert_cache")
    } else {
        env::temp_dir().join("litert_cache")
    }
}

fn get_http_agent() -> ureq::Agent {
    let mut builder = ureq::builder().timeout(std::time::Duration::from_secs(60));
    let proxy_keys = [
        "https_proxy",
        "HTTPS_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ];
    for key in proxy_keys {
        if let Ok(val) = env::var(key)
            && !val.trim().is_empty()
            && let Ok(proxy) = ureq::Proxy::new(&val)
        {
            builder = builder.proxy(proxy);
            break;
        }
    }
    builder.build()
}

fn download_file(url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = dest_path.with_extension("tmp_download");
    let agent = get_http_agent();
    let resp = agent.get(url).call();

    let download_success = match resp {
        Ok(response) => {
            let mut reader = response.into_reader();
            let mut out = File::create(&tmp_path)?;
            copy(&mut reader, &mut out)?;
            out.flush()?;
            true
        }
        Err(_e) => {
            let curl_cmd = if cfg!(target_os = "windows") {
                "curl.exe"
            } else {
                "curl"
            };
            let status = std::process::Command::new(curl_cmd)
                .arg("-L")
                .arg("-o")
                .arg(&tmp_path)
                .arg(url)
                .status();

            matches!(status, Ok(s) if s.success())
        }
    };

    if download_success && tmp_path.exists() {
        let len = fs::metadata(&tmp_path)?.len();
        if len > 1024 * 1024 {
            if dest_path.exists() {
                let _ = fs::remove_file(dest_path);
            }
            fs::rename(&tmp_path, dest_path)?;
            return Ok(());
        }
    }

    let _ = fs::remove_file(&tmp_path);
    Err(format!("Failed to download valid LiteRT binary from {url}").into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LITERT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LIB_PATH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let platform_info = match get_platform_info(&target_os, &target_arch) {
        Some(info) => info,
        None => {
            println!(
                "cargo:warning=Target {}-{} has no official prebuilt LiteRT binary download configured. Assuming system-provided library.",
                target_os, target_arch
            );
            return Ok(());
        }
    };

    let target_file_name = platform_info.file_name;
    let target_lib_path = out_dir.join(target_file_name);

    // 1. Check if user provided an override directory via environment variable
    let custom_dir = env::var("LITERT_LIB_DIR")
        .or_else(|_| env::var("TFLITE_LIB_DIR"))
        .ok()
        .map(PathBuf::from);

    let effective_lib_path = if let Some(dir) = custom_dir {
        let src = dir.join(target_file_name);
        if src.exists() {
            fs::copy(&src, &target_lib_path)?;
            target_lib_path
        } else {
            src
        }
    } else {
        // 2. Use persistent shared cache directory across all build profiles
        let cache_file = get_cache_dir()
            .join(LITERT_VERSION)
            .join(platform_info.dir_name)
            .join(platform_info.file_name);

        let cache_valid = cache_file.exists()
            && fs::metadata(&cache_file).map(|m| m.len()).unwrap_or(0) > 1024 * 1024;

        if !cache_valid {
            let download_url = format!(
                "{}{}/{}/{}",
                LITERT_BASE_URL, LITERT_VERSION, platform_info.dir_name, platform_info.file_name
            );
            download_file(&download_url, &cache_file)?;
        }

        // Copy from persistent cache to current OUT_DIR
        fs::copy(&cache_file, &target_lib_path)?;
        target_lib_path
    };

    if let Some(parent) = effective_lib_path.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }

    println!(
        "cargo:rustc-env=LITERT_BUILTIN_LIB_PATH={}",
        effective_lib_path.display()
    );

    Ok(())
}
