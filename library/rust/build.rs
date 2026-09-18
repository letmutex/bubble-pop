use std::env;
use std::fs::{self, File};
use std::io::{Write, copy};
use std::path::{Path, PathBuf};
use std::process::Command;

const TFLITE_VERSION: &str = "2.17.1";

#[derive(Debug)]
struct PlatformSource {
    target_filename: &'static str,
    aliases: &'static [&'static str],
    download_url: &'static str,
}

fn get_platform_source(os: &str, arch: &str) -> Option<PlatformSource> {
    match (os, arch) {
        ("windows", "x86_64") => Some(PlatformSource {
            target_filename: "libtensorflowlite_c.dll",
            aliases: &["tensorflowlite_c.dll", "libLiteRt.dll"],
            download_url: "https://github.com/tphakala/tflite_c/releases/download/v2.17.1/tflite_c_v2.17.1_windows_amd64.zip",
        }),
        ("linux", "x86_64") => Some(PlatformSource {
            target_filename: "libtensorflowlite_c.so",
            aliases: &["libtensorflowlite_c.so.2.17.1", "libLiteRt.so"],
            download_url: "https://github.com/tphakala/tflite_c/releases/download/v2.17.1/tflite_c_v2.17.1_linux_amd64.tar.gz",
        }),
        ("linux", "aarch64") => Some(PlatformSource {
            target_filename: "libtensorflowlite_c.so",
            aliases: &["libtensorflowlite_c.so.2.17.1", "libLiteRt.so"],
            download_url: "https://github.com/tphakala/tflite_c/releases/download/v2.17.1/tflite_c_v2.17.1_linux_arm64.tar.gz",
        }),
        ("macos", "aarch64") => Some(PlatformSource {
            target_filename: "libtensorflowlite_c.dylib",
            aliases: &["libLiteRt.dylib"],
            download_url: "https://github.com/tphakala/tflite_c/releases/download/v2.17.1/tflite_c_v2.17.1_darwin_arm64.tar.gz",
        }),
        ("macos", "x86_64") => Some(PlatformSource {
            target_filename: "libtensorflowlite_c.dylib",
            aliases: &["libLiteRt.dylib"],
            download_url: "https://github.com/tphakala/tflite_c/releases/download/v2.17.0/tflite_c_v2.17.0_darwin_amd64.tar.gz",
        }),
        _ => None,
    }
}

fn get_cache_dir() -> PathBuf {
    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        PathBuf::from(cargo_home).join("tflite_cache")
    } else if let Ok(user_home) = env::var("USERPROFILE").or_else(|_| env::var("HOME")) {
        PathBuf::from(user_home).join(".cargo").join("tflite_cache")
    } else {
        env::temp_dir().join("tflite_cache")
    }
}

fn get_http_agent() -> ureq::Agent {
    let mut builder = ureq::builder().timeout(std::time::Duration::from_secs(15));
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

fn download_single_file(url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = dest_path.with_extension("tmp_download");
    let _ = fs::remove_file(&tmp_path);

    let agent = get_http_agent();
    let mut download_success = false;

    if let Ok(response) = agent.get(url).call()
        && let Ok(mut out) = File::create(&tmp_path)
    {
        let mut reader = response.into_reader();
        if copy(&mut reader, &mut out).is_ok() && out.flush().is_ok() {
            download_success = true;
        }
    }

    if !download_success {
        let curl_cmd = if cfg!(target_os = "windows") {
            "curl.exe"
        } else {
            "curl"
        };
        let status = Command::new(curl_cmd)
            .arg("-L")
            .arg("-o")
            .arg(&tmp_path)
            .arg(url)
            .status();

        download_success = matches!(status, Ok(s) if s.success());
    }

    if download_success && tmp_path.exists() {
        let len = fs::metadata(&tmp_path)?.len();
        if len > 100 * 1024 {
            if dest_path.exists() {
                let _ = fs::remove_file(dest_path);
            }
            fs::rename(&tmp_path, dest_path)?;
            return Ok(());
        }
    }

    let _ = fs::remove_file(&tmp_path);
    Err(format!("Failed to download file from {url}").into())
}

fn find_library_in_dir(dir: &Path, target_filename: &str) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    let direct = dir.join(target_filename);
    if direct.exists() && fs::metadata(&direct).map(|m| m.len()).unwrap_or(0) > 100 * 1024 {
        return Some(direct);
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_library_in_dir(&path, target_filename) {
                    return Some(found);
                }
            } else if let Some(file_name) = path.file_name().and_then(|f| f.to_str()) {
                let is_match = file_name == target_filename
                    || (target_filename.ends_with(".so")
                        && (file_name.starts_with("libtensorflowlite_c.so")
                            || file_name.starts_with("libLiteRt.so")))
                    || (target_filename.ends_with(".dylib")
                        && (file_name.contains("tensorflowlite_c")
                            || file_name.contains("LiteRt")))
                    || (target_filename.ends_with(".dll")
                        && (file_name.contains("tensorflowlite_c")
                            || file_name.contains("LiteRt")));

                if is_match && fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 100 * 1024 {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn extract_archive(
    archive_path: &Path,
    extract_to: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(extract_to)?;
    let tar_cmd = if cfg!(target_os = "windows") {
        "tar.exe"
    } else {
        "tar"
    };

    let status = Command::new(tar_cmd)
        .arg("-xf")
        .arg(archive_path)
        .arg("-C")
        .arg(extract_to)
        .status();

    if matches!(status, Ok(s) if s.success()) {
        return Ok(());
    }

    // Windows fallback for zip if tar failed
    #[cfg(target_os = "windows")]
    {
        let ps_cmd = format!(
            "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
            archive_path.display(),
            extract_to.display()
        );
        let ps_status = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg(&ps_cmd)
            .status();

        if matches!(ps_status, Ok(s) if s.success()) {
            return Ok(());
        }
    }

    Err(format!("Failed to extract archive {}", archive_path.display()).into())
}

fn fetch_platform_binary(
    source: &PlatformSource,
    cache_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cached_target = cache_dir.join(source.target_filename);
    if cached_target.exists()
        && fs::metadata(&cached_target).map(|m| m.len()).unwrap_or(0) > 100 * 1024
    {
        return Ok(cached_target);
    }

    let url = source.download_url;
    if url.is_empty() {
        return Err(format!(
            "No prebuilt download URL configured for {}",
            source.target_filename
        )
        .into());
    }

    let is_archive = url.ends_with(".tar.gz") || url.ends_with(".tgz") || url.ends_with(".zip");
    let download_name = if is_archive {
        url.split('/').next_back().unwrap_or("archive.tmp")
    } else {
        source.target_filename
    };
    let downloaded_file = cache_dir.join(download_name);

    let file_missing_or_small = !downloaded_file.exists()
        || fs::metadata(&downloaded_file).map(|m| m.len()).unwrap_or(0) <= 100 * 1024;

    if file_missing_or_small {
        download_single_file(url, &downloaded_file)?;
    }

    if is_archive {
        let extract_dir = cache_dir.join(format!("extract_{download_name}"));
        extract_archive(&downloaded_file, &extract_dir)?;
        if let Some(found_lib) = find_library_in_dir(&extract_dir, source.target_filename) {
            fs::copy(&found_lib, &cached_target)?;
            return Ok(cached_target);
        }
        return Err(format!(
            "Could not find {} inside {}",
            source.target_filename,
            downloaded_file.display()
        )
        .into());
    } else if downloaded_file.exists() {
        if downloaded_file != cached_target {
            fs::copy(&downloaded_file, &cached_target)?;
        }
        return Ok(cached_target);
    }

    Err(format!("Failed to retrieve binary for {}", source.target_filename).into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_PATH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let platform_source = match get_platform_source(&target_os, &target_arch) {
        Some(info) => info,
        None => {
            println!(
                "cargo:warning=Target {}-{} has no prebuilt TensorFlow Lite C binary download configured. Assuming system-provided library.",
                target_os, target_arch
            );
            return Ok(());
        }
    };

    let target_lib_path = out_dir.join(platform_source.target_filename);

    // 1. Check if user provided an override directory via environment variable
    let custom_dir = env::var("TFLITE_LIB_DIR").ok().map(PathBuf::from);

    let effective_lib_path = if let Some(dir) = custom_dir {
        let src = dir.join(platform_source.target_filename);
        if src.exists() {
            fs::copy(&src, &target_lib_path)?;
            target_lib_path.clone()
        } else {
            src
        }
    } else {
        // 2. Use persistent shared cache directory across all build profiles
        let cache_platform_dir = get_cache_dir()
            .join(TFLITE_VERSION)
            .join(format!("{target_os}_{target_arch}"));

        let cached_lib = fetch_platform_binary(&platform_source, &cache_platform_dir)?;
        fs::copy(&cached_lib, &target_lib_path)?;
        target_lib_path.clone()
    };

    // 3. Create aliases in OUT_DIR so dynamic loaders find the expected name under all conventions
    for alias in platform_source.aliases {
        let alias_path = out_dir.join(alias);
        if !alias_path.exists() && target_lib_path.exists() {
            let _ = fs::copy(&target_lib_path, &alias_path);
        }
    }

    if let Some(parent) = effective_lib_path.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    println!(
        "cargo:rustc-env=TFLITE_BUILTIN_LIB_PATH={}",
        effective_lib_path.display()
    );

    Ok(())
}
