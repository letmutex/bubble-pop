# BubblePop Rust Library

Fast, lightweight speech bubble extraction for comics and manga in Rust.

## Usage

### Add Dependency

```toml
[dependencies]
bubblepop = "0.1.0"
```

### Quick Detection

```rust
use bubblepop::detect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Detect directly from file path
    let result = detect("image1.jpg")?;
    println!("Found {} bubbles in {:.2}ms", result.bubbles.len(), result.total_ms);

    Ok(())
}
```

### Custom Options & Instance API

```rust
use bubblepop::{BubbleDetector, Options, Backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::default()
        .with_backend(Backend::Cpu)
        .with_threads(4)
        .with_confidence(0.75);

    let detector = BubbleDetector::with_options(options)?;
    let _ = detector.detect("image1.jpg")?;
    let _ = detector.detect("image2.jpg")?;
    Ok(())
}
```

## Running Examples

An example demonstrating speech bubble detection with visualization overlays:

```bash
# Detect and generate overlay from the default sample image
cargo run --release --example overlay

# Or pass a custom image file
cargo run --release --example overlay -- path/to/page.jpg
```

This writes `<filename>_overlay.jpg` in the current working directory with semi-transparent bubble masks and thick boundary contours.
