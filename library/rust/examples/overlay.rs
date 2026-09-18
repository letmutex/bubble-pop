use std::env;
use std::path::{Path, PathBuf};

/// Palette of distinct colors (RGB) for visualizing multiple bubble instances.
const COLOR_PALETTE: &[[u8; 3]] = &[
    [235, 54, 54],   // Coral Red
    [54, 162, 235],  // Sky Blue
    [75, 192, 192],  // Teal / Cyan
    [255, 159, 64],  // Orange
    [153, 102, 255], // Purple
    [255, 205, 86],  // Yellow
    [46, 204, 113],  // Emerald Green
    [231, 76, 60],   // Crimson
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Resolve input image path from CLI argument or default sample
    let input_arg = env::args().nth(1);
    let input_path = match input_arg.as_deref() {
        Some(path) => PathBuf::from(path),
        None => {
            let default_sample = Path::new("assets/page-1.jpg");
            if default_sample.exists() {
                default_sample.to_path_buf()
            } else {
                eprintln!("Usage: cargo run --example overlay -- <path_to_image>");
                eprintln!(
                    "Error: No image path provided and default 'assets/page-1.jpg' not found."
                );
                std::process::exit(1);
            }
        }
    };

    println!("Loading image: {}", input_path.display());
    let mut img = image::open(&input_path)?.to_rgb8();
    let (width, height) = (img.width(), img.height());

    // 2. Detect speech bubbles
    println!("Detecting speech bubbles...");
    let result = bubblepop::detect(&input_path)?;

    println!(
        "Detected {} speech bubble(s) in {:.2} ms (inference: {:.2} ms, image: {}x{})",
        result.bubbles.len(),
        result.total_ms,
        result.inference_ms,
        width,
        height
    );

    for (i, bubble) in result.bubbles.iter().enumerate() {
        println!(
            "  Bubble #{:<2}: confidence = {:.3}, mask_prob = {:.3}, area = {} px, points = {}",
            i + 1,
            bubble.confidence,
            bubble.mask_probability,
            bubble.area_pixels,
            bubble.points.len()
        );
    }

    // 3. Draw bubble overlays on the image
    for (i, bubble) in result.bubbles.iter().enumerate() {
        let color = COLOR_PALETTE[i % COLOR_PALETTE.len()];
        draw_bubble_overlay(
            &mut img,
            &bubble.points,
            color,
            0.30, // 30% semi-transparent fill
            color,
            2, // 2px thick border
        );
    }

    // 4. Derive output filename: <filename_stem>_overlay.jpg in current working directory
    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let output_path = PathBuf::from(format!("{file_stem}_overlay.jpg"));

    // 5. Save the overlay image
    img.save(&output_path)?;
    println!(
        "\nOverlay image saved successfully to: {}",
        output_path.display()
    );

    Ok(())
}

/// Draws a semi-transparent polygon fill and thick boundary contour on an RGB image.
fn draw_bubble_overlay(
    img: &mut image::RgbImage,
    points: &[bubblepop::BubblePoint],
    fill_color: [u8; 3],
    fill_alpha: f32,
    outline_color: [u8; 3],
    outline_thickness: i32,
) {
    if points.len() < 3 {
        return;
    }

    let (width, height) = (img.width(), img.height());
    let pixel_pts: Vec<(f32, f32)> = points
        .iter()
        .map(|p| p.to_pixel_coords(width, height))
        .collect();

    let n = pixel_pts.len();

    // --- 1. Semi-transparent polygon fill (scanline rasterization) ---
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for &(_, y) in &pixel_pts {
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    let start_y = (min_y.floor() as i32).max(0);
    let end_y = (max_y.ceil() as i32).min(height as i32 - 1);

    let mut intersections = Vec::with_capacity(16);

    for y_int in start_y..=end_y {
        let y = y_int as f32 + 0.5;
        intersections.clear();

        for i in 0..n {
            let (x1, y1) = pixel_pts[i];
            let (x2, y2) = pixel_pts[(i + 1) % n];

            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                let t = (y - y1) / (y2 - y1);
                let x = x1 + t * (x2 - x1);
                intersections.push(x);
            }
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for chunk in intersections.chunks_exact(2) {
            let x_start = (chunk[0].floor() as i32).max(0);
            let x_end = (chunk[1].ceil() as i32).min(width as i32 - 1);

            for x_int in x_start..=x_end {
                let pixel = img.get_pixel_mut(x_int as u32, y_int as u32);
                for c in 0..3 {
                    pixel[c] = ((1.0 - fill_alpha) * pixel[c] as f32
                        + fill_alpha * fill_color[c] as f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    // --- 2. Thick contour outline ---
    let r = outline_thickness;
    for i in 0..n {
        let (x1, y1) = pixel_pts[i];
        let (x2, y2) = pixel_pts[(i + 1) % n];

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = dx.hypot(dy).max(1.0);
        let steps = dist.ceil() as usize;

        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let cx = (x1 + t * dx).round() as i32;
            let cy = (y1 + t * dy).round() as i32;

            for dy_off in -r..=r {
                for dx_off in -r..=r {
                    if dx_off * dx_off + dy_off * dy_off <= r * r {
                        let px = cx + dx_off;
                        let py = cy + dy_off;
                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                            let pixel = img.get_pixel_mut(px as u32, py as u32);
                            *pixel = image::Rgb(outline_color);
                        }
                    }
                }
            }
        }
    }
}
