use crate::input::PixelFormat;
use super::letterbox::{Letterbox, INPUT_HEIGHT, INPUT_WIDTH, MEAN, STD};

#[derive(Clone, Copy)]
struct XSample {
    x0: usize,
    x1: usize,
    wx: f32,
}

#[inline(always)]
fn get_grayscale(pixels: &[u8], offset: usize, format: PixelFormat) -> f32 {
    match format {
        PixelFormat::Rgba8 => {
            let r = pixels[offset] as f32;
            let g = pixels[offset + 1] as f32;
            let b = pixels[offset + 2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        PixelFormat::Rgb8 => {
            let r = pixels[offset] as f32;
            let g = pixels[offset + 1] as f32;
            let b = pixels[offset + 2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        PixelFormat::Bgra8 => {
            let b = pixels[offset] as f32;
            let g = pixels[offset + 1] as f32;
            let r = pixels[offset + 2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        PixelFormat::Bgr8 => {
            let b = pixels[offset] as f32;
            let g = pixels[offset + 1] as f32;
            let r = pixels[offset + 2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        PixelFormat::Argb8 => {
            let r = pixels[offset + 1] as f32;
            let g = pixels[offset + 2] as f32;
            let b = pixels[offset + 3] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        PixelFormat::Gray8 => {
            pixels[offset] as f32 / 255.0
        }
    }
}

/// Preprocesses raw source pixels into a normalized [1, 1024, 768, 1] (or flat 768*1024) NHWC float buffer.
pub fn preprocess(
    pixels: &[u8],
    source_width: usize,
    source_height: usize,
    stride_bytes: usize,
    format: PixelFormat,
    box_info: &Letterbox,
    destination: &mut [f32],
) {
    let white = (1.0 - MEAN) / STD;
    let bpp = format.bytes_per_pixel();

    // 1. Precalculate X-mapping table once per image
    let mut x_table = Vec::with_capacity(box_info.resized_width);
    for local_x in 0..box_info.resized_width {
        let sx = ((local_x as f32 + 0.5) * source_width as f32 / box_info.resized_width as f32 - 0.5)
            .clamp(0.0, (source_width - 1) as f32);
        let x0 = sx as usize;
        let x1 = (x0 + 1).min(source_width - 1);
        let wx = sx - x0 as f32;
        x_table.push(XSample { x0, x1, wx });
    }

    // 2. Fill top padding rows
    for y in 0..box_info.pad_y {
        let row_start = y * INPUT_WIDTH;
        destination[row_start..row_start + INPUT_WIDTH].fill(white);
    }

    // 3. Bilinear interpolation for image rows
    for local_y in 0..box_info.resized_height {
        let y = box_info.pad_y + local_y;
        let dst_row_start = y * INPUT_WIDTH;

        // Left padding
        if box_info.pad_x > 0 {
            destination[dst_row_start..dst_row_start + box_info.pad_x].fill(white);
        }

        let sy = ((local_y as f32 + 0.5) * source_height as f32 / box_info.resized_height as f32 - 0.5)
            .clamp(0.0, (source_height - 1) as f32);
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(source_height - 1);
        let wy = sy - y0 as f32;

        let row0_offset = y0 * stride_bytes;
        let row1_offset = y1 * stride_bytes;

        for (local_x, &xs) in x_table.iter().enumerate() {
            let g00 = get_grayscale(pixels, row0_offset + xs.x0 * bpp, format);
            let g10 = get_grayscale(pixels, row0_offset + xs.x1 * bpp, format);
            let g01 = get_grayscale(pixels, row1_offset + xs.x0 * bpp, format);
            let g11 = get_grayscale(pixels, row1_offset + xs.x1 * bpp, format);

            let top = g00 + (g10 - g00) * xs.wx;
            let bottom = g01 + (g11 - g01) * xs.wx;
            let val = top + (bottom - top) * wy;

            destination[dst_row_start + box_info.pad_x + local_x] = (val - MEAN) / STD;
        }

        // Right padding
        let right_start = box_info.pad_x + box_info.resized_width;
        if right_start < INPUT_WIDTH {
            destination[dst_row_start + right_start..dst_row_start + INPUT_WIDTH].fill(white);
        }
    }

    // 4. Fill bottom padding rows
    let pad_bottom_start = box_info.pad_y + box_info.resized_height;
    for y in pad_bottom_start..INPUT_HEIGHT {
        let row_start = y * INPUT_WIDTH;
        destination[row_start..row_start + INPUT_WIDTH].fill(white);
    }
}
