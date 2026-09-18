use super::contour::{simplify_polygon_rdp, trace_contour};
use super::letterbox::{INPUT_HEIGHT, INPUT_WIDTH, Letterbox};
use crate::types::{BubblePath, BubblePoint};

const MASK_LOGIT_THRESHOLD: f32 = 0.0; // sigmoid(logit) >= 0.50
const MIN_AREA_PIXELS: f32 = 60.0;
const RDP_EPSILON_PX: f32 = 1.0;

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    let clamped = value.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Reusable buffer context for postprocessing to prevent heap reallocations.
#[derive(Default)]
pub struct PostprocessContext {
    visited_epoch: Vec<u32>,
    current_epoch: u32,
    component: Vec<u8>,
    queue: Vec<usize>,
}

pub struct PostprocessTimings {
    pub scoring_ms: f64,
    pub contour_ms: f64,
    pub packing_ms: f64,
}

pub fn extract_bubbles(
    ctx: &mut PostprocessContext,
    logits: &[f32],
    pixel_stride: usize,
    output_width: usize,
    output_height: usize,
    box_info: &Letterbox,
    confidence_threshold: f32,
) -> (Vec<BubblePath>, PostprocessTimings) {
    let scoring_start = std::time::Instant::now();
    let plane = output_width * output_height;

    if ctx.visited_epoch.len() != plane {
        ctx.visited_epoch = vec![0; plane];
        ctx.current_epoch = 0;
    }
    if ctx.component.len() != plane {
        ctx.component = vec![0; plane];
    }
    if ctx.queue.len() != plane {
        ctx.queue.resize(plane, 0);
    }

    ctx.current_epoch = ctx.current_epoch.wrapping_add(1);
    if ctx.current_epoch == 0 {
        ctx.visited_epoch.fill(0);
        ctx.current_epoch = 1;
    }
    let epoch = ctx.current_epoch;

    let mut bubbles = Vec::new();
    let mut total_contour_duration = std::time::Duration::ZERO;

    let output_scale_x = output_width as f32 / INPUT_WIDTH as f32;
    let output_scale_y = output_height as f32 / INPUT_HEIGHT as f32;
    let crop_x = box_info.pad_x as f32 * output_scale_x;
    let crop_y = box_info.pad_y as f32 * output_scale_y;
    let crop_width = box_info.resized_width as f32 * output_scale_x;
    let crop_height = box_info.resized_height as f32 * output_scale_y;

    let min_model_area =
        (MIN_AREA_PIXELS * box_info.scale * box_info.scale * output_scale_x * output_scale_y)
            .ceil()
            .max(1.0) as usize;

    for start in 0..plane {
        let mask_logit = logits[start * pixel_stride];
        if ctx.visited_epoch[start] == epoch || mask_logit < MASK_LOGIT_THRESHOLD {
            continue;
        }

        let mut head = 0;
        let mut tail = 0;
        let mut mask_sum = 0.0f64;
        let mut confidence_sum = 0.0f64;

        ctx.queue[tail] = start;
        tail += 1;
        ctx.visited_epoch[start] = epoch;

        while head < tail {
            let index = ctx.queue[head];
            head += 1;

            ctx.component[index] = 1;
            let m_logit = logits[index * pixel_stride];
            let c_logit = logits[index * pixel_stride + 2];
            mask_sum += sigmoid(m_logit) as f64;
            confidence_sum += sigmoid(c_logit) as f64;

            let x = (index % output_width) as i32;
            let y = (index / output_width) as i32;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || nx >= output_width as i32 || ny < 0 || ny >= output_height as i32 {
                        continue;
                    }
                    let next = ny as usize * output_width + nx as usize;
                    if ctx.visited_epoch[next] != epoch
                        && logits[next * pixel_stride] >= MASK_LOGIT_THRESHOLD
                    {
                        ctx.visited_epoch[next] = epoch;
                        ctx.queue[tail] = next;
                        tail += 1;
                    }
                }
            }
        }

        let mean_mask = (mask_sum / tail as f64) as f32;
        let score = ((mask_sum + confidence_sum) / (2.0 * tail as f64)) as f32;

        if tail >= min_model_area && score >= confidence_threshold {
            let contour_start = std::time::Instant::now();
            let raw_contour = trace_contour(
                &ctx.component,
                &ctx.queue,
                tail,
                output_width,
                output_height,
            );
            let simplified = simplify_polygon_rdp(&raw_contour, RDP_EPSILON_PX);

            let mut points = Vec::with_capacity(simplified.len());
            for pt in &simplified {
                let nx = ((pt.x as f32 + 0.5 - crop_x) / crop_width).clamp(0.0, 1.0);
                let ny = ((pt.y as f32 + 0.5 - crop_y) / crop_height).clamp(0.0, 1.0);
                points.push(BubblePoint::new(nx, ny));
            }

            let area_pixels = (tail as f32 / (box_info.scale * box_info.scale)).round() as u32;

            if points.len() >= 3 {
                bubbles.push(BubblePath {
                    points,
                    confidence: score,
                    mask_probability: mean_mask,
                    area_pixels,
                });
            }
            total_contour_duration += contour_start.elapsed();
        }

        for i in 0..tail {
            ctx.component[ctx.queue[i]] = 0;
        }
    }

    let scoring_duration = scoring_start
        .elapsed()
        .saturating_sub(total_contour_duration);

    let packing_start = std::time::Instant::now();
    bubbles.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let packing_duration = packing_start.elapsed();

    (
        bubbles,
        PostprocessTimings {
            scoring_ms: scoring_duration.as_secs_f64() * 1000.0,
            contour_ms: total_contour_duration.as_secs_f64() * 1000.0,
            packing_ms: packing_duration.as_secs_f64() * 1000.0,
        },
    )
}
