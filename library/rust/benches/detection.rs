use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

use bubblepop::{BubbleDetector, PixelFormat, RawPixelView};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 1920;
const FORMAT: PixelFormat = PixelFormat::Rgba8;

fn generate_random_pixels(size: usize) -> Vec<u8> {
    let mut state: u64 = 0x853c49e6748fea9b;
    let mut buffer = Vec::with_capacity(size);
    for _ in 0..size {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        buffer.push((state & 0xFF) as u8);
    }
    buffer
}

fn bench_detection(c: &mut Criterion) {
    let detector = BubbleDetector::new().expect("Failed to initialize BubbleDetector");
    detector.prewarm().expect("Failed to prewarm detector");

    let total_bytes = (WIDTH * HEIGHT * FORMAT.bytes_per_pixel() as u32) as usize;
    let pixels = generate_random_pixels(total_bytes);

    let mut group = c.benchmark_group("speech_bubble_detection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Bytes(total_bytes as u64));

    let mut avg_infer_ms = 0.0;
    let mut k = 0;

    group.bench_function("detect_raw_1280x1920_rgba8", |b| {
        b.iter(|| {
            let view = RawPixelView {
                pixels: black_box(&pixels),
                width: WIDTH,
                height: HEIGHT,
                format: FORMAT,
                stride_bytes: None,
            };
            let result = detector.detect_raw(view).unwrap();

            k += 1;
            avg_infer_ms = avg_infer_ms + (result.inference_ms - avg_infer_ms) / k as f64;

            black_box(result)
        });
    });

    println!("Avg inference latency: {}ms, runs: {}", avg_infer_ms, k);

    group.finish();
}

criterion_group!(benches, bench_detection);
criterion_main!(benches);
