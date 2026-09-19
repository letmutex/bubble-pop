use bubblepop::{BubbleDetector, Options, detect, detect_with_options};
use std::path::PathBuf;

fn sample_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/page-1.jpg")
}

#[test]
fn test_global_detect_file() {
    let img_path = sample_image_path();
    assert!(
        img_path.exists(),
        "Sample image does not exist: {:?}",
        img_path
    );

    let path_str = img_path.to_str().unwrap();
    let result = detect(path_str).expect("Global detect(file_path) should succeed");

    println!("Detected {} bubbles", result.bubbles.len());
    println!(
        "Inference: {:.2}ms, Total: {:.2}ms",
        result.inference_ms, result.total_ms
    );

    for (i, bubble) in result.bubbles.iter().enumerate() {
        println!(
            "Bubble #{}: conf={:.3}, mask_prob={:.3}, area={}px, points={}",
            i + 1,
            bubble.confidence,
            bubble.mask_probability,
            bubble.area_pixels,
            bubble.points.len()
        );
        let svg = bubble.to_svg_path(result.image_width, result.image_height);
        assert!(!svg.is_empty());
        assert!(bubble.confidence >= 0.60);
        assert!(bubble.points.len() >= 3);
    }

    assert!(
        !result.bubbles.is_empty(),
        "Should detect bubbles on sample manga page"
    );
}

#[test]
fn test_detect_bytes() {
    let img_path = sample_image_path();
    let bytes = std::fs::read(&img_path).expect("Read sample image");

    let result = detect(bytes.as_slice()).expect("detect(&[u8]) should succeed");
    assert!(!result.bubbles.is_empty());

    let result_vec = detect(bytes).expect("detect(Vec<u8>) should succeed");
    assert_eq!(result.bubbles.len(), result_vec.bubbles.len());
}

#[test]
fn test_detect_with_options() {
    let img_path = sample_image_path();
    let options = Options::default().with_confidence(0.70);

    let result = detect_with_options(img_path.to_str().unwrap(), options)
        .expect("detect_with_options should succeed");

    for bubble in &result.bubbles {
        assert!(bubble.confidence >= 0.70);
    }
}

#[test]
fn test_detector_instance_prewarm() {
    let detector = BubbleDetector::new().expect("BubbleDetector::new");
    let prewarm_time = detector.prewarm().expect("prewarm");
    println!("Prewarm took: {:?}", prewarm_time);

    let img_path = sample_image_path();
    let result = detector
        .detect(&img_path)
        .expect("detector.detect(&PathBuf)");
    assert!(!result.bubbles.is_empty());
}

#[test]
fn test_invalid_dimensions_error() {
    let detector = BubbleDetector::new().expect("BubbleDetector::new");
    let raw = bubblepop::RawPixelView {
        pixels: &[0u8; 100],
        width: 0,
        height: 100,
        format: bubblepop::PixelFormat::Rgb8,
        stride_bytes: None,
    };
    let err = detector.detect_raw(raw).unwrap_err();
    assert!(matches!(err, bubblepop::BubblePopError::InvalidInput(_)));
}

#[test]
fn test_buffer_too_short_error() {
    let detector = BubbleDetector::new().expect("BubbleDetector::new");
    let raw = bubblepop::RawPixelView {
        pixels: &[0u8; 10],
        width: 100,
        height: 100,
        format: bubblepop::PixelFormat::Rgb8,
        stride_bytes: None,
    };
    let err = detector.detect_raw(raw).unwrap_err();
    assert!(matches!(err, bubblepop::BubblePopError::InvalidInput(_)));
}

#[test]
fn test_raw_pixel_view_into_image_source() {
    let detector = BubbleDetector::new().expect("BubbleDetector::new");
    let width = 50;
    let height = 50;
    let pixels = vec![255u8; width * height * 4];
    let raw = bubblepop::RawPixelView {
        pixels: &pixels,
        width: width as u32,
        height: height as u32,
        format: bubblepop::PixelFormat::Rgba8,
        stride_bytes: Some(width * 4),
    };
    // Should pass directly via IntoImageSource
    let result = detector.detect(raw).expect("detect(RawPixelView)");
    assert_eq!(result.image_width, width as u32);
    assert_eq!(result.image_height, height as u32);
}

#[test]
fn test_from_model_bytes_caller_drop_safety() {
    let detector = {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.tflite");
        let bytes = std::fs::read(model_path).expect("Read model.tflite");
        BubbleDetector::from_model_bytes(&bytes, Options::default())
            .expect("from_model_bytes should succeed")
        // `bytes` is dropped right here at the end of block!
    };

    let img_path = sample_image_path();
    let result = detector
        .detect(&img_path)
        .expect("detect after model bytes dropped");
    assert!(!result.bubbles.is_empty());
}

#[test]
fn test_multithreaded_concurrent_detect() {
    use std::sync::Arc;
    let detector = Arc::new(BubbleDetector::new().expect("BubbleDetector::new"));
    let img_path = sample_image_path();
    let bytes = Arc::new(std::fs::read(&img_path).expect("Read sample image"));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let det = Arc::clone(&detector);
        let b = Arc::clone(&bytes);
        handles.push(std::thread::spawn(move || {
            let res = det
                .detect(b.as_slice())
                .expect("Concurrent detect should succeed");
            assert!(!res.bubbles.is_empty());
        }));
    }

    for h in handles {
        h.join().expect("Thread join");
    }
}
