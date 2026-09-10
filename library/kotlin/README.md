# BubblePop

Kotlin APIs for bubble detection, with separate BubblePop and YOLO LiteRT runtimes.
Currently, only Android target is supported.

## Runtimes

`runtime-bubblepop`: LiteRT + BubblePop FP16 model. Recommended runtime for mobile inference, small and fast.

`runtime-yolo`: LiteRT + [huyvux3005/manga109-segmentation-bubble](https://huggingface.co/huyvux3005/manga109-segmentation-bubble) FP16 model, based on YOLO11n-seg. Could produce better edge details, but is larger and slower.

There is a quick comparison table:

|  | BubblePop | YOLO11n-seg | 
| :--- | :--- | :--- |
| Format | LiteRT FP 16 | LiteRT FP 16 |
| Shared library size <br>(arm64-v8a) | 6.73 MB + 400KB | 6.73 MB + 400KB |
| Model weights size | 1.84 MB | 5.85 MB |
| Inference time (CPU) | ~110ms | ~920ms |
| Inference time (GPU) | ~40ms | ~310ms |
| Input size | 768x1024 | 1600x1600 |
| Mask quality | High | High+ |

*Results on Qualcomm Snapdragon 845*

## Usages

Add dependency:

```kotlin
// Core detector API
implementation("io.github.letmutex:bubblepop:1.0.0")

// Include model and runtime
implementation("io.github.letmutex:runtime-bubblepop:1.0.0")
// or use YOLO
implementation("io.github.letmutex:runtime-yolo:1.0.0")
```

Detect:

```kotlin
val detector = BubbleDetector()
val runtime = BubblePopRuntime(context, LiteRtBackend.CPU)
// for YOLO:
// val yoloRuntime = YoloRuntime(context, LiteRtBackend.CPU)

// Optional prewarm on startup/page load
runtime.prewarm()

val result = detector.detect(
    bitmap, 
    bitmap.width, 
    bitmap.height, 
    runtime,
)

val bubbles = result.bubbles // List<BubblePath>
val inferenceMs = result.inferenceMs
val totalMs = result.totalMs

runtime.release()
```

Each runtime owns its native resources and lifecycle.

## Structure

```text
library/kotlin/
|-- core/               KMP API for Android
|-- runtime-bubblepop/  BubblePop LiteRT JNI implementation
|-- runtime-yolo/       YOLO11n-seg LiteRT JNI implementation
`-- sample/
    `-- androidApp/     Android sample application
```

## Build the sample

```bash
./gradlew :sample:androidApp:assembleDebug
```
