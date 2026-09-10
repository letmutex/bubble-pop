package letmutex.bubblepop

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.InputStream
import kotlin.time.TimeMark
import kotlin.time.TimeSource

/**
 * Hardware execution backend used for LiteRT model inference.
 */
enum class LiteRtBackend {
    /** Executes model inference on CPU. */
    CPU,

    /** Executes model inference on GPU using FP16 precision. */
    GPU_FP16,
}

internal actual fun decodeAndDetect(
    detector: BubbleDetector,
    image: Any,
    width: Int,
    height: Int,
    runtime: BubbleRuntime,
    confidenceThreshold: Float,
): DetectionResult {
    return when (image) {
        is Bitmap -> detectBitmap(image, runtime, confidenceThreshold)
        is IntArray -> detectPixels(image, width, height, runtime, confidenceThreshold)
        is ByteArray -> {
            val bmp = BitmapFactory.decodeByteArray(image, 0, image.size)
                ?: throw IllegalArgumentException("Failed to decode Bitmap from ByteArray")
            try {
                detectBitmap(bmp, runtime, confidenceThreshold)
            } finally {
                bmp.recycle()
            }
        }
        is File -> {
            val bmp = BitmapFactory.decodeFile(image.absolutePath)
                ?: throw IllegalArgumentException("Failed to decode Bitmap from File: ${image.path}")
            try {
                detectBitmap(bmp, runtime, confidenceThreshold)
            } finally {
                bmp.recycle()
            }
        }
        is InputStream -> {
            val bmp = BitmapFactory.decodeStream(image)
                ?: throw IllegalArgumentException("Failed to decode Bitmap from InputStream")
            try {
                detectBitmap(bmp, runtime, confidenceThreshold)
            } finally {
                bmp.recycle()
            }
        }
        else -> throw IllegalArgumentException(
            "Unsupported image type: ${image::class.qualifiedName ?: image::class.simpleName}. Expected Bitmap, IntArray, ByteArray, File, or InputStream."
        )
    }
}

private fun detectBitmap(
    bitmap: Bitmap,
    runtime: BubbleRuntime,
    confidenceThreshold: Float = BubbleDetector.DEFAULT_CONFIDENCE_THRESHOLD,
): DetectionResult {
    val started = TimeSource.Monotonic.markNow()
    val image = if (bitmap.config == Bitmap.Config.ARGB_8888) {
        bitmap
    } else {
        bitmap.copy(Bitmap.Config.ARGB_8888, false)
    }

    try {
        val width = image.width
        val height = image.height
        require(confidenceThreshold.isFinite() && confidenceThreshold in 0f..1f) {
            "Confidence threshold must be between 0 and 1"
        }
        return measureDetection(started, width, height, runtime, started.elapsedMs()) {
            runtime.detect(image, width, height, confidenceThreshold)
        }
    } finally {
        if (image !== bitmap) {
            image.recycle()
        }
    }
}

private fun detectPixels(
    pixels: IntArray,
    width: Int,
    height: Int,
    runtime: BubbleRuntime,
    confidenceThreshold: Float,
): DetectionResult {
    require(width > 0 && height > 0) { "Image dimensions must be positive" }
    require(width <= Int.MAX_VALUE / height && pixels.size == width * height) {
        "ARGB pixel count must equal width * height"
    }
    require(confidenceThreshold.isFinite() && confidenceThreshold in 0f..1f) {
        "Confidence threshold must be between 0 and 1"
    }

    val started = TimeSource.Monotonic.markNow()
    return measureDetection(started, width, height, runtime, 0.0) {
        runtime.detect(pixels, width, height, confidenceThreshold)
    }
}
