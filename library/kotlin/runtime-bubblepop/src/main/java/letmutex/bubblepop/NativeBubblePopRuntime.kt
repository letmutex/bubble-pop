package letmutex.bubblepop

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap

/** Raw result returned by one native Android LiteRT runtime. */
internal data class NativeBubblePopDetectionPayload(
    val packedBubbles: FloatArray,
    val inferenceMs: Double,
    val nativeTotalMs: Double,
    val phaseMs: DoubleArray,
)

/** Internal JNI owner for one CPU or GPU interpreter. */
internal class NativeBubblePopRuntime(
    context: Context,
    runtimeId: Int,
    assetPath: String,
    modelToken: String,
) {
    private var handle = nativeCreate(
        context.assets,
        context.codeCacheDir.absolutePath,
        runtimeId,
        assetPath,
        modelToken,
    )

    @Synchronized
    fun detect(
        pixels: IntArray,
        width: Int,
        height: Int,
        confidenceThreshold: Float,
    ): NativeBubblePopDetectionPayload {
        checkAvailable()
        require(confidenceThreshold.isFinite() && confidenceThreshold in 0f..1f) {
            "Confidence threshold must be between 0 and 1"
        }
        return nativeDetect(handle, pixels, width, height, confidenceThreshold)
    }

    @Synchronized
    fun detectBitmap(
        bitmap: Bitmap,
        confidenceThreshold: Float,
    ): NativeBubblePopDetectionPayload {
        checkAvailable()
        require(confidenceThreshold.isFinite() && confidenceThreshold in 0f..1f) {
            "Confidence threshold must be between 0 and 1"
        }
        return nativeDetectBitmap(handle, bitmap, confidenceThreshold)
    }

    @Synchronized
    fun prewarm(invokeOnce: Boolean = true): Double {
        checkAvailable()
        return nativePrewarm(handle, invokeOnce)
    }

    /** Permanently releases this native handle. */
    @Synchronized
    fun release() {
        if (handle == 0L) return
        nativeClose(handle)
        handle = 0L
    }

    private fun checkAvailable() {
        check(handle != 0L) { "Native bubble runtime has been released" }
    }

    private external fun nativeCreate(
        assetManager: AssetManager,
        cacheDir: String,
        runtime: Int,
        assetPath: String,
        modelToken: String,
    ): Long

    private external fun nativeDetect(
        handle: Long,
        pixels: IntArray,
        width: Int,
        height: Int,
        confidenceThreshold: Float,
    ): NativeBubblePopDetectionPayload

    private external fun nativeDetectBitmap(
        handle: Long,
        bitmap: Bitmap,
        confidenceThreshold: Float,
    ): NativeBubblePopDetectionPayload

    private external fun nativePrewarm(
        handle: Long,
        invokeOnce: Boolean,
    ): Double

    private external fun nativeClose(handle: Long)

    private companion object {
        init {
            System.loadLibrary("bubblepop_runtime")
        }
    }
}
