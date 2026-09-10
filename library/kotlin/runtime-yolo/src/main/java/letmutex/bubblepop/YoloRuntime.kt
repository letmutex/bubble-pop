package letmutex.bubblepop

import android.content.Context
import android.graphics.Bitmap
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Creates one independently owned YOLO segmentation LiteRT runtime. */
@Suppress("FunctionName")
fun YoloRuntime(
    context: Context,
    backend: LiteRtBackend,
): BubbleRuntime = AndroidYoloRuntime(context.applicationContext, backend)

private class AndroidYoloRuntime(
    context: Context,
    backend: LiteRtBackend,
) : BubbleRuntime {
    override val name: String = "YOLO ${backend.name}"

    private val nativeRuntime = NativeYoloRuntime(
        context = context,
        runtimeId = backend.nativeId(),
        assetPath = MODEL_ASSET_PATH,
        modelToken = MODEL_TOKEN,
    )
    private val nativeExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "Yolo${backend.name}Runtime")
    }

    @Volatile
    private var currentStatus = RuntimeStatus(RuntimeState.NOT_INITIALIZED)

    override val status: RuntimeStatus
        get() = currentStatus

    override fun prewarm(invokeOnce: Boolean): RuntimeStatus {
        checkAvailable()
        currentStatus = RuntimeStatus(RuntimeState.PREWARMING)
        return try {
            val initializationMs = onNativeThread { nativeRuntime.prewarm(invokeOnce) }
            RuntimeStatus(RuntimeState.READY, initializationMs).also(::updateIfAvailable)
        } catch (error: RuntimeException) {
            RuntimeStatus(RuntimeState.FAILED, error = error.message).also(::updateIfAvailable)
        }
    }

    override fun detect(
        image: Any,
        width: Int,
        height: Int,
        confidenceThreshold: Float,
    ): InferencePayload {
        checkAvailable()
        if (!status.isReady) currentStatus = RuntimeStatus(RuntimeState.PREWARMING)
        val pixels = when (image) {
            is IntArray -> image
            is Bitmap -> IntArray(width * height).also { image.getPixels(it, 0, width, 0, 0, width, height) }
            else -> throw IllegalArgumentException("Unsupported image type: ${image::class}")
        }
        return try {
            onNativeThread {
                nativeRuntime.detect(pixels, width, height, confidenceThreshold).let {
                    InferencePayload(it.packedBubbles, it.inferenceMs)
                }
            }.also { updateIfAvailable(RuntimeStatus(RuntimeState.READY)) }
        } catch (error: RuntimeException) {
            updateIfAvailable(RuntimeStatus(RuntimeState.FAILED, error = error.message))
            throw error
        }
    }

    override fun release() {
        synchronized(this) {
            if (currentStatus.isReleased) return
            currentStatus = RuntimeStatus(RuntimeState.RELEASED)
        }
        try {
            onNativeThread { nativeRuntime.release() }
        } finally {
            nativeExecutor.shutdown()
        }
    }

    private fun checkAvailable() {
        check(!status.isReleased) { "$name runtime has been released" }
    }

    private fun updateIfAvailable(status: RuntimeStatus) {
        synchronized(this) {
            if (!currentStatus.isReleased) currentStatus = status
        }
    }

    private fun <T> onNativeThread(block: () -> T): T = try {
        nativeExecutor.submit<T> { block() }.get()
    } catch (error: InterruptedException) {
        Thread.currentThread().interrupt()
        throw IllegalStateException("Native inference was interrupted", error)
    } catch (error: ExecutionException) {
        val cause = error.cause ?: error
        throw IllegalStateException(cause.message ?: "Native inference failed", cause)
    }
}

private fun LiteRtBackend.nativeId(): Int = when (this) {
    LiteRtBackend.CPU -> 0
    LiteRtBackend.GPU_FP16 -> 1
}

private const val MODEL_ASSET_PATH = "manga109-segmentation-bubble-best_fp16.tflite"
private const val MODEL_TOKEN = "manga109_segmentation_fp16_v1"
