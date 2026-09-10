package letmutex.bubblepop

import android.content.Context
import android.graphics.Bitmap
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Creates one independently owned BubblePop LiteRT runtime. */
@Suppress("FunctionName")
fun BubblePopRuntime(
    context: Context,
    backend: LiteRtBackend,
): BubbleRuntime = AndroidBubblePopRuntime(context.applicationContext, backend)

private class AndroidBubblePopRuntime(
    context: Context,
    backend: LiteRtBackend,
) : BubbleRuntime {
    override val name: String = "BubblePop ${backend.name}"

    private val nativeRuntime = NativeBubblePopRuntime(
        context = context,
        runtimeId = backend.nativeId(),
        assetPath = MODEL_ASSET_PATH,
        modelToken = MODEL_TOKEN,
    )
    private val nativeExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "BubblePop${backend.name}Runtime")
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
        return when (image) {
            is Bitmap -> detectBitmap(image, confidenceThreshold)
            is IntArray -> {
                checkAvailable()
                if (!status.isReady) currentStatus = RuntimeStatus(RuntimeState.PREWARMING)
                try {
                    onNativeThread {
                        nativeRuntime.detect(image, width, height, confidenceThreshold).let {
                            it.toInferencePayload()
                        }
                    }.also { updateIfAvailable(RuntimeStatus(RuntimeState.READY)) }
                } catch (error: RuntimeException) {
                    updateIfAvailable(RuntimeStatus(RuntimeState.FAILED, error = error.message))
                    throw error
                }
            }
            else -> throw IllegalArgumentException("Unsupported image type: ${image::class}")
        }
    }

    private fun detectBitmap(
        bitmap: Bitmap,
        confidenceThreshold: Float,
    ): InferencePayload {
        checkAvailable()
        if (!status.isReady) currentStatus = RuntimeStatus(RuntimeState.PREWARMING)
        return try {
            onNativeThread {
                nativeRuntime.detectBitmap(bitmap, confidenceThreshold).let {
                    it.toInferencePayload()
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

private const val MODEL_ASSET_PATH = "model_fp16.tflite"
private const val MODEL_TOKEN = "mobilev3_unet"

// Order matches MakePayload in runtime_native.cpp.
private val NATIVE_BUBBLEPOP_PHASES = listOf(
    Phase.RuntimeInitialization,
    Phase.Preprocessing,
    Phase.Inference,
    Phase.ComponentsScoring,
    Phase.ContoursSimplification,
    Phase.ResultPacking,
)

private fun NativeBubblePopDetectionPayload.toInferencePayload(): InferencePayload {
    check(phaseMs.size == NATIVE_BUBBLEPOP_PHASES.size) { "Unexpected native timing payload" }
    return InferencePayload(packedBubbles, inferenceMs, NATIVE_BUBBLEPOP_PHASES.mapIndexed { index, phase ->
        PhaseTiming(phase, phaseMs[index])
    })
}
