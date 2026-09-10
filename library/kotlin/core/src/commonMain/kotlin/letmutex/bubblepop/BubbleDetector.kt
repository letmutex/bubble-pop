package letmutex.bubblepop

import kotlin.math.roundToInt
import kotlin.time.TimeSource
import kotlin.time.TimeMark

/** A normalized coordinate in the source image. */
data class BubblePoint(
    val x: Float,
    val y: Float,
)

/** A closed contour in source-image normalized coordinates, plus model scores. */
data class BubblePath(
    val points: List<BubblePoint>,
    val confidence: Float,
    val maskProbability: Float,
    val areaPixels: Int,
)

/** Lifecycle states of an inference runtime. */
enum class RuntimeState {
    NOT_INITIALIZED,
    PREWARMING,
    READY,
    FAILED,
    RELEASED,
}

/** Current lifecycle status and initialization metrics of an inference runtime. */
data class RuntimeStatus(
    val state: RuntimeState,
    val initializationMs: Double? = null,
    val error: String? = null,
) {
    val isReady: Boolean get() = state == RuntimeState.READY
    val isReleased: Boolean get() = state == RuntimeState.RELEASED
}

/** Distinct execution phases measured during speech bubble detection and decoding. */
enum class Phase {
    ImagePreparation,
    RuntimeInitialization,
    Preprocessing,
    Inference,
    ComponentsScoring,
    ContoursSimplification,
    ResultPacking,
    RuntimeOverhead,
    OtherRuntimeWork,
    ResultUnpacking,
    CallerOverhead,
}

/** Non-overlapping elapsed wall-clock time for one detection phase. */
data class PhaseTiming(val phase: Phase, val durationMs: Double)

/** Result of a speech bubble detection run, containing extracted contours, dimensions, and timing metrics. */
data class DetectionResult(
    val bubbles: List<BubblePath>,
    val imageWidth: Int,
    val imageHeight: Int,
    val runtimeName: String,
    val inferenceMs: Double,
    val totalMs: Double,
    val phaseTimings: List<PhaseTiming> = emptyList(),
) {
    val bubbleCount: Int get() = bubbles.size
}

/** Raw runtime result. Bubble records use the documented compact wire format. */
data class InferencePayload(
    val packedBubbles: FloatArray,
    val inferenceMs: Double,
    val phaseTimings: List<PhaseTiming> = emptyList(),
)

/**
 * One independently owned inference runtime.
 *
 * A released runtime is permanently unavailable.
 */
interface BubbleRuntime {
    val name: String
    val status: RuntimeStatus

    fun prewarm(invokeOnce: Boolean = true): RuntimeStatus

    fun detect(
        image: Any,
        width: Int,
        height: Int,
        confidenceThreshold: Float,
    ): InferencePayload

    fun release()
}

/** Stateless decoder for platform runtime results. */
class BubbleDetector {
    companion object {
        const val DEFAULT_CONFIDENCE_THRESHOLD: Float = 0.6f
    }

    fun detect(
        image: Any,
        width: Int,
        height: Int,
        runtime: BubbleRuntime,
        confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    ): DetectionResult = decodeAndDetect(this, image, width, height, runtime, confidenceThreshold)
}

internal expect fun decodeAndDetect(
    detector: BubbleDetector,
    image: Any,
    width: Int,
    height: Int,
    runtime: BubbleRuntime,
    confidenceThreshold: Float,
): DetectionResult

internal fun TimeMark.elapsedMs(): Double = elapsedNow().inWholeNanoseconds / 1_000_000.0

internal fun measureDetection(
    started: TimeMark,
    width: Int,
    height: Int,
    runtime: BubbleRuntime,
    sourceMs: Double,
    detect: () -> InferencePayload,
): DetectionResult {
    val runtimeStarted = TimeSource.Monotonic.markNow()
    val payload = detect()
    val runtimeMs = runtimeStarted.elapsedMs()
    val unpackStarted = TimeSource.Monotonic.markNow()
    val bubbles = unpackBubbles(payload.packedBubbles)
    val unpackMs = unpackStarted.elapsedMs()
    val runtimePhases = payload.phaseTimings.ifEmpty {
        listOf(PhaseTiming(Phase.Inference, payload.inferenceMs))
    }
    val phases = buildList {
        add(PhaseTiming(Phase.ImagePreparation, sourceMs))
        addAll(runtimePhases)
        add(PhaseTiming(
            if (payload.phaseTimings.isEmpty()) Phase.OtherRuntimeWork
            else Phase.RuntimeOverhead,
            (runtimeMs - runtimePhases.sumOf { it.durationMs }).coerceAtLeast(0.0),
        ))
        add(PhaseTiming(Phase.ResultUnpacking, unpackMs))
    }
    val totalMs = started.elapsedMs()
    return DetectionResult(
        bubbles, width, height, runtime.name, payload.inferenceMs, totalMs,
        phases + PhaseTiming(Phase.CallerOverhead, (totalMs - phases.sumOf { it.durationMs }).coerceAtLeast(0.0)),
    )
}

internal fun unpackBubbles(packed: FloatArray): List<BubblePath> {
    val bubbles = mutableListOf<BubblePath>()
    var offset = 0
    while (offset < packed.size) {
        require(packed.size - offset >= RECORD_HEADER_SIZE) {
            "Inference runtime returned a truncated bubble record"
        }
        val confidence = packed[offset++]
        val maskProbability = packed[offset++]
        val areaPixels = packed[offset++].roundToInt()
        val pointCount = packed[offset++].roundToInt()
        require(pointCount >= 3 && pointCount <= (packed.size - offset) / 2) {
            "Inference runtime returned an invalid contour"
        }
        val points = ArrayList<BubblePoint>(pointCount)
        repeat(pointCount) {
            points += BubblePoint(packed[offset++], packed[offset++])
        }
        bubbles += BubblePath(
            points = points,
            confidence = confidence,
            maskProbability = maskProbability,
            areaPixels = areaPixels,
        )
    }
    return bubbles
}

private const val RECORD_HEADER_SIZE = 4
