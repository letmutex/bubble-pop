package letmutex.bubblepop

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BubbleDetectorTest {
    @Test
    fun preservesRuntimePhasesAndAccountsForTotal() {
        val phases = listOf(PhaseTiming(Phase.Preprocessing, 0.0), PhaseTiming(Phase.Inference, 0.0))
        val result = BubbleDetector().detect(
            IntArray(1), 1, 1, FakeRuntime(floatArrayOf(), 0.0, phases),
        )
        assertEquals(phases, result.phaseTimings.filter { it.phase in phases.map { phase -> phase.phase } })
        assertTrue(result.phaseTimings.any { it.phase == Phase.ResultUnpacking })
        assertTrue(result.phaseTimings.all { it.durationMs.isFinite() && it.durationMs >= 0.0 })
        assertEquals(result.totalMs, result.phaseTimings.sumOf { it.durationMs }, 0.000001)
    }

    @Test
    fun suppliesTimingBreakdownForRuntimesWithoutDetailedPhases() {
        val result = BubbleDetector().detect(IntArray(1), 1, 1, FakeRuntime(floatArrayOf(), 0.0))
        assertEquals(0.0, result.phaseTimings.single { it.phase == Phase.Inference }.durationMs)
        assertTrue(result.phaseTimings.any { it.phase == Phase.OtherRuntimeWork })
        assertEquals(result.totalMs, result.phaseTimings.sumOf { it.durationMs }, 0.000001)
    }

    @Test
    fun decodesPackedContours() {
        val runtime = FakeRuntime(
            floatArrayOf(
                0.9f, 0.8f, 42f, 3f,
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            ),
        )

        val result = BubbleDetector().detect(
            IntArray(6),
            width = 3,
            height = 2,
            runtime = runtime,
        )

        assertEquals(1, result.bubbleCount)
        assertEquals(BubblePoint(0.3f, 0.4f), result.bubbles.single().points[1])
        assertEquals(42, result.bubbles.single().areaPixels)
        assertEquals("Fake runtime", result.runtimeName)
    }

    @Test
    fun rejectsMalformedPayloads() {
        val runtime = FakeRuntime(floatArrayOf(0.9f, 0.8f, 42f, 4f, 0.1f, 0.2f))
        assertFailsWith<IllegalArgumentException> {
            BubbleDetector().detect(IntArray(1), 1, 1, runtime)
        }
    }

    @Test
    fun validatesPixelDimensionsBeforeCallingRuntime() {
        val runtime = FakeRuntime(floatArrayOf())
        assertFailsWith<IllegalArgumentException> {
            BubbleDetector().detect(IntArray(3), 2, 2, runtime)
        }
        assertEquals(0, runtime.calls)
    }

    @Test
    fun forwardsConfidenceThreshold() {
        val runtime = FakeRuntime(packedBubble(confidence = 0.5f))
        val detector = BubbleDetector()

        assertEquals(0.6f, BubbleDetector.DEFAULT_CONFIDENCE_THRESHOLD)

        detector.detect(IntArray(1), 1, 1, runtime, confidenceThreshold = 0.5f)
        assertEquals(0.5f, runtime.lastConfidenceThreshold)

        detector.detect(IntArray(1), 1, 1, runtime)
        assertEquals(BubbleDetector.DEFAULT_CONFIDENCE_THRESHOLD, runtime.lastConfidenceThreshold)
    }

    @Test
    fun rejectsInvalidConfidenceThreshold() {
        val runtime = FakeRuntime(floatArrayOf())
        val detector = BubbleDetector()

        for (threshold in listOf(-0.1f, 1.1f, Float.NaN, Float.POSITIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException> {
                detector.detect(IntArray(1), 1, 1, runtime, threshold)
            }
        }
        assertEquals(0, runtime.calls)
    }

    @Test
    fun releasedRuntimeIsPermanentlyUnavailable() {
        val runtime = FakeRuntime(floatArrayOf())

        assertEquals(RuntimeState.NOT_INITIALIZED, runtime.status.state)
        assertEquals(RuntimeState.READY, runtime.prewarm(invokeOnce = false).state)
        assertEquals(false, runtime.lastInvokeOnce)

        runtime.release()
        assertEquals(RuntimeState.RELEASED, runtime.status.state)
        assertFailsWith<IllegalStateException> { runtime.prewarm() }
        assertFailsWith<IllegalStateException> {
            BubbleDetector().detect(IntArray(1), 1, 1, runtime)
        }

        val replacement = FakeRuntime(floatArrayOf())
        assertEquals(RuntimeState.READY, replacement.prewarm().state)
    }

    @Test
    fun acceptsAnyIntArrayWithDimensions() {
        val runtime = FakeRuntime(floatArrayOf())
        val detector = BubbleDetector()
        val image: Any = IntArray(4)
        val result = detector.detect(image, 2, 2, runtime)
        assertEquals(2, result.imageWidth)
        assertEquals(2, result.imageHeight)
        assertEquals(1, runtime.calls)
    }

    @Test
    fun rejectsUnsupportedAnyType() {
        val runtime = FakeRuntime(floatArrayOf())
        val detector = BubbleDetector()
        assertFailsWith<IllegalArgumentException> {
            detector.detect("unsupported string", 1, 1, runtime)
        }
    }

    private fun packedBubble(confidence: Float) = floatArrayOf(
        confidence, 0.8f, 42f, 3f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
    )

    private class FakeRuntime(
        private val packed: FloatArray,
        private val inferenceMs: Double = 12.5,
        private val phases: List<PhaseTiming> = emptyList(),
    ) : BubbleRuntime {
        override val name = "Fake runtime"
        override var status = RuntimeStatus(RuntimeState.NOT_INITIALIZED)
            private set

        var calls = 0
        var lastConfidenceThreshold = Float.NaN
        var lastInvokeOnce = false

        override fun detect(
            image: Any,
            width: Int,
            height: Int,
            confidenceThreshold: Float,
        ): InferencePayload {
            checkAvailable()
            calls++
            lastConfidenceThreshold = confidenceThreshold
            status = RuntimeStatus(RuntimeState.READY)
            return InferencePayload(packed, inferenceMs, phases)
        }

        override fun prewarm(invokeOnce: Boolean): RuntimeStatus {
            checkAvailable()
            lastInvokeOnce = invokeOnce
            return RuntimeStatus(RuntimeState.READY, initializationMs = 4.2).also {
                status = it
            }
        }

        override fun release() {
            status = RuntimeStatus(RuntimeState.RELEASED)
        }

        private fun checkAvailable() {
            check(!status.isReleased) { "$name has been released" }
        }
    }
}
