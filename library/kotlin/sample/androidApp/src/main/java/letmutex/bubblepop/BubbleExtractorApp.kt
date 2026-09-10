package letmutex.bubblepop

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color as AndroidColor
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.TextButton
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

@Composable
fun BubbleExtractorApp() {
    val context = LocalContext.current
    val detector = remember { BubbleDetector() }
    val runtimeOptions = remember {
        listOf(
            SampleRuntimeOption(
                label = "BubblePop CPU",
                runtime = BubblePopRuntime(context.applicationContext, LiteRtBackend.CPU),
            ),
            SampleRuntimeOption(
                label = "BubblePop GPU",
                runtime = BubblePopRuntime(context.applicationContext, LiteRtBackend.GPU_FP16),
            ),
            SampleRuntimeOption(
                label = "YOLO CPU",
                runtime = YoloRuntime(context.applicationContext, LiteRtBackend.CPU),
            ),
            SampleRuntimeOption(
                label = "YOLO GPU",
                runtime = YoloRuntime(context.applicationContext, LiteRtBackend.GPU_FP16),
            ),
        )
    }
    val scope = rememberCoroutineScope()
    var selectedImage by remember { mutableStateOf<Uri?>(null) }
    var result by remember { mutableStateOf<DetectionResult?>(null) }
    var timingResult by remember { mutableStateOf<DetectionResult?>(null) }
    var selectedRuntimeOption by remember { mutableStateOf(runtimeOptions.first()) }
    var running by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    DisposableEffect(runtimeOptions) {
        onDispose { runtimeOptions.forEach { it.runtime.release() } }
    }

    fun run(image: Uri, runtime: BubbleRuntime = selectedRuntimeOption.runtime) {
        selectedImage = image
        result = null
        timingResult = null
        error = null
        running = true
        scope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    val bitmap = decodeUriToBitmap(context, image)
                    try {
                        detector.detect(bitmap, bitmap.width, bitmap.height, runtime)
                    } finally {
                        bitmap.recycle()
                    }
                }
            }.onSuccess { result = it }
                .onFailure { error = it.message ?: it.javaClass.simpleName }
            running = false
        }
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) run(uri)
    }

    MaterialTheme {
        timingResult?.let { TimingDialog(it) { timingResult = null } }
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text("Bubble Extractor", style = MaterialTheme.typography.headlineMedium)
                Text("Model runtime", style = MaterialTheme.typography.labelLarge)
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    runtimeOptions.forEach { option ->
                        FilterChip(
                            selected = selectedRuntimeOption == option,
                            onClick = {
                                selectedRuntimeOption = option
                                selectedImage?.let { run(it, option.runtime) }
                            },
                            enabled = !running,
                            label = { Text(option.label) },
                        )
                    }
                }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(16.dp),
                        ),
                    contentAlignment = Alignment.Center,
                ) {
                    if (selectedImage == null) {
                        Text("Choose a manga or comic page")
                    } else {
                        AsyncImage(
                            model = selectedImage,
                            contentDescription = "Selected page",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit,
                        )
                        result?.let { BubbleOverlay(it, Modifier.fillMaxSize()) }
                    }
                    if (running) CircularProgressIndicator()
                }

                val ret = result
                if (ret != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text("${ret.bubbleCount} bubbles", fontSize = 14.sp)
                        Text(
                            text = "${ret.inferenceMs.ms()} / ${ret.totalMs.ms()} total",
                            modifier = Modifier.clickable(onClickLabel = "Show phase timings") {
                                timingResult = ret
                            }.padding(vertical = 8.dp),
                            color = MaterialTheme.colorScheme.primary,
                            fontSize = 13.sp,
                         )
                    }
                } else {
                    Text("Pending...", fontSize = 14.sp)
                }
                error?.let {
                    Text(it, color = MaterialTheme.colorScheme.error)
                }
                Spacer(Modifier.height(2.dp))
                Button(
                    onClick = { picker.launch("image/*") },
                    enabled = !running,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (selectedImage == null) "Select image" else "Select another image")
                }
            }
        }
    }
}

private fun decodeUriToBitmap(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    val decoded = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
    }
    return if (decoded.config == Bitmap.Config.ARGB_8888) {
        decoded
    } else {
        decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
    }
}

private data class SampleRuntimeOption(
    val label: String,
    val runtime: BubbleRuntime,
)

private fun Double.ms(): String = String.format(Locale.US, "%.1f ms", this)

@Composable
private fun TimingDialog(result: DetectionResult, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Detection timings") },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(result.runtimeName, style = MaterialTheme.typography.labelLarge)
                result.phaseTimings.ifEmpty {
                    listOf(PhaseTiming(Phase.Inference, result.inferenceMs))
                }.forEach { phase ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Text(phase.phase.name, modifier = Modifier.weight(1f))
                        Text(phase.durationMs.ms())
                    }
                }
                HorizontalDivider()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("Total", style = MaterialTheme.typography.titleSmall)
                    Text(result.totalMs.ms(), style = MaterialTheme.typography.titleSmall)
                }
                Text(
                    "Elapsed time for this detection. Initialization is included when needed. " +
                        "Image display and drawing are excluded. Values are rounded.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
    )
}

@Composable
private fun BubbleOverlay(result: DetectionResult, modifier: Modifier = Modifier) {
    val colors = listOf(
        Color(0xFF00BCD4),
        Color(0xFFFF5722),
        Color(0xFF7C4DFF),
        Color(0xFF43A047),
        Color(0xFFFFB300),
        Color(0xFF03A9F4),
    )
    Canvas(modifier) {
        val imageScale = minOf(size.width / result.imageWidth, size.height / result.imageHeight)
        val drawnWidth = result.imageWidth * imageScale
        val drawnHeight = result.imageHeight * imageScale
        val origin = Offset((size.width - drawnWidth) / 2f, (size.height - drawnHeight) / 2f)

        result.bubbles.forEachIndexed { index, bubble ->
            if (bubble.points.size < 3) return@forEachIndexed
            val color = colors[index % colors.size]
            val path = Path().apply {
                val first = bubble.points.first()
                moveTo(origin.x + first.x * drawnWidth, origin.y + first.y * drawnHeight)
                bubble.points.drop(1).forEach { point ->
                    lineTo(origin.x + point.x * drawnWidth, origin.y + point.y * drawnHeight)
                }
                close()
            }
            drawPath(path, color.copy(alpha = 0.28f))
            drawPath(path, color, style = Stroke(width = 2.dp.toPx()))

            val anchor = bubble.points.minBy { it.y }
            val labelX = origin.x + anchor.x * drawnWidth
            val labelY = origin.y + anchor.y * drawnHeight - 4.dp.toPx()
            val text = String.format(Locale.US, "%.2f", bubble.confidence)
            drawContext.canvas.nativeCanvas.apply {
                drawText(text, labelX, labelY, Paint().apply {
                    textSize = 12.dp.toPx()
                    isAntiAlias = true
                    style = Paint.Style.STROKE
                    strokeWidth = 3.dp.toPx()
                    this.color = AndroidColor.WHITE
                })
                drawText(text, labelX, labelY, Paint().apply {
                    textSize = 12.dp.toPx()
                    isAntiAlias = true
                    this.color = AndroidColor.BLACK
                })
            }
        }
    }
}
