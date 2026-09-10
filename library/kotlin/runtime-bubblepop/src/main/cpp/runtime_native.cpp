#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tflite/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kInputWidth = 768;
constexpr int kInputHeight = 1024;
constexpr int kInputChannels = 1;
constexpr int kOutputChannels = 3;
constexpr int kCpuThreads = 4;
constexpr float kMaskLogitThreshold = 0.0f;  // sigmoid(logit) >= 0.50
constexpr int kMinAreaPixels = 60;
constexpr int kMaxPathPoints = 512;
constexpr float kMean = 0.449f;
constexpr float kStd = 0.226f;
constexpr float kRdpEpsilonPx = 1.0f;
constexpr int kDx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int kDy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

enum class Backend : int {
  kInterpreterCpu = 0,
  kInterpreterGpu = 1,
};

enum class PixelFormat {
  kArgb8888,
  kRgba8888,
};

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

void CheckTfLite(TfLiteStatus status, const char* operation) {
  if (status == kTfLiteOk) return;
  throw std::runtime_error(std::string(operation) + " failed");
}

void TfLiteErrorReporter(void*, const char* format, va_list arguments) {
  __android_log_vprint(ANDROID_LOG_ERROR, "BubbleTfLite", format, arguments);
}

void ThrowIllegalState(JNIEnv* env, const std::string& message) {
  jclass type = env->FindClass("java/lang/IllegalStateException");
  if (type != nullptr) env->ThrowNew(type, message.c_str());
}

std::string FromJavaString(JNIEnv* env, jstring value) {
  if (value == nullptr) return {};
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) throw std::runtime_error("Unable to read Java string");
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

std::vector<uint8_t> ReadAsset(AAssetManager* assets, const char* filename) {
  AAsset* asset = AAssetManager_open(assets, filename, AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    throw std::runtime_error(std::string("Unable to open model asset: ") + filename);
  }
  const auto length = static_cast<size_t>(AAsset_getLength64(asset));
  std::vector<uint8_t> bytes(length);
  const int64_t read = AAsset_read(asset, bytes.data(), length);
  AAsset_close(asset);
  if (read < 0 || static_cast<size_t>(read) != length) {
    throw std::runtime_error(std::string("Unable to read model asset: ") + filename);
  }
  return bytes;
}

struct InterpreterState {
  std::vector<uint8_t> model_bytes;
  TfLiteModel* model = nullptr;
  TfLiteInterpreterOptions* options = nullptr;
  TfLiteDelegate* cpu_delegate = nullptr;
  TfLiteDelegate* gpu_delegate = nullptr;
  TfLiteInterpreter* interpreter = nullptr;
  TfLiteTensor* input_tensor = nullptr;
  const TfLiteTensor* output_tensor = nullptr;
  int output_height = 0;
  int output_width = 0;
  int output_channels = 0;

  ~InterpreterState() {
    if (interpreter != nullptr) TfLiteInterpreterDelete(interpreter);
    if (options != nullptr) TfLiteInterpreterOptionsDelete(options);
    if (gpu_delegate != nullptr) TfLiteGpuDelegateV2Delete(gpu_delegate);
    if (cpu_delegate != nullptr) TfLiteXNNPackDelegateDelete(cpu_delegate);
    if (model != nullptr) TfLiteModelDelete(model);
  }
};

struct Runtime {
  AAssetManager* assets;
  std::string cache_dir;
  std::string asset_path;
  std::string serialization_token;
  Backend backend;
  std::unique_ptr<InterpreterState> interpreter;
  std::mutex mutex;
  std::vector<uint32_t> visited_epoch;
  uint32_t current_epoch = 0;
  std::vector<uint8_t> component;
  std::vector<int> queue;

  Runtime(AAssetManager* asset_manager, std::string cache, std::string asset,
          std::string token, Backend selected_backend)
      : assets(asset_manager),
        cache_dir(std::move(cache)),
        asset_path(std::move(asset)),
        serialization_token(std::move(token)),
        backend(selected_backend) {}

  InterpreterState& GetInterpreter() {
    if (interpreter != nullptr) return *interpreter;

    auto created = std::make_unique<InterpreterState>();
    const bool use_gpu = backend == Backend::kInterpreterGpu;
    created->model_bytes = ReadAsset(assets, asset_path.c_str());
    created->model = TfLiteModelCreate(created->model_bytes.data(),
                                       created->model_bytes.size());
    if (created->model == nullptr) {
      throw std::runtime_error("TfLiteModelCreate failed");
    }

    created->options = TfLiteInterpreterOptionsCreate();
    if (created->options == nullptr) {
      throw std::runtime_error("TfLiteInterpreterOptionsCreate failed");
    }
    TfLiteInterpreterOptionsSetNumThreads(created->options, kCpuThreads);
    TfLiteInterpreterOptionsSetErrorReporter(created->options,
                                             TfLiteErrorReporter, nullptr);
    if (use_gpu) {
      TfLiteGpuDelegateOptionsV2 gpu_options =
          TfLiteGpuDelegateOptionsV2Default();
      gpu_options.is_precision_loss_allowed = 1;
      gpu_options.inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
      if (!cache_dir.empty()) {
        gpu_options.experimental_flags |=
            TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
        gpu_options.serialization_dir = cache_dir.c_str();
        gpu_options.model_token = serialization_token.c_str();
      }
      created->gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_options);
      if (created->gpu_delegate == nullptr) {
        throw std::runtime_error("TfLiteGpuDelegateV2Create failed");
      }
      TfLiteInterpreterOptionsAddDelegate(created->options,
                                          created->gpu_delegate);
    } else {
      TfLiteXNNPackDelegateOptions xnnpack_options =
          TfLiteXNNPackDelegateOptionsDefault();
      xnnpack_options.num_threads = kCpuThreads;
      created->cpu_delegate =
          TfLiteXNNPackDelegateCreate(&xnnpack_options);
      if (created->cpu_delegate == nullptr) {
        throw std::runtime_error("TfLiteXNNPackDelegateCreate failed");
      }
      TfLiteInterpreterOptionsAddDelegate(created->options,
                                          created->cpu_delegate);
    }

    created->interpreter =
        TfLiteInterpreterCreate(created->model, created->options);
    if (created->interpreter == nullptr) {
      throw std::runtime_error("TfLiteInterpreterCreate failed");
    }
    CheckTfLite(TfLiteInterpreterAllocateTensors(created->interpreter),
                "TfLiteInterpreterAllocateTensors");
    if (TfLiteInterpreterGetInputTensorCount(created->interpreter) != 1 ||
        TfLiteInterpreterGetOutputTensorCount(created->interpreter) != 1) {
      throw std::runtime_error("Expected one model input and one output");
    }
    created->input_tensor =
        TfLiteInterpreterGetInputTensor(created->interpreter, 0);
    created->output_tensor =
        TfLiteInterpreterGetOutputTensor(created->interpreter, 0);
    if (created->input_tensor == nullptr || created->output_tensor == nullptr ||
        TfLiteTensorType(created->input_tensor) != kTfLiteFloat32 ||
        TfLiteTensorNumDims(created->input_tensor) != 4 ||
        TfLiteTensorDim(created->input_tensor, 0) != 1 ||
        TfLiteTensorDim(created->input_tensor, 1) != kInputHeight ||
        TfLiteTensorDim(created->input_tensor, 2) != kInputWidth ||
        TfLiteTensorDim(created->input_tensor, 3) != kInputChannels) {
      throw std::runtime_error(
          "Unexpected TFLite input tensor; expected [1,1024,768,1] float32");
    }
    if (TfLiteTensorType(created->output_tensor) != kTfLiteFloat32 ||
        TfLiteTensorNumDims(created->output_tensor) != 4 ||
        TfLiteTensorDim(created->output_tensor, 0) != 1 ||
        TfLiteTensorDim(created->output_tensor, 3) < kOutputChannels) {
      throw std::runtime_error("Unexpected TFLite output tensor");
    }
    created->output_height = TfLiteTensorDim(created->output_tensor, 1);
    created->output_width = TfLiteTensorDim(created->output_tensor, 2);
    created->output_channels = TfLiteTensorDim(created->output_tensor, 3);
    __android_log_print(
        ANDROID_LOG_INFO, "BubbleNative",
        "TFLite %s backend=%s input=%zu bytes output=%zu bytes",
        TfLiteVersion(), use_gpu ? "gpu-fp16" : "cpu-fp16",
        TfLiteTensorByteSize(created->input_tensor),
        TfLiteTensorByteSize(created->output_tensor));

    interpreter = std::move(created);
    return *interpreter;
  }
};

struct Letterbox {
  float scale;
  int resized_width;
  int resized_height;
  int pad_x;
  int pad_y;
};

struct XSample {
  int x0;
  int x1;
  float wx;
};

Letterbox CalculateLetterbox(int source_width, int source_height) {
  const float scale = std::min(
      static_cast<float>(kInputWidth) / source_width,
      static_cast<float>(kInputHeight) / source_height);
  const int resized_width = std::max(1, static_cast<int>(std::lround(source_width * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::lround(source_height * scale)));
  return {scale, resized_width, resized_height,
          (kInputWidth - resized_width) / 2,
          (kInputHeight - resized_height) / 2};
}

template <PixelFormat Format>
inline float Grayscale(uint32_t color) {
  if constexpr (Format == PixelFormat::kArgb8888) {
    const float r = static_cast<float>((color >> 16) & 0xff);
    const float g = static_cast<float>((color >> 8) & 0xff);
    const float b = static_cast<float>(color & 0xff);
    return (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
  } else {
    // ANDROID_BITMAP_FORMAT_RGBA_8888 memory byte order: [R, G, B, A].
    // On little-endian architectures, Byte 0 (R) is bits 0..7 and Byte 2 (B) is bits 16..23.
    const float r = static_cast<float>(color & 0xff);
    const float g = static_cast<float>((color >> 8) & 0xff);
    const float b = static_cast<float>((color >> 16) & 0xff);
    return (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
  }
}

template <PixelFormat Format>
void Preprocess(const uint32_t* pixels, int source_width, int source_height,
                int stride_pixels, const Letterbox& box, float* destination) {
  const float white = (1.0f - kMean) / kStd;

  // 1. Precalculate X-mapping table once per image
  std::vector<XSample> x_table(box.resized_width);
  for (int local_x = 0; local_x < box.resized_width; ++local_x) {
    const float sx = std::clamp(
        (local_x + 0.5f) * source_width / box.resized_width - 0.5f,
        0.0f, static_cast<float>(source_width - 1));
    const int x0 = static_cast<int>(sx);
    const int x1 = std::min(x0 + 1, source_width - 1);
    const float wx = sx - x0;
    x_table[local_x] = {x0, x1, wx};
  }

  // 2. Fill top padding rows
  for (int y = 0; y < box.pad_y; ++y) {
    std::fill_n(destination + y * kInputWidth, kInputWidth, white);
  }

  // 3. Bilinear interpolation for image rows
  for (int local_y = 0; local_y < box.resized_height; ++local_y) {
    const int y = box.pad_y + local_y;
    float* dst_row = destination + y * kInputWidth;

    // Left padding
    if (box.pad_x > 0) {
      std::fill_n(dst_row, box.pad_x, white);
    }

    const float sy = std::clamp(
        (local_y + 0.5f) * source_height / box.resized_height - 0.5f,
        0.0f, static_cast<float>(source_height - 1));
    const int y0 = static_cast<int>(sy);
    const int y1 = std::min(y0 + 1, source_height - 1);
    const float wy = sy - y0;

    const uint32_t* row0 = pixels + y0 * stride_pixels;
    const uint32_t* row1 = pixels + y1 * stride_pixels;

    for (int local_x = 0; local_x < box.resized_width; ++local_x) {
      const auto& xs = x_table[local_x];
      const float g00 = Grayscale<Format>(row0[xs.x0]);
      const float g10 = Grayscale<Format>(row0[xs.x1]);
      const float g01 = Grayscale<Format>(row1[xs.x0]);
      const float g11 = Grayscale<Format>(row1[xs.x1]);

      const float top = g00 + (g10 - g00) * xs.wx;
      const float bottom = g01 + (g11 - g01) * xs.wx;
      const float val = top + (bottom - top) * wy;
      dst_row[box.pad_x + local_x] = (val - kMean) / kStd;
    }

    // Right padding
    const int right_start = box.pad_x + box.resized_width;
    if (right_start < kInputWidth) {
      std::fill_n(dst_row + right_start, kInputWidth - right_start, white);
    }
  }

  // 4. Fill bottom padding rows
  const int pad_bottom_start = box.pad_y + box.resized_height;
  for (int y = pad_bottom_start; y < kInputHeight; ++y) {
    std::fill_n(destination + y * kInputWidth, kInputWidth, white);
  }
}

inline float Sigmoid(float value) {
  value = std::clamp(value, -20.0f, 20.0f);
  return 1.0f / (1.0f + std::exp(-value));
}

int DirectionIndex(int dx, int dy) {
  for (int i = 0; i < 8; ++i) {
    if (kDx[i] == dx && kDy[i] == dy) return i;
  }
  return 6;
}

struct IntPoint {
  int x;
  int y;
};

std::vector<IntPoint> TraceContour(const std::vector<uint8_t>& component,
                                   const std::vector<int>& queue, int count,
                                   int width, int height) {
  int start = queue[0];
  for (int i = 1; i < count; ++i) {
    const int candidate = queue[i];
    if (candidate / width < start / width ||
        (candidate / width == start / width && candidate % width < start % width)) {
      start = candidate;
    }
  }
  int x = start % width;
  int y = start / width;
  const int start_x = x;
  const int start_y = y;
  int back_x = x - 1;
  int back_y = y;
  const int start_back_x = back_x;
  const int start_back_y = back_y;
  std::vector<IntPoint> contour;
  contour.reserve(std::min(count * 2, kMaxPathPoints * 4));
  const int max_steps = std::max(8, count * 8);
  for (int iteration = 0; iteration < max_steps; ++iteration) {
    contour.push_back({x, y});
    const int back_direction = DirectionIndex(back_x - x, back_y - y);
    int found_direction = -1;
    for (int step = 1; step <= 8; ++step) {
      const int direction = (back_direction + step) & 7;
      const int nx = x + kDx[direction];
      const int ny = y + kDy[direction];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
          component[ny * width + nx]) {
        found_direction = direction;
        break;
      }
    }
    if (found_direction < 0) break;
    const int preceding = (found_direction + 7) & 7;
    back_x = x + kDx[preceding];
    back_y = y + kDy[preceding];
    x += kDx[found_direction];
    y += kDy[found_direction];
    if (x == start_x && y == start_y && back_x == start_back_x &&
        back_y == start_back_y) {
      break;
    }
  }
  return contour;
}

float PerpendicularDistanceSq(const IntPoint& p, const IntPoint& a, const IntPoint& b) {
  const float dx = static_cast<float>(b.x - a.x);
  const float dy = static_cast<float>(b.y - a.y);
  const float d2 = dx * dx + dy * dy;
  if (d2 < 1e-6f) {
    const float px = static_cast<float>(p.x - a.x);
    const float py = static_cast<float>(p.y - a.y);
    return px * px + py * py;
  }
  const float cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
  return (cross * cross) / d2;
}

void SimplifyRDPRecursive(const std::vector<IntPoint>& points, int start, int end,
                          float epsilon_sq, std::vector<bool>& keep) {
  if (end <= start + 1) return;
  float max_dist_sq = 0.0f;
  int max_idx = start;
  for (int i = start + 1; i < end; ++i) {
    const float dist_sq = PerpendicularDistanceSq(points[i], points[start], points[end]);
    if (dist_sq > max_dist_sq) {
      max_dist_sq = dist_sq;
      max_idx = i;
    }
  }
  if (max_dist_sq > epsilon_sq) {
    keep[max_idx] = true;
    SimplifyRDPRecursive(points, start, max_idx, epsilon_sq, keep);
    SimplifyRDPRecursive(points, max_idx, end, epsilon_sq, keep);
  }
}

std::vector<IntPoint> SimplifyPolygonRDP(const std::vector<IntPoint>& contour, float epsilon_px) {
  if (contour.size() <= 4) return contour;
  const float epsilon_sq = epsilon_px * epsilon_px;
  std::vector<bool> keep(contour.size(), false);
  keep.front() = true;
  keep.back() = true;

  float max_d2 = 0.0f;
  size_t split_idx = contour.size() / 2;
  for (size_t i = 1; i < contour.size(); ++i) {
    const float dx = static_cast<float>(contour[i].x - contour[0].x);
    const float dy = static_cast<float>(contour[i].y - contour[0].y);
    const float d2 = dx * dx + dy * dy;
    if (d2 > max_d2) {
      max_d2 = d2;
      split_idx = i;
    }
  }
  keep[split_idx] = true;

  SimplifyRDPRecursive(contour, 0, static_cast<int>(split_idx), epsilon_sq, keep);
  SimplifyRDPRecursive(contour, static_cast<int>(split_idx),
                       static_cast<int>(contour.size() - 1), epsilon_sq, keep);

  std::vector<IntPoint> result;
  result.reserve(contour.size());
  for (size_t i = 0; i < contour.size(); ++i) {
    if (keep[i]) result.push_back(contour[i]);
  }
  if (result.size() < 3) return contour;

  if (result.size() > kMaxPathPoints) {
    const int step = static_cast<int>(
        std::ceil(static_cast<double>(result.size()) / kMaxPathPoints));
    std::vector<IntPoint> capped;
    capped.reserve(kMaxPathPoints);
    for (size_t i = 0; i < result.size(); i += step) {
      capped.push_back(result[i]);
    }
    return capped;
  }
  return result;
}

struct Bubble {
  float confidence;
  float mask_probability;
  int area_pixels;
  std::vector<float> points;
};

std::vector<Bubble> ExtractBubbles(Runtime& runtime, const float* mask_logits,
                                   const float* confidence_logits,
                                   int pixel_stride, int width, int height,
                                   const Letterbox& box,
                                   float confidence_threshold,
                                   double& contour_ms) {
  contour_ms = 0.0;
  const int plane = width * height;
  if (runtime.visited_epoch.size() != static_cast<size_t>(plane)) {
    runtime.visited_epoch.assign(plane, 0);
    runtime.current_epoch = 0;
  }
  if (runtime.component.size() != static_cast<size_t>(plane)) {
    runtime.component.assign(plane, 0);
  }
  if (runtime.queue.size() != static_cast<size_t>(plane)) {
    runtime.queue.resize(plane);
  }

  // Increment epoch; if wrapped around, reset visited buffer once
  if (++runtime.current_epoch == 0) {
    runtime.visited_epoch.assign(plane, 0);
    runtime.current_epoch = 1;
  }
  const uint32_t epoch = runtime.current_epoch;
  std::vector<Bubble> bubbles;

  const float output_scale_x = static_cast<float>(width) / kInputWidth;
  const float output_scale_y = static_cast<float>(height) / kInputHeight;
  const float crop_x = box.pad_x * output_scale_x;
  const float crop_y = box.pad_y * output_scale_y;
  const float crop_width = box.resized_width * output_scale_x;
  const float crop_height = box.resized_height * output_scale_y;
  const int min_model_area = std::max(
      1, static_cast<int>(std::ceil(kMinAreaPixels * box.scale * box.scale *
                                    output_scale_x * output_scale_y)));

  for (int start = 0; start < plane; ++start) {
    if (runtime.visited_epoch[start] == epoch ||
        mask_logits[start * pixel_stride] < kMaskLogitThreshold) {
      continue;
    }
    int head = 0;
    int tail = 0;
    double mask_sum = 0.0;
    double confidence_sum = 0.0;
    runtime.queue[tail++] = start;
    runtime.visited_epoch[start] = epoch;
    while (head < tail) {
      const int index = runtime.queue[head++];
      runtime.component[index] = 1;
      mask_sum += Sigmoid(mask_logits[index * pixel_stride]);
      confidence_sum += Sigmoid(confidence_logits[index * pixel_stride]);
      const int x = index % width;
      const int y = index / width;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int nx = x + dx;
          const int ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const int next = ny * width + nx;
          if (runtime.visited_epoch[next] != epoch &&
              mask_logits[next * pixel_stride] >= kMaskLogitThreshold) {
            runtime.visited_epoch[next] = epoch;
            runtime.queue[tail++] = next;
          }
        }
      }
    }

    const float mean_mask = static_cast<float>(mask_sum / tail);
    const float score = static_cast<float>((mask_sum + confidence_sum) /
                                           (2.0 * tail));
    if (tail >= min_model_area && score >= confidence_threshold) {
      const auto contour_started = Clock::now();
      const std::vector<IntPoint> raw_contour = TraceContour(
          runtime.component, runtime.queue, tail, width, height);
      const std::vector<IntPoint> simplified = SimplifyPolygonRDP(
          raw_contour, kRdpEpsilonPx);

      Bubble bubble{score, mean_mask,
                    static_cast<int>(std::lround(tail /
                                                 (box.scale * box.scale))),
                    {}};
      bubble.points.reserve(simplified.size() * 2);
      for (const auto& pt : simplified) {
        const float nx = std::clamp(
            (pt.x + 0.5f - crop_x) / crop_width, 0.0f, 1.0f);
        const float ny = std::clamp(
            (pt.y + 0.5f - crop_y) / crop_height, 0.0f, 1.0f);
        bubble.points.push_back(nx);
        bubble.points.push_back(ny);
      }
      if (bubble.points.size() >= 6) bubbles.push_back(std::move(bubble));
      contour_ms += Milliseconds(Clock::now() - contour_started);
    }
    for (int i = 0; i < tail; ++i) runtime.component[runtime.queue[i]] = 0;
  }
  std::sort(bubbles.begin(), bubbles.end(),
            [](const Bubble& left, const Bubble& right) {
              return left.confidence > right.confidence;
            });
  return bubbles;
}

std::vector<float> PackBubbles(const std::vector<Bubble>& bubbles) {
  size_t size = 0;
  for (const Bubble& bubble : bubbles) size += 4 + bubble.points.size();
  std::vector<float> packed;
  packed.reserve(size);
  for (const Bubble& bubble : bubbles) {
    packed.push_back(bubble.confidence);
    packed.push_back(bubble.mask_probability);
    packed.push_back(static_cast<float>(bubble.area_pixels));
    packed.push_back(static_cast<float>(bubble.points.size() / 2));
    packed.insert(packed.end(), bubble.points.begin(), bubble.points.end());
  }
  return packed;
}

jobject MakePayload(JNIEnv* env, const std::vector<float>& packed,
                    double inference_ms, double native_total_ms,
                    const double (&phase_ms)[6]) {
  jfloatArray java_packed = env->NewFloatArray(static_cast<jsize>(packed.size()));
  if (java_packed == nullptr) return nullptr;
  if (!packed.empty()) {
    env->SetFloatArrayRegion(java_packed, 0, static_cast<jsize>(packed.size()),
                             packed.data());
    if (env->ExceptionCheck()) return nullptr;
  }
  jclass payload_class =
      env->FindClass(
          "letmutex/bubblepop/NativeBubblePopDetectionPayload");
  if (payload_class == nullptr) return nullptr;
  jdoubleArray java_phases = env->NewDoubleArray(6);
  if (java_phases == nullptr) return nullptr;
  env->SetDoubleArrayRegion(java_phases, 0, 6, phase_ms);
  if (env->ExceptionCheck()) return nullptr;
  jmethodID constructor = env->GetMethodID(payload_class, "<init>", "([FDD[D)V");
  if (constructor == nullptr) return nullptr;
  return env->NewObject(payload_class, constructor, java_packed, inference_ms,
                        native_total_ms, java_phases);
}

template <PixelFormat Format, typename ReleaseCallback>
jobject ExecuteDetectionPipeline(JNIEnv* env, Runtime& runtime,
                                 const uint32_t* pixels, int source_width,
                                 int source_height, int stride_pixels,
                                 float confidence_threshold,
                                 ReleaseCallback&& release_source_pixels) {
  std::lock_guard<std::mutex> lock(runtime.mutex);
  const auto native_started = Clock::now();
  const Letterbox letterbox = CalculateLetterbox(source_width, source_height);
  InterpreterState& state = runtime.GetInterpreter();
  const auto initialization_finished = Clock::now();

  auto* input_address =
      static_cast<float*>(TfLiteTensorData(state.input_tensor));
  if (input_address == nullptr) {
    throw std::runtime_error("TFLite input tensor is not CPU-accessible");
  }

  Preprocess<Format>(pixels, source_width, source_height, stride_pixels,
                     letterbox, input_address);

  // Release Java arrays or unlock Android Bitmap as soon as preprocessing completes
  release_source_pixels();

  const auto inference_started = Clock::now();
  CheckTfLite(TfLiteInterpreterInvoke(state.interpreter),
              "TfLiteInterpreterInvoke");
  const double inference_ms = Milliseconds(Clock::now() - inference_started);
  const auto postprocess_started = Clock::now();

  const float* logits =
      static_cast<const float*>(TfLiteTensorData(state.output_tensor));
  if (logits == nullptr) {
    throw std::runtime_error("TFLite output tensor is not CPU-accessible");
  }

  const float* mask_logits = logits;
  const float* confidence_logits = logits + 2;
  const int pixel_stride = state.output_channels;
  const int output_width = state.output_width;
  const int output_height = state.output_height;

  double contour_ms = 0.0;
  const std::vector<Bubble> bubbles = ExtractBubbles(
      runtime, mask_logits, confidence_logits, pixel_stride,
      output_width, output_height, letterbox, confidence_threshold, contour_ms);
  const auto packing_started = Clock::now();
  const std::vector<float> packed = PackBubbles(bubbles);
  const auto packing_finished = Clock::now();
  // Non-overlapping phases. JNI acquisition/payload creation and executor waiting
  // are accounted for by the caller's runtime overhead measurement.
  const double phase_ms[6] = {
      Milliseconds(initialization_finished - native_started),
      Milliseconds(inference_started - initialization_finished),
      inference_ms,
      std::max(0.0, Milliseconds(packing_started - postprocess_started) - contour_ms),
      contour_ms,
      Milliseconds(packing_finished - packing_started),
  };
  return MakePayload(env, packed, inference_ms,
                     Milliseconds(packing_finished - native_started), phase_ms);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_letmutex_bubblepop_NativeBubblePopRuntime_nativeCreate(
    JNIEnv* env, jobject, jobject java_asset_manager, jstring java_cache_dir,
    jint backend_value, jstring java_asset_path, jstring java_model_token) {
  if (backend_value < static_cast<int>(Backend::kInterpreterCpu) ||
      backend_value > static_cast<int>(Backend::kInterpreterGpu)) {
    ThrowIllegalState(env, "Unknown inference backend");
    return 0;
  }
  try {
    AAssetManager* assets = AAssetManager_fromJava(env, java_asset_manager);
    if (assets == nullptr) throw std::runtime_error("Unable to access Android assets");
    std::string asset_path = FromJavaString(env, java_asset_path);
    std::string model_token = FromJavaString(env, java_model_token);
    if (asset_path.empty() || model_token.empty()) {
      throw std::runtime_error("Asset path and model token must not be empty");
    }
    auto* runtime = new Runtime(
        assets, FromJavaString(env, java_cache_dir), std::move(asset_path),
        std::move(model_token), static_cast<Backend>(backend_value));
    return reinterpret_cast<jlong>(runtime);
  } catch (const std::exception& error) {
    ThrowIllegalState(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_letmutex_bubblepop_NativeBubblePopRuntime_nativeDetect(
    JNIEnv* env, jobject, jlong handle, jintArray java_pixels,
    jint source_width, jint source_height, jfloat confidence_threshold) {
  auto* runtime = reinterpret_cast<Runtime*>(handle);
  if (runtime == nullptr) {
    ThrowIllegalState(env, "Native bubble runtime is closed");
    return nullptr;
  }
  if (source_width <= 0 || source_height <= 0 || java_pixels == nullptr ||
      env->GetArrayLength(java_pixels) != source_width * source_height) {
    ThrowIllegalState(env, "Invalid ARGB pixel buffer");
    return nullptr;
  }
  if (!std::isfinite(confidence_threshold) || confidence_threshold < 0.0f ||
      confidence_threshold > 1.0f) {
    ThrowIllegalState(env, "Confidence threshold must be between 0 and 1");
    return nullptr;
  }

  jint* pixels = env->GetIntArrayElements(java_pixels, nullptr);
  if (pixels == nullptr) {
    ThrowIllegalState(env, "Unable to access ARGB pixels");
    return nullptr;
  }

  try {
    bool released = false;
    auto release_action = [&]() {
      if (!released) {
        env->ReleaseIntArrayElements(java_pixels, pixels, JNI_ABORT);
        released = true;
      }
    };
    try {
      jobject result = ExecuteDetectionPipeline<PixelFormat::kArgb8888>(
          env, *runtime, reinterpret_cast<const uint32_t*>(pixels),
          source_width, source_height, source_width,
          confidence_threshold, release_action);
      release_action();
      return result;
    } catch (...) {
      release_action();
      throw;
    }
  } catch (const std::exception& error) {
    ThrowIllegalState(env, error.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_letmutex_bubblepop_NativeBubblePopRuntime_nativeDetectBitmap(
    JNIEnv* env, jobject, jlong handle, jobject java_bitmap,
    jfloat confidence_threshold) {
  auto* runtime = reinterpret_cast<Runtime*>(handle);
  if (runtime == nullptr) {
    ThrowIllegalState(env, "Native bubble runtime is closed");
    return nullptr;
  }
  if (java_bitmap == nullptr) {
    ThrowIllegalState(env, "Bitmap must not be null");
    return nullptr;
  }
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, java_bitmap, &info) < 0) {
    ThrowIllegalState(env, "Failed to get Bitmap info");
    return nullptr;
  }
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    ThrowIllegalState(env, "Bitmap must be ARGB_8888 format");
    return nullptr;
  }
  if (!std::isfinite(confidence_threshold) || confidence_threshold < 0.0f ||
      confidence_threshold > 1.0f) {
    ThrowIllegalState(env, "Confidence threshold must be between 0 and 1");
    return nullptr;
  }

  void* pixels_ptr = nullptr;
  if (AndroidBitmap_lockPixels(env, java_bitmap, &pixels_ptr) < 0 || pixels_ptr == nullptr) {
    ThrowIllegalState(env, "Failed to lock Bitmap pixels");
    return nullptr;
  }
  const int source_width = static_cast<int>(info.width);
  const int source_height = static_cast<int>(info.height);
  const int stride_pixels = static_cast<int>(info.stride / 4);

  try {
    bool unlocked = false;
    auto unlock_action = [&]() {
      if (!unlocked) {
        AndroidBitmap_unlockPixels(env, java_bitmap);
        unlocked = true;
      }
    };
    try {
      jobject result = ExecuteDetectionPipeline<PixelFormat::kRgba8888>(
          env, *runtime, static_cast<const uint32_t*>(pixels_ptr),
          source_width, source_height, stride_pixels,
          confidence_threshold, unlock_action);
      unlock_action();
      return result;
    } catch (...) {
      unlock_action();
      throw;
    }
  } catch (const std::exception& error) {
    ThrowIllegalState(env, error.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_letmutex_bubblepop_NativeBubblePopRuntime_nativePrewarm(
    JNIEnv* env, jobject, jlong handle, jboolean invoke_once) {
  auto* runtime = reinterpret_cast<Runtime*>(handle);
  if (runtime == nullptr) {
    ThrowIllegalState(env, "Native bubble runtime is closed");
    return -1.0;
  }
  try {
    std::lock_guard<std::mutex> lock(runtime->mutex);
    const auto started = Clock::now();
    InterpreterState& state = runtime->GetInterpreter();
    if (invoke_once == JNI_TRUE) {
      auto* input = static_cast<float*>(TfLiteTensorData(state.input_tensor));
      if (input == nullptr) {
        throw std::runtime_error("TFLite input tensor is not CPU-accessible");
      }
      const size_t input_values =
          TfLiteTensorByteSize(state.input_tensor) / sizeof(float);
      std::fill(input, input + input_values, 0.0f);
      CheckTfLite(TfLiteInterpreterInvoke(state.interpreter),
                  "TFLite prewarm invocation");
    }
    return Milliseconds(Clock::now() - started);
  } catch (const std::exception& error) {
    ThrowIllegalState(env, error.what());
    return -1.0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_letmutex_bubblepop_NativeBubblePopRuntime_nativeClose(
    JNIEnv*, jobject, jlong handle) {
  delete reinterpret_cast<Runtime*>(handle);
}
