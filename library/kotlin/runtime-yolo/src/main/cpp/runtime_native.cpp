#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

constexpr int kInputWidth = 1600;
constexpr int kInputHeight = 1600;
constexpr int kDetectionChannels = 37;
constexpr int kMaskChannels = 32;
constexpr int kCpuThreads = 4;
constexpr float kMaskLogitThreshold = 0.0f;  // sigmoid(logit) >= 0.50
constexpr int kMinAreaPixels = 60;
constexpr int kMaxPathPoints = 512;
constexpr float kNmsIouThreshold = 0.70f;
constexpr int kMaxDetections = 300;
constexpr float kLetterboxValue = 114.0f / 255.0f;
constexpr int kDx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int kDy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

enum class Backend : int {
  kCpu = 0,
  kGpu = 1,
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
  const TfLiteTensor* detection_tensor = nullptr;
  const TfLiteTensor* prototype_tensor = nullptr;
  int input_width = 0;
  int input_height = 0;
  int detection_count = 0;
  int prototype_height = 0;
  int prototype_width = 0;

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
  std::vector<uint8_t> visited;
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
    const bool use_gpu = backend == Backend::kGpu;
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
        TfLiteInterpreterGetOutputTensorCount(created->interpreter) != 2) {
      throw std::runtime_error("Unexpected model input/output count");
    }
    created->input_tensor =
        TfLiteInterpreterGetInputTensor(created->interpreter, 0);
    if (created->input_tensor == nullptr ||
        TfLiteTensorType(created->input_tensor) != kTfLiteFloat32 ||
        TfLiteTensorNumDims(created->input_tensor) != 4 ||
        TfLiteTensorDim(created->input_tensor, 0) != 1 ||
        TfLiteTensorDim(created->input_tensor, 3) != 3) {
      throw std::runtime_error("Unexpected model input tensor");
    }
    created->input_height = TfLiteTensorDim(created->input_tensor, 1);
    created->input_width = TfLiteTensorDim(created->input_tensor, 2);
    if (created->input_height != kInputHeight ||
        created->input_width != kInputWidth) {
      throw std::runtime_error(
          "Unexpected input; expected [1,1600,1600,3] float32");
    }
    for (int i = 0; i < 2; ++i) {
      const TfLiteTensor* output =
          TfLiteInterpreterGetOutputTensor(created->interpreter, i);
      if (output == nullptr || TfLiteTensorType(output) != kTfLiteFloat32 ||
          TfLiteTensorDim(output, 0) != 1) {
        throw std::runtime_error("Unexpected output tensor");
      }
      if (TfLiteTensorNumDims(output) == 3 &&
          TfLiteTensorDim(output, 1) == kDetectionChannels) {
        created->detection_tensor = output;
        created->detection_count = TfLiteTensorDim(output, 2);
      } else if (TfLiteTensorNumDims(output) == 4 &&
                 TfLiteTensorDim(output, 3) == kMaskChannels) {
        created->prototype_tensor = output;
        created->prototype_height = TfLiteTensorDim(output, 1);
        created->prototype_width = TfLiteTensorDim(output, 2);
      }
    }
    if (created->detection_tensor == nullptr ||
        created->prototype_tensor == nullptr ||
        created->detection_count <= 0 || created->prototype_height <= 0 ||
        created->prototype_width <= 0) {
      throw std::runtime_error(
          "Unexpected outputs; expected [1,37,N] and [1,H,W,32]");
    }
    __android_log_print(
        ANDROID_LOG_INFO, "BubbleNative",
        "TFLite %s backend=%s input=%dx%d", TfLiteVersion(),
        use_gpu ? "gpu-fp16" : "cpu-fp16", created->input_width,
        created->input_height);

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

Letterbox CalculateLetterbox(int source_width, int source_height,
                             int input_width, int input_height) {
  const float scale = std::min(
      static_cast<float>(input_width) / source_width,
      static_cast<float>(input_height) / source_height);
  const int resized_width = std::max(1, static_cast<int>(std::lround(source_width * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::lround(source_height * scale)));
  return {scale, resized_width, resized_height,
          (input_width - resized_width) / 2,
          (input_height - resized_height) / 2};
}

inline float Channel(uint32_t color, int channel) {
  const int shift = channel == 0 ? 16 : (channel == 1 ? 8 : 0);
  return static_cast<float>((color >> shift) & 0xff) / 255.0f;
}

void Preprocess(const jint* pixels, int source_width, int source_height,
                const Letterbox& box, int input_width, int input_height,
                float* destination) {
  for (int y = 0; y < input_height; ++y) {
    for (int x = 0; x < input_width; ++x) {
      float normalized[3] = {
          kLetterboxValue, kLetterboxValue, kLetterboxValue};
      const int local_x = x - box.pad_x;
      const int local_y = y - box.pad_y;
      if (local_x >= 0 && local_x < box.resized_width &&
          local_y >= 0 && local_y < box.resized_height) {
        const float sx = std::clamp(
            (local_x + 0.5f) * source_width / box.resized_width - 0.5f,
            0.0f, static_cast<float>(source_width - 1));
        const float sy = std::clamp(
            (local_y + 0.5f) * source_height / box.resized_height - 0.5f,
            0.0f, static_cast<float>(source_height - 1));
        const int x0 = static_cast<int>(sx);
        const int y0 = static_cast<int>(sy);
        const int x1 = std::min(x0 + 1, source_width - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float wx = sx - x0;
        const float wy = sy - y0;
        const uint32_t p00 = static_cast<uint32_t>(pixels[y0 * source_width + x0]);
        const uint32_t p10 = static_cast<uint32_t>(pixels[y0 * source_width + x1]);
        const uint32_t p01 = static_cast<uint32_t>(pixels[y1 * source_width + x0]);
        const uint32_t p11 = static_cast<uint32_t>(pixels[y1 * source_width + x1]);
        for (int channel = 0; channel < 3; ++channel) {
          const float top = Channel(p00, channel) +
                            (Channel(p10, channel) - Channel(p00, channel)) * wx;
          const float bottom = Channel(p01, channel) +
                               (Channel(p11, channel) - Channel(p01, channel)) * wx;
          normalized[channel] = top + (bottom - top) * wy;
        }
      }
      const int pixel = y * input_width + x;
      destination[pixel * 3] = normalized[0];
      destination[pixel * 3 + 1] = normalized[1];
      destination[pixel * 3 + 2] = normalized[2];
    }
  }
}

inline float Sigmoid(float value) {
  value = std::clamp(value, -30.0f, 30.0f);
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
                                   const int* queue, int count,
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

struct Bubble {
  float confidence;
  float mask_probability;
  int area_pixels;
  std::vector<float> points;
};

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float confidence;
  int index;
};

float IntersectionOverUnion(const Detection& left, const Detection& right) {
  const float intersection_width =
      std::max(0.0f, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float intersection_height =
      std::max(0.0f, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = intersection_width * intersection_height;
  const float left_area = std::max(0.0f, left.x2 - left.x1) *
                          std::max(0.0f, left.y2 - left.y1);
  const float right_area = std::max(0.0f, right.x2 - right.x1) *
                           std::max(0.0f, right.y2 - right.y1);
  const float union_area = left_area + right_area - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<Detection> ApplyNms(const float* predictions, int count,
                                float confidence_threshold) {
  std::vector<Detection> candidates;
  for (int index = 0; index < count; ++index) {
    const float confidence = predictions[4 * count + index];
    if (!std::isfinite(confidence) || confidence < confidence_threshold) continue;
    const float center_x = predictions[index];
    const float center_y = predictions[count + index];
    const float width = predictions[2 * count + index];
    const float height = predictions[3 * count + index];
    if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
        !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0f || height <= 0.0f) {
      continue;
    }
    candidates.push_back({center_x - width * 0.5f,
                          center_y - height * 0.5f,
                          center_x + width * 0.5f,
                          center_y + height * 0.5f,
                          confidence, index});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& left, const Detection& right) {
              return left.confidence > right.confidence;
            });
  std::vector<Detection> selected;
  selected.reserve(std::min(static_cast<int>(candidates.size()),
                            kMaxDetections));
  for (const Detection& candidate : candidates) {
    bool suppressed = false;
    for (const Detection& accepted : selected) {
      if (IntersectionOverUnion(candidate, accepted) > kNmsIouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      selected.push_back(candidate);
      if (selected.size() == kMaxDetections) break;
    }
  }
  return selected;
}

std::vector<Bubble> ExtractBubbles(Runtime& runtime, const float* predictions,
                                   int detection_count,
                                   const float* prototypes,
                                   int prototype_width, int prototype_height,
                                   int input_width, int input_height,
                                   const Letterbox& box,
                                   float confidence_threshold) {
  const std::vector<Detection> detections =
      ApplyNms(predictions, detection_count, confidence_threshold);
  const int plane = prototype_width * prototype_height;
  runtime.visited.resize(plane);
  runtime.component.resize(plane);
  runtime.queue.resize(plane);
  std::vector<int> largest_component;
  std::vector<float> mask_logits(plane);
  std::vector<Bubble> bubbles;
  bubbles.reserve(detections.size());

  const float prototype_to_input_x =
      static_cast<float>(input_width) / prototype_width;
  const float prototype_to_input_y =
      static_cast<float>(input_height) / prototype_height;
  const float source_area_per_prototype_pixel =
      prototype_to_input_x * prototype_to_input_y /
      (box.scale * box.scale);
  const int min_prototype_area = std::max(
      1, static_cast<int>(std::ceil(
             kMinAreaPixels / source_area_per_prototype_pixel)));

  for (const Detection& detection : detections) {
    std::fill(runtime.visited.begin(), runtime.visited.end(), 0);
    std::fill(runtime.component.begin(), runtime.component.end(), 0);
    const int left = std::clamp(
        static_cast<int>(std::floor(detection.x1 * prototype_width)),
        0, prototype_width);
    const int top = std::clamp(
        static_cast<int>(std::floor(detection.y1 * prototype_height)),
        0, prototype_height);
    const int right = std::clamp(
        static_cast<int>(std::ceil(detection.x2 * prototype_width)),
        0, prototype_width);
    const int bottom = std::clamp(
        static_cast<int>(std::ceil(detection.y2 * prototype_height)),
        0, prototype_height);
    if (left >= right || top >= bottom) continue;

    for (int y = top; y < bottom; ++y) {
      for (int x = left; x < right; ++x) {
        const int pixel = y * prototype_width + x;
        const float* prototype = prototypes + pixel * kMaskChannels;
        float logit = 0.0f;
        for (int channel = 0; channel < kMaskChannels; ++channel) {
          const float coefficient =
              predictions[(5 + channel) * detection_count + detection.index];
          logit += coefficient * prototype[channel];
        }
        mask_logits[pixel] = logit;
        runtime.component[pixel] = logit >= kMaskLogitThreshold;
      }
    }

    largest_component.clear();
    for (int start = top * prototype_width; start < bottom * prototype_width;
         ++start) {
      const int x = start % prototype_width;
      if (x < left || x >= right || runtime.visited[start] ||
          !runtime.component[start]) {
        continue;
      }
      int head = 0;
      int tail = 0;
      runtime.queue[tail++] = start;
      runtime.visited[start] = 1;
      while (head < tail) {
        const int index = runtime.queue[head++];
        const int px = index % prototype_width;
        const int py = index / prototype_width;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = px + dx;
            const int ny = py + dy;
            if (nx < left || nx >= right || ny < top || ny >= bottom) continue;
            const int next = ny * prototype_width + nx;
            if (!runtime.visited[next] && runtime.component[next]) {
              runtime.visited[next] = 1;
              runtime.queue[tail++] = next;
            }
          }
        }
      }
      if (tail > static_cast<int>(largest_component.size())) {
        largest_component.assign(runtime.queue.begin(),
                                 runtime.queue.begin() + tail);
      }
    }
    if (largest_component.size() < static_cast<size_t>(min_prototype_area)) {
      continue;
    }

    std::fill(runtime.component.begin(), runtime.component.end(), 0);
    double mask_probability_sum = 0.0;
    for (int index : largest_component) {
      runtime.component[index] = 1;
      mask_probability_sum += Sigmoid(mask_logits[index]);
    }
    const std::vector<IntPoint> contour = TraceContour(
        runtime.component, largest_component.data(),
        static_cast<int>(largest_component.size()), prototype_width,
        prototype_height);
    if (contour.size() < 3) continue;

    const int sample_step = std::max(
        1, static_cast<int>(std::ceil(
               static_cast<double>(contour.size()) / kMaxPathPoints)));
    Bubble bubble{
        detection.confidence,
        static_cast<float>(mask_probability_sum / largest_component.size()),
        static_cast<int>(std::lround(largest_component.size() *
                                     source_area_per_prototype_pixel)),
        {}};
    bubble.points.reserve((contour.size() / sample_step + 1) * 2);
    for (size_t i = 0; i < contour.size(); i += sample_step) {
      const float input_x = (contour[i].x + 0.5f) * prototype_to_input_x;
      const float input_y = (contour[i].y + 0.5f) * prototype_to_input_y;
      bubble.points.push_back(std::clamp(
          (input_x - box.pad_x) / box.resized_width, 0.0f, 1.0f));
      bubble.points.push_back(std::clamp(
          (input_y - box.pad_y) / box.resized_height, 0.0f, 1.0f));
    }
    if (bubble.points.size() >= 6) bubbles.push_back(std::move(bubble));
  }
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
                    double inference_ms, double native_total_ms) {
  jfloatArray java_packed = env->NewFloatArray(static_cast<jsize>(packed.size()));
  if (java_packed == nullptr) return nullptr;
  if (!packed.empty()) {
    env->SetFloatArrayRegion(java_packed, 0, static_cast<jsize>(packed.size()),
                            packed.data());
    if (env->ExceptionCheck()) return nullptr;
  }
  jclass payload_class =
      env->FindClass("letmutex/bubblepop/NativeYoloDetectionPayload");
  if (payload_class == nullptr) return nullptr;
  jmethodID constructor = env->GetMethodID(payload_class, "<init>", "([FDD)V");
  if (constructor == nullptr) return nullptr;
  return env->NewObject(payload_class, constructor, java_packed, inference_ms,
                        native_total_ms);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_letmutex_bubblepop_NativeYoloRuntime_nativeCreate(
    JNIEnv* env, jobject, jobject java_asset_manager, jstring java_cache_dir,
    jint backend_value, jstring java_asset_path, jstring java_model_token) {
  if (backend_value < static_cast<int>(Backend::kCpu) ||
      backend_value > static_cast<int>(Backend::kGpu)) {
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
Java_letmutex_bubblepop_NativeYoloRuntime_nativeDetect(
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

  try {
    std::lock_guard<std::mutex> lock(runtime->mutex);
    const auto native_started = Clock::now();
    double inference_ms = 0.0;

    // Model loading/accelerator compilation can be slow on first use. Resolve
    // it before obtaining the Java pixel array so the VM is never pinned while
    // TFLite initializes.
    InterpreterState& state = runtime->GetInterpreter();
    const Letterbox letterbox = CalculateLetterbox(
        source_width, source_height, state.input_width, state.input_height);

    jint* pixels = env->GetIntArrayElements(java_pixels, nullptr);
    if (pixels == nullptr) throw std::runtime_error("Unable to access ARGB pixels");
    try {
      auto* input_address =
          static_cast<float*>(TfLiteTensorData(state.input_tensor));
      if (input_address == nullptr) {
        throw std::runtime_error("TFLite input tensor is not CPU-accessible");
      }
      Preprocess(pixels, source_width, source_height, letterbox,
                 state.input_width, state.input_height, input_address);
      env->ReleaseIntArrayElements(java_pixels, pixels, JNI_ABORT);
      pixels = nullptr;

      const auto inference_started = Clock::now();
      CheckTfLite(TfLiteInterpreterInvoke(state.interpreter),
                  "TfLiteInterpreterInvoke");
      inference_ms = Milliseconds(Clock::now() - inference_started);
      const float* predictions = static_cast<const float*>(
          TfLiteTensorData(state.detection_tensor));
      const float* prototypes = static_cast<const float*>(
          TfLiteTensorData(state.prototype_tensor));
      if (predictions == nullptr || prototypes == nullptr) {
        throw std::runtime_error("Outputs are not CPU-accessible");
      }
      const std::vector<Bubble> bubbles = ExtractBubbles(
          *runtime, predictions, state.detection_count, prototypes,
          state.prototype_width, state.prototype_height, state.input_width,
          state.input_height, letterbox, confidence_threshold);
      const std::vector<float> packed = PackBubbles(bubbles);
      return MakePayload(env, packed, inference_ms,
                         Milliseconds(Clock::now() - native_started));
    } catch (...) {
      if (pixels != nullptr) {
        env->ReleaseIntArrayElements(java_pixels, pixels, JNI_ABORT);
      }
      throw;
    }
  } catch (const std::exception& error) {
    ThrowIllegalState(env, error.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_letmutex_bubblepop_NativeYoloRuntime_nativePrewarm(
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
Java_letmutex_bubblepop_NativeYoloRuntime_nativeClose(
    JNIEnv*, jobject, jlong handle) {
  delete reinterpret_cast<Runtime*>(handle);
}
