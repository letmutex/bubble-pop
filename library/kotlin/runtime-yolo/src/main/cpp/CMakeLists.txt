cmake_minimum_required(VERSION 3.22.1)
project(bubblepop_yolo_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLITE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/litert-interpreter-1.4.2")

if(NOT EXISTS "${TFLITE_ROOT}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so" OR
   NOT EXISTS "${TFLITE_ROOT}/jni/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so")
    message(FATAL_ERROR "LiteRT Interpreter SDK does not contain Android ABI ${ANDROID_ABI}")
endif()

add_library(tflite_runtime SHARED IMPORTED)
set_target_properties(tflite_runtime PROPERTIES
    IMPORTED_LOCATION "${TFLITE_ROOT}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_ROOT}/include")

add_library(tflite_gpu_delegate SHARED IMPORTED)
set_target_properties(tflite_gpu_delegate PROPERTIES
    IMPORTED_LOCATION "${TFLITE_ROOT}/jni/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_ROOT}/include")

add_library(bubblepop_yolo_runtime SHARED runtime_native.cpp)
target_compile_options(bubblepop_yolo_runtime PRIVATE -O3)
target_link_libraries(bubblepop_yolo_runtime PRIVATE
    tflite_runtime tflite_gpu_delegate android log)
