# LiteRT Interpreter runtime 1.4.2

This directory contains the native C API headers and Android shared libraries
extracted from Google's official `com.google.ai.edge.litert:litert:1.4.2` and
`com.google.ai.edge.litert:litert-gpu:1.4.2` AARs.

The XNNPACK header comes from the official LiteRT source tree. Its option ABI
was checked against the packaged 1.4.2 runtime before enabling the explicit
four-thread CPU delegate.

Only the application ABIs are retained: `arm64-v8a`, `armeabi-v7a`, and
`x86_64`. The accompanying AAR license files are preserved in this directory.
LiteRT is licensed under Apache License 2.0.
