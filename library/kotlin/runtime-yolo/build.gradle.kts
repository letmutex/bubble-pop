plugins {
    id("com.android.library")
    id("com.vanniktech.maven.publish")
}

dependencies {
    api(project(":core"))
}

android {
    namespace = "letmutex.bubblepop.runtime.yolo"
    compileSdk = 37

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++20", "-O3")
            }
        }
    }

    androidResources {
        noCompress += "tflite"
    }

    sourceSets {
        getByName("main") {
            jniLibs.directories.add("../third_party/litert-interpreter-1.4.2/jni")
        }
    }

    ndkVersion = "28.2.13676358"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
