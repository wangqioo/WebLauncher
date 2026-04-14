cmake_minimum_required(VERSION 3.22.1)
project("weblauncher_npu")

# RKNN 头文件路径（放在 cpp/include/）
include_directories(${CMAKE_SOURCE_DIR}/include)

# RKNN 运行时库（放在 jniLibs/arm64-v8a/librknnrt.so）
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/librknnrt.so)

# 编译 JNI 库
add_library(rknn_jni SHARED rknn_jni.cpp)

target_link_libraries(rknn_jni
    rknnrt
    android
    log)
