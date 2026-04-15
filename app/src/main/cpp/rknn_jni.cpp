#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "rknn_api.h"

#define TAG "rknn_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

// 初始化 RKNN，加载模型，返回 handle
JNIEXPORT jlong JNICALL
Java_com_example_weblauncher_npu_RknnInference_nativeInit(
        JNIEnv *env, jobject, jbyteArray modelData) {

    jsize modelLen = env->GetArrayLength(modelData);
    jbyte *modelBuf = env->GetByteArrayElements(modelData, nullptr);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, modelBuf, modelLen, 0, nullptr);
    env->ReleaseByteArrayElements(modelData, modelBuf, JNI_ABORT);

    if (ret != RKNN_SUCC) {
        LOGE("rknn_init failed: %d", ret);
        return 0;
    }

    // 打印模型信息
    rknn_sdk_version version;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    LOGI("RKNN SDK version: %s, driver: %s", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    LOGI("Input num: %d, Output num: %d", io_num.n_input, io_num.n_output);

    return (jlong) ctx;
}

// NPU 推理，输入 RGB 数据，返回 float 数组
JNIEXPORT jfloatArray JNICALL
Java_com_example_weblauncher_npu_RknnInference_nativeInfer(
        JNIEnv *env, jobject, jlong handle, jbyteArray rgbData) {

    rknn_context ctx = (rknn_context) handle;
    if (ctx == 0) return nullptr;

    jsize dataLen = env->GetArrayLength(rgbData);
    jbyte *dataBuf = env->GetByteArrayElements(rgbData, nullptr);

    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = dataLen;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = dataBuf;

    int ret = rknn_inputs_set(ctx, 1, inputs);
    env->ReleaseByteArrayElements(rgbData, dataBuf, JNI_ABORT);

    if (ret != RKNN_SUCC) {
        LOGE("rknn_inputs_set failed: %d", ret);
        return nullptr;
    }

    // 运行推理
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        LOGE("rknn_run failed: %d", ret);
        return nullptr;
    }

    // 获取输出数量
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    // 获取输出
    std::vector<rknn_output> outputs(io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        LOGE("rknn_outputs_get failed: %d", ret);
        return nullptr;
    }

    // 把第一个输出转成 Java float 数组
    float *outData = (float *) outputs[0].buf;
    int outSize = outputs[0].size / sizeof(float);

    jfloatArray result = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(result, 0, outSize, outData);

    rknn_outputs_release(ctx, io_num.n_output, outputs.data());

    return result;
}

// 释放 RKNN context
JNIEXPORT void JNICALL
Java_com_example_weblauncher_npu_RknnInference_nativeRelease(
        JNIEnv *, jobject, jlong handle) {
    if (handle != 0) {
        rknn_destroy((rknn_context) handle);
        LOGI("RKNN context released");
    }
}

// ---- HandInference JNI (same logic, different class name) ----

JNIEXPORT jlong JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeInit(
        JNIEnv *env, jobject, jbyteArray modelData) {
    jsize modelLen = env->GetArrayLength(modelData);
    jbyte *modelBuf = env->GetByteArrayElements(modelData, nullptr);
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, modelBuf, modelLen, 0, nullptr);
    env->ReleaseByteArrayElements(modelData, modelBuf, JNI_ABORT);
    if (ret != RKNN_SUCC) { LOGE("HandInference rknn_init failed: %d", ret); return 0; }
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    LOGI("HandInference loaded: in=%d out=%d", io_num.n_input, io_num.n_output);
    return (jlong) ctx;
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeInfer(
        JNIEnv *env, jobject, jlong handle, jbyteArray rgbData) {
    rknn_context ctx = (rknn_context) handle;
    if (ctx == 0) return nullptr;
    jsize dataLen = env->GetArrayLength(rgbData);
    jbyte *dataBuf = env->GetByteArrayElements(rgbData, nullptr);
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = dataLen;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = dataBuf;
    int ret = rknn_inputs_set(ctx, 1, inputs);
    env->ReleaseByteArrayElements(rgbData, dataBuf, JNI_ABORT);
    if (ret != RKNN_SUCC) { LOGE("HandInference inputs_set failed: %d", ret); return nullptr; }
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) { LOGE("HandInference run failed: %d", ret); return nullptr; }
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    std::vector<rknn_output> outputs(io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) outputs[i].want_float = 1;
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) { LOGE("HandInference outputs_get failed: %d", ret); return nullptr; }
    float *outData = (float *) outputs[0].buf;
    int outSize = outputs[0].size / sizeof(float);
    jfloatArray result = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(result, 0, outSize, outData);
    rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    return result;
}

JNIEXPORT void JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeRelease(
        JNIEnv *, jobject, jlong handle) {
    if (handle != 0) { rknn_destroy((rknn_context) handle); LOGI("HandInference released"); }
}

} // extern "C"
