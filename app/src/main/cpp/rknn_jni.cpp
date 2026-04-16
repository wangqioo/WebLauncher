#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <android/log.h>
#include "rknn_api.h"

#define TAG "rknn_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static long nowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000L + ts.tv_nsec / 1000000L);
}

static float sigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// DFL decode: softmax + weighted sum for one direction group (16 bins)
static float dflDecode(const float* dfl, int group) {
    float bins[16];
    float maxV = dfl[group*16];
    for (int i = 0; i < 16; i++) { bins[i] = dfl[group*16+i]; if (bins[i] > maxV) maxV = bins[i]; }
    float sum = 0;
    for (int i = 0; i < 16; i++) { bins[i] = expf(bins[i] - maxV); sum += bins[i]; }
    float val = 0;
    for (int i = 0; i < 16; i++) val += i * (bins[i] / sum);
    return val;
}

// YOLOv8-pose decode: returns JSON string directly, avoids copying 974K floats to Java
// outputs: [out0, out1, out2, out3] already split
// out0/1/2: [65, gridH*gridW], out3: [51, 8400]
static std::string poseDecodeJson(
        const float** outs, const int* sizes,
        int origW, int origH, int padTop, int padLeft, float scale,
        float objThresh, int inputSize) {

    const int strideCfg[3][3] = {{8,80,80},{16,40,40},{32,20,20}};
    const int indexOffsets[3] = {0, 80*80, 80*80+40*40};

    float bestConf = objThresh;
    int bestAnchorIdx = -1;
    float bestBox[4] = {0,0,0,0};

    for (int si = 0; si < 3; si++) {
        int stride = strideCfg[si][0];
        int gridH  = strideCfg[si][1];
        int gridW  = strideCfg[si][2];
        int gridSize = gridH * gridW;
        const float* feat = outs[si];

        for (int h = 0; h < gridH; h++) {
            for (int w = 0; w < gridW; w++) {
                int pos = h * gridW + w;
                float conf = sigmoidF(feat[64 * gridSize + pos]);
                if (conf > bestConf) {
                    bestConf = conf;
                    bestAnchorIdx = indexOffsets[si] + pos;
                    float dfl[64];
                    for (int c = 0; c < 64; c++) dfl[c] = feat[c * gridSize + pos];
                    float x1 = (w + 0.5f) - dflDecode(dfl, 0);
                    float y1 = (h + 0.5f) - dflDecode(dfl, 1);
                    float x2 = (w + 0.5f) + dflDecode(dfl, 2);
                    float y2 = (h + 0.5f) + dflDecode(dfl, 3);
                    bestBox[0] = x1 * stride; bestBox[1] = y1 * stride;
                    bestBox[2] = x2 * stride; bestBox[3] = y2 * stride;
                }
            }
        }
    }

    if (bestAnchorIdx < 0) return "{\"detected\":false,\"confidence\":0}";

    float invScaleW = 1.0f / scale / origW;
    float invScaleH = 1.0f / scale / origH;
    auto clamp01 = [](float v) { return v < 0 ? 0.0f : v > 1.0f ? 1.0f : v; };
    float nx1 = clamp01((bestBox[0] - padLeft) * invScaleW);
    float ny1 = clamp01((bestBox[1] - padTop)  * invScaleH);
    float nx2 = clamp01((bestBox[2] - padLeft) * invScaleW);
    float ny2 = clamp01((bestBox[3] - padTop)  * invScaleH);

    const float* kpFlat = outs[3]; // [51, 8400]
    const int totalAnchors = 8400;
    const char* kpNames[] = {"nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"};

    char buf[4096];
    int pos = 0;
    pos += snprintf(buf+pos, sizeof(buf)-pos,
        "{\"detected\":true,\"confidence\":%.4f,\"box\":{\"x\":%.4f,\"y\":%.4f,\"x2\":%.4f,\"y2\":%.4f},\"keypoints\":[",
        bestConf, nx1, ny1, nx2, ny2);

    for (int k = 0; k < 17; k++) {
        float kx = kpFlat[(k*3)   * totalAnchors + bestAnchorIdx];
        float ky = kpFlat[(k*3+1) * totalAnchors + bestAnchorIdx];
        float ks = sigmoidF(kpFlat[(k*3+2) * totalAnchors + bestAnchorIdx]);
        float nkx = clamp01((kx - padLeft) * invScaleW);
        float nky = clamp01((ky - padTop)  * invScaleH);
        if (k > 0) pos += snprintf(buf+pos, sizeof(buf)-pos, ",");
        pos += snprintf(buf+pos, sizeof(buf)-pos,
            "{\"name\":\"%s\",\"x\":%.4f,\"y\":%.4f,\"score\":%.4f}",
            kpNames[k], nkx, nky, ks);
    }
    pos += snprintf(buf+pos, sizeof(buf)-pos, "]}");
    return std::string(buf);
}

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

    // 把所有输出拼接成一个 float 数组返回
    // 格式: [总长度, out0大小, out1大小, ..., out0数据..., out1数据..., ...]
    int totalFloats = 0;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        totalFloats += outputs[i].size / sizeof(float);
    }

    // 头部: n_output+1 个 int 作为 metadata (用 float 存)
    // [n_output, size0, size1, size2, size3, data0..., data1..., ...]
    int headerSize = 1 + io_num.n_output;
    jfloatArray result = env->NewFloatArray(headerSize + totalFloats);

    // 写 header
    std::vector<float> header(headerSize);
    header[0] = (float) io_num.n_output;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        header[1 + i] = (float)(outputs[i].size / sizeof(float));
    }
    env->SetFloatArrayRegion(result, 0, headerSize, header.data());

    // 写各输出数据
    int offset = headerSize;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        float *outData = (float *) outputs[i].buf;
        int outSize = outputs[i].size / sizeof(float);
        env->SetFloatArrayRegion(result, offset, outSize, outData);
        offset += outSize;
    }

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

// ---- 带 resize 的推理（避免 Java 层像素循环）----
// letterbox resize: nearest-neighbor, gray padding (0x38)
static void letterboxResize(const uint8_t* src, int srcW, int srcH,
                             uint8_t* dst, int dstW, int dstH,
                             int* padTop_out, int* padLeft_out, float* scale_out) {
    float scale = std::min((float)dstW / srcW, (float)dstH / srcH);
    int scaledW = (int)(srcW * scale + 0.5f);
    int scaledH = (int)(srcH * scale + 0.5f);
    int padLeft = (dstW - scaledW) / 2;
    int padTop  = (dstH - scaledH) / 2;
    if (padTop_out)  *padTop_out  = padTop;
    if (padLeft_out) *padLeft_out = padLeft;
    if (scale_out)   *scale_out   = scale;

    // 灰色填充
    memset(dst, 0x38, dstW * dstH * 3);

    float inv = 1.0f / scale;
    for (int dy = 0; dy < scaledH; dy++) {
        int srcy = std::min(srcH-1, (int)(dy * inv));
        for (int dx = 0; dx < scaledW; dx++) {
            int srcx = std::min(srcW-1, (int)(dx * inv));
            int si = (srcy * srcW + srcx) * 3;
            int di = ((padTop + dy) * dstW + (padLeft + dx)) * 3;
            dst[di]   = src[si];
            dst[di+1] = src[si+1];
            dst[di+2] = src[si+2];
        }
    }
}

// nearest-neighbor resize (no padding)
static void nnResize(const uint8_t* src, int srcW, int srcH,
                     uint8_t* dst, int dstW, int dstH) {
    float sx = (float)srcW / dstW;
    float sy = (float)srcH / dstH;
    for (int dy = 0; dy < dstH; dy++) {
        int srcy = std::min(srcH-1, (int)(dy * sy));
        for (int dx = 0; dx < dstW; dx++) {
            int srcx = std::min(srcW-1, (int)(dx * sx));
            int si = (srcy * srcW + srcx) * 3;
            int di = (dy * dstW + dx) * 3;
            dst[di]   = src[si];
            dst[di+1] = src[si+1];
            dst[di+2] = src[si+2];
        }
    }
}

// 通用推理 helper
static jfloatArray doInfer(JNIEnv* env, rknn_context ctx,
                           const uint8_t* rgbBuf, int size) {
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = size;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = (void*)rgbBuf;
    if (rknn_inputs_set(ctx, 1, inputs) != RKNN_SUCC) return nullptr;
    if (rknn_run(ctx, nullptr)          != RKNN_SUCC) return nullptr;

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    std::vector<rknn_output> outputs(io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output)*io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) outputs[i].want_float = 1;
    if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr) != RKNN_SUCC) return nullptr;

    int totalFloats = 0;
    for (uint32_t i = 0; i < io_num.n_output; i++) totalFloats += outputs[i].size / sizeof(float);
    int headerSize = 1 + io_num.n_output;
    jfloatArray result = env->NewFloatArray(headerSize + totalFloats);
    std::vector<float> header(headerSize);
    header[0] = (float)io_num.n_output;
    for (uint32_t i = 0; i < io_num.n_output; i++) header[1+i] = (float)(outputs[i].size/sizeof(float));
    env->SetFloatArrayRegion(result, 0, headerSize, header.data());
    int offset = headerSize;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        int sz = outputs[i].size / sizeof(float);
        env->SetFloatArrayRegion(result, offset, sz, (float*)outputs[i].buf);
        offset += sz;
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    return result;
}

// RknnInference: 接收原始 RGB + 尺寸，内部做 letterbox resize
// 返回: [padTop, padLeft, scaledW_unused, ...] 作为 float[] 头部，后跟推理结果
// 实际返回格式与 nativeInfer 相同，但额外在最前面插入 padTop/padLeft/scale
JNIEXPORT jfloatArray JNICALL
Java_com_example_weblauncher_npu_RknnInference_nativeInferResized(
        JNIEnv *env, jobject, jlong handle,
        jbyteArray rgbData, jint srcW, jint srcH, jint dstW, jint dstH) {
    rknn_context ctx = (rknn_context)handle;
    if (ctx == 0) return nullptr;
    jbyte* src = env->GetByteArrayElements(rgbData, nullptr);
    std::vector<uint8_t> dst(dstW * dstH * 3);
    int padTop = 0, padLeft = 0;
    float scale = 1.0f;
    letterboxResize((const uint8_t*)src, srcW, srcH, dst.data(), dstW, dstH, &padTop, &padLeft, &scale);
    env->ReleaseByteArrayElements(rgbData, src, JNI_ABORT);

    jfloatArray inner = doInfer(env, ctx, dst.data(), dstW * dstH * 3);
    if (!inner) return nullptr;

    // 在最前面插入 3 个 float: padTop, padLeft, scale
    jsize innerLen = env->GetArrayLength(inner);
    jfloatArray result = env->NewFloatArray(3 + innerLen);
    float meta[3] = {(float)padTop, (float)padLeft, scale};
    env->SetFloatArrayRegion(result, 0, 3, meta);
    std::vector<float> tmp(innerLen);
    env->GetFloatArrayRegion(inner, 0, innerLen, tmp.data());
    env->SetFloatArrayRegion(result, 3, innerLen, tmp.data());
    return result;
}

// HandInference: 接收原始 RGB，内部做 nearest-neighbor resize 到指定尺寸
JNIEXPORT jfloatArray JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeInferResized(
        JNIEnv *env, jobject, jlong handle,
        jbyteArray rgbData, jint srcW, jint srcH, jint dstW, jint dstH) {
    rknn_context ctx = (rknn_context)handle;
    if (ctx == 0) return nullptr;
    jbyte* src = env->GetByteArrayElements(rgbData, nullptr);
    std::vector<uint8_t> dst(dstW * dstH * 3);
    nnResize((const uint8_t*)src, srcW, srcH, dst.data(), dstW, dstH);
    env->ReleaseByteArrayElements(rgbData, src, JNI_ABORT);
    return doInfer(env, ctx, dst.data(), dstW * dstH * 3);
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
    int totalFloats = 0;
    for (uint32_t i = 0; i < io_num.n_output; i++) totalFloats += outputs[i].size / sizeof(float);
    int headerSize = 1 + io_num.n_output;
    jfloatArray result = env->NewFloatArray(headerSize + totalFloats);
    std::vector<float> header(headerSize);
    header[0] = (float) io_num.n_output;
    for (uint32_t i = 0; i < io_num.n_output; i++) header[1+i] = (float)(outputs[i].size / sizeof(float));
    env->SetFloatArrayRegion(result, 0, headerSize, header.data());
    int offset = headerSize;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        int sz = outputs[i].size / sizeof(float);
        env->SetFloatArrayRegion(result, offset, sz, (float*)outputs[i].buf);
        offset += sz;
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    return result;
}

JNIEXPORT void JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeRelease(
        JNIEnv *, jobject, jlong handle) {
    if (handle != 0) { rknn_destroy((rknn_context) handle); LOGI("HandInference released"); }
}

// YUV_420_888 → RGB byte[], 同时做 nearest-neighbor resize 到 dstW×dstH
// 输入: yData/uData/vData 对应 Image.Plane[0/1/2] 的 buffer bytes
// yRowStride, uvRowStride, uvPixelStride 来自 getRowStride/getPixelStride
JNIEXPORT jbyteArray JNICALL
Java_com_example_weblauncher_npu_CameraInferenceManager_nativeYuvToRgb(
        JNIEnv *env, jobject,
        jbyteArray yArr, jbyteArray uArr, jbyteArray vArr,
        jint srcW, jint srcH,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint dstW, jint dstH) {

    jbyte* yData = env->GetByteArrayElements(yArr, nullptr);
    jbyte* uData = env->GetByteArrayElements(uArr, nullptr);
    jbyte* vData = env->GetByteArrayElements(vArr, nullptr);

    jbyteArray result = env->NewByteArray(dstW * dstH * 3);
    jbyte* rgb = env->GetByteArrayElements(result, nullptr);

    int sx_fp = (srcW << 16) / dstW;
    int sy_fp = (srcH << 16) / dstH;

    for (int dy = 0; dy < dstH; dy++) {
        int srcY = (dy * sy_fp) >> 16; if (srcY >= srcH) srcY = srcH - 1;
        int uvRow = srcY >> 1;
        const uint8_t* yRow = (const uint8_t*)yData + srcY * yRowStride;
        const uint8_t* uRow = (const uint8_t*)uData + uvRow * uvRowStride;
        const uint8_t* vRow = (const uint8_t*)vData + uvRow * uvRowStride;
        jbyte* dstRow = rgb + dy * dstW * 3;
        for (int dx = 0; dx < dstW; dx++) {
            int srcX = (dx * sx_fp) >> 16; if (srcX >= srcW) srcX = srcW - 1;
            int uvOff = (srcX >> 1) * uvPixelStride;
            int Y = yRow[srcX];
            int U = uRow[uvOff] - 128;
            int V = vRow[uvOff] - 128;
            int r = Y + ((11277 * V) >> 13);
            int g = Y - ((2765  * U + 5726 * V) >> 13);
            int b = Y + ((14216 * U) >> 13);
            dstRow[dx*3]   = (jbyte)(r < 0 ? 0 : r > 255 ? 255 : r);
            dstRow[dx*3+1] = (jbyte)(g < 0 ? 0 : g > 255 ? 255 : g);
            dstRow[dx*3+2] = (jbyte)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }

    env->ReleaseByteArrayElements(yArr, yData, JNI_ABORT);
    env->ReleaseByteArrayElements(uArr, uData, JNI_ABORT);
    env->ReleaseByteArrayElements(vArr, vData, JNI_ABORT);
    env->ReleaseByteArrayElements(result, rgb, 0);
    return result;
}

// YUV_420_888 → letterbox → pose 推理 → JSON 字符串
// 直接在 C++ 做 decode，避免传 974K float 到 Java
JNIEXPORT jstring JNICALL
Java_com_example_weblauncher_npu_RknnInference_nativeInferYuv(
        JNIEnv *env, jobject, jlong handle,
        jbyteArray yArr, jbyteArray uArr, jbyteArray vArr,
        jint srcW, jint srcH,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint dstW, jint dstH) {

    rknn_context ctx = (rknn_context)handle;
    if (ctx == 0) return env->NewStringUTF("{\"error\":\"invalid handle\"}");

    long t0 = nowMs();
    jbyte* yData = env->GetByteArrayElements(yArr, nullptr);
    jbyte* uData = env->GetByteArrayElements(uArr, nullptr);
    jbyte* vData = env->GetByteArrayElements(vArr, nullptr);

    float scale = std::min((float)dstW / srcW, (float)dstH / srcH);
    int scaledW = (int)(srcW * scale + 0.5f);
    int scaledH = (int)(srcH * scale + 0.5f);
    int padLeft = (dstW - scaledW) / 2;
    int padTop  = (dstH - scaledH) / 2;

    std::vector<uint8_t> dst(dstW * dstH * 3, 0x38);
    // "push" 模式: 遍历源像素写到目标，只循环 srcH×srcW 次
    // 每个源像素对应目标区域 scale×scale，用整数 Q16 定点
    int scale_fp = (int)(scale * 65536 + 0.5f); // scale in Q16
    for (int sy = 0; sy < srcH; sy++) {
        int uvRow = sy >> 1;
        const uint8_t* yRow = (const uint8_t*)yData + sy * yRowStride;
        const uint8_t* uRow = (const uint8_t*)uData + uvRow * uvRowStride;
        const uint8_t* vRow = (const uint8_t*)vData + uvRow * uvRowStride;
        int dy_base = padTop + ((sy * scale_fp) >> 16);
        int dy_end  = padTop + (((sy + 1) * scale_fp) >> 16);
        if (dy_end > padTop + scaledH) dy_end = padTop + scaledH;
        for (int sx = 0; sx < srcW; sx++) {
            int uvOff = (sx >> 1) * uvPixelStride;
            int Y = yRow[sx];
            int U = uRow[uvOff] - 128;
            int V = vRow[uvOff] - 128;
            int r = Y + ((11277 * V) >> 13);
            int g = Y - ((2765  * U + 5726 * V) >> 13);
            int b = Y + ((14216 * U) >> 13);
            uint8_t rb = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            uint8_t gb = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            uint8_t bb = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
            int dx_base = padLeft + ((sx * scale_fp) >> 16);
            int dx_end  = padLeft + (((sx + 1) * scale_fp) >> 16);
            if (dx_end > padLeft + scaledW) dx_end = padLeft + scaledW;
            for (int dy = dy_base; dy < dy_end; dy++) {
                uint8_t* dstRow = dst.data() + dy * dstW * 3;
                for (int dx = dx_base; dx < dx_end; dx++) {
                    dstRow[dx*3]   = rb;
                    dstRow[dx*3+1] = gb;
                    dstRow[dx*3+2] = bb;
                }
            }
        }
    }
    env->ReleaseByteArrayElements(yArr, yData, JNI_ABORT);
    env->ReleaseByteArrayElements(uArr, uData, JNI_ABORT);
    env->ReleaseByteArrayElements(vArr, vData, JNI_ABORT);
    long tConvert = nowMs();

    // 推理
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0; inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = dstW * dstH * 3; inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = dst.data();
    if (rknn_inputs_set(ctx, 1, inputs) != RKNN_SUCC)
        return env->NewStringUTF("{\"error\":\"inputs_set failed\"}");
    if (rknn_run(ctx, nullptr) != RKNN_SUCC)
        return env->NewStringUTF("{\"error\":\"run failed\"}");

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output < 4)
        return env->NewStringUTF("{\"error\":\"unexpected n_output\"}");

    std::vector<rknn_output> outputs(io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output)*io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) outputs[i].want_float = 1;
    if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr) != RKNN_SUCC)
        return env->NewStringUTF("{\"error\":\"outputs_get failed\"}");
    long tInfer = nowMs();

    const float* outs[4];
    int sizes[4];
    for (int i = 0; i < 4; i++) {
        outs[i] = (const float*)outputs[i].buf;
        sizes[i] = outputs[i].size / sizeof(float);
    }
    std::string json = poseDecodeJson(outs, sizes, srcW, srcH, padTop, padLeft, scale, 0.25f, dstW);
    long tDecode = nowMs();
    static int frameCount = 0;
    if (++frameCount % 30 == 0) {
        LOGI("PoseYuv[30f]: convert=%ldms infer=%ldms decode=%ldms total=%ldms",
             tConvert-t0, tInfer-tConvert, tDecode-tInfer, tDecode-t0);
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    return env->NewStringUTF(json.c_str());
}

// HandInference: YUV → resize → infer
JNIEXPORT jfloatArray JNICALL
Java_com_example_weblauncher_npu_HandInference_nativeInferYuv(
        JNIEnv *env, jobject, jlong handle,
        jbyteArray yArr, jbyteArray uArr, jbyteArray vArr,
        jint srcW, jint srcH,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint dstW, jint dstH) {

    rknn_context ctx = (rknn_context)handle;
    if (ctx == 0) return nullptr;

    jbyte* yData = env->GetByteArrayElements(yArr, nullptr);
    jbyte* uData = env->GetByteArrayElements(uArr, nullptr);
    jbyte* vData = env->GetByteArrayElements(vArr, nullptr);

    std::vector<uint8_t> dst(dstW * dstH * 3);
    int sx_fp = (srcW << 16) / dstW;
    int sy_fp = (srcH << 16) / dstH;

    for (int dy = 0; dy < dstH; dy++) {
        int srcY = (dy * sy_fp) >> 16; if (srcY >= srcH) srcY = srcH - 1;
        int uvRow = srcY >> 1;
        const uint8_t* yRow = (const uint8_t*)yData + srcY * yRowStride;
        const uint8_t* uRow = (const uint8_t*)uData + uvRow * uvRowStride;
        const uint8_t* vRow = (const uint8_t*)vData + uvRow * uvRowStride;
        uint8_t* dstRow = dst.data() + dy * dstW * 3;
        for (int dx = 0; dx < dstW; dx++) {
            int srcX = (dx * sx_fp) >> 16; if (srcX >= srcW) srcX = srcW - 1;
            int uvOff = (srcX >> 1) * uvPixelStride;
            int Y = yRow[srcX];
            int U = uRow[uvOff] - 128;
            int V = vRow[uvOff] - 128;
            int r = Y + ((11277 * V) >> 13);
            int g = Y - ((2765  * U + 5726 * V) >> 13);
            int b = Y + ((14216 * U) >> 13);
            dstRow[dx*3]   = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            dstRow[dx*3+1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            dstRow[dx*3+2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }

    env->ReleaseByteArrayElements(yArr, yData, JNI_ABORT);
    env->ReleaseByteArrayElements(uArr, uData, JNI_ABORT);
    env->ReleaseByteArrayElements(vArr, vData, JNI_ABORT);

    return doInfer(env, ctx, dst.data(), dstW * dstH * 3);
}

} // extern "C"
