package com.example.weblauncher.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;
import java.io.InputStream;

/**
 * RKNN NPU 推理封装
 * 通过 JNI 调用 librknnrt.so
 */
public class RknnInference {

    private static final String TAG = "RknnInference";
    private static final String MODEL_FILE = "yolov8n-pose.rknn";
    private static final int INPUT_SIZE = 640;

    private final Context context;
    private long rknnHandle = 0;
    private boolean initialized = false;

    static {
        try {
            System.loadLibrary("rknn_jni");
            Log.i(TAG, "librknn_jni.so loaded");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load librknn_jni.so: " + e.getMessage());
        }
    }

    public RknnInference(Context context) {
        this.context = context;
    }

    public boolean init() {
        try {
            // 从 assets 读取模型
            InputStream is = context.getAssets().open(MODEL_FILE);
            byte[] modelData = new byte[is.available()];
            is.read(modelData);
            is.close();

            rknnHandle = nativeInit(modelData);
            initialized = (rknnHandle != 0);
            Log.i(TAG, "Model loaded, handle=" + rknnHandle);
            return initialized;
        } catch (Exception e) {
            Log.e(TAG, "Init failed: " + e.getMessage());
            return false;
        }
    }

    /**
     * 推理入口：输入 base64 图像，返回 JSON 结果
     */
    public String infer(String base64Image) {
        if (!initialized) {
            return "{\"error\":\"RKNN not initialized\"}";
        }
        try {
            // base64 → Bitmap
            byte[] imageBytes = Base64.decode(
                base64Image.replaceAll("^data:image/[^;]+;base64,", ""),
                Base64.DEFAULT
            );
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            if (bitmap == null) return "{\"error\":\"Invalid image\"}";

            // 缩放到模型输入尺寸 640x640
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
            int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
            scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

            // 转 RGB byte 数组
            byte[] rgbData = new byte[INPUT_SIZE * INPUT_SIZE * 3];
            for (int i = 0; i < pixels.length; i++) {
                rgbData[i * 3]     = (byte) ((pixels[i] >> 16) & 0xFF); // R
                rgbData[i * 3 + 1] = (byte) ((pixels[i] >> 8)  & 0xFF); // G
                rgbData[i * 3 + 2] = (byte) (pixels[i]         & 0xFF); // B
            }

            // 调用 NPU 推理
            float[] outputs = nativeInfer(rknnHandle, rgbData);
            if (outputs == null) return "{\"error\":\"Inference failed\"}";

            return buildResult(outputs);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    /**
     * 解析 yolov8n-pose 输出
     * 输出格式: [1, 56, 8400] → 每个检测框有 4(box) + 1(conf) + 51(17*3 keypoints) = 56 个值
     * 取置信度最高的一个 person 检测框
     */
    private String buildResult(float[] outputs) {
        // yolov8-pose 输出: shape [56 x 8400], 按列存储 (每个 anchor 56个值)
        // outputs 长度 = 56 * 8400 = 470400
        int numAnchors = 8400;
        int stride = 56; // 4 box + 1 conf + 51 keypoints (17*3)

        if (outputs.length < stride * numAnchors) {
            // 输出长度不匹配，直接返回原始数据摘要
            return String.format("{\"detected\":false,\"outputLen\":%d}", outputs.length);
        }

        // 找置信度最高的 anchor
        float bestConf = 0;
        int bestIdx = -1;
        for (int i = 0; i < numAnchors; i++) {
            float conf = outputs[4 * numAnchors + i]; // conf 在第5行
            if (conf > bestConf) {
                bestConf = conf;
                bestIdx = i;
            }
        }

        if (bestConf < 0.3f || bestIdx < 0) {
            return "{\"detected\":false,\"confidence\":0}";
        }

        // 提取关键点 (17个，每个3个值: x, y, score)
        StringBuilder kpJson = new StringBuilder("[");
        String[] keypointNames = {"nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"};

        for (int k = 0; k < 17; k++) {
            int baseOffset = (5 + k * 3) * numAnchors + bestIdx;
            float kx = outputs[baseOffset];
            float ky = outputs[baseOffset + numAnchors];
            float ks = outputs[baseOffset + 2 * numAnchors];
            if (k > 0) kpJson.append(",");
            kpJson.append(String.format("{\"name\":\"%s\",\"x\":%.3f,\"y\":%.3f,\"score\":%.3f}",
                keypointNames[k], kx / INPUT_SIZE, ky / INPUT_SIZE, ks));
        }
        kpJson.append("]");

        return String.format("{\"detected\":true,\"confidence\":%.4f,\"keypoints\":%s}",
            bestConf, kpJson.toString());
    }

    public void release() {
        if (rknnHandle != 0) {
            nativeRelease(rknnHandle);
            rknnHandle = 0;
            initialized = false;
        }
    }

    // JNI 方法声明
    private native long nativeInit(byte[] modelData);
    private native float[] nativeInfer(long handle, byte[] rgbData);
    private native void nativeRelease(long handle);
}
