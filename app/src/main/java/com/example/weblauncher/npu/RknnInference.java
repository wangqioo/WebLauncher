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
    private static final String MODEL_FILE = "hand_gesture.rknn";

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

            // 缩放到模型输入尺寸 256x256
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
            int[] pixels = new int[256 * 256];
            scaled.getPixels(pixels, 0, 256, 0, 0, 256, 256);

            // 转 RGB byte 数组
            byte[] rgbData = new byte[256 * 256 * 3];
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

    private String buildResult(float[] outputs) {
        // 手势分类标签
        String[] labels = {"none", "palm", "fist", "thumb_up", "thumb_down",
                           "peace", "ok", "point", "rock", "call"};

        // 找最高置信度
        int maxIdx = 0;
        float maxVal = outputs[0];
        for (int i = 1; i < Math.min(outputs.length, labels.length); i++) {
            if (outputs[i] > maxVal) {
                maxVal = outputs[i];
                maxIdx = i;
            }
        }

        String gesture = maxIdx < labels.length ? labels[maxIdx] : "unknown";
        return String.format(
            "{\"gesture\":\"%s\",\"confidence\":%.4f,\"scores\":%s}",
            gesture, maxVal, floatArrayToJson(outputs)
        );
    }

    private String floatArrayToJson(float[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format("%.4f", arr[i]));
        }
        sb.append("]");
        return sb.toString();
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
