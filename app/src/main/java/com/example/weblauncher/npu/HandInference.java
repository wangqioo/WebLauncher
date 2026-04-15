package com.example.weblauncher.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.util.Base64;
import android.util.Log;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * 手部检测 + 21关键点推理
 * 两阶段流程:
 *   1. hand_detection.rknn  → 找到手的位置 (boxes + scores, NMS 在此做)
 *   2. hand_landmark.rknn   → 对裁剪后的手部区域提取 21 个关键点
 */
public class HandInference {

    private static final String TAG = "HandInference";
    private static final String DETECT_MODEL  = "hand_detection.rknn";
    private static final String LANDMARK_MODEL = "hand_landmark.rknn";

    private static final int DETECT_W = 640;
    private static final int DETECT_H = 480;
    private static final int LANDMARK_SIZE = 224;

    private final Context context;
    private long detectHandle   = 0;
    private long landmarkHandle = 0;
    private boolean initialized = false;

    static {
        try {
            System.loadLibrary("rknn_jni");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load librknn_jni.so: " + e.getMessage());
        }
    }

    public HandInference(Context context) {
        this.context = context;
    }

    public boolean init() {
        try {
            detectHandle   = loadModel(DETECT_MODEL);
            landmarkHandle = loadModel(LANDMARK_MODEL);
            initialized = (detectHandle != 0 && landmarkHandle != 0);
            Log.i(TAG, "HandInference init: " + (initialized ? "OK" : "FAILED"));
            return initialized;
        } catch (Exception e) {
            Log.e(TAG, "Init failed: " + e.getMessage());
            return false;
        }
    }

    private long loadModel(String filename) throws Exception {
        InputStream is = context.getAssets().open(filename);
        byte[] data = new byte[is.available()];
        is.read(data);
        is.close();
        long handle = nativeInit(data);
        Log.i(TAG, filename + " handle=" + handle);
        return handle;
    }

    public String infer(String base64Image) {
        if (!initialized) return "{\"error\":\"HandInference not initialized\"}";
        try {
            byte[] imageBytes = Base64.decode(
                base64Image.replaceAll("^data:image/[^;]+;base64,", ""), Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            if (bitmap == null) return "{\"error\":\"Invalid image\"}";

            int origW = bitmap.getWidth();
            int origH = bitmap.getHeight();

            // ---- Stage 1: 手部检测 ----
            Bitmap detectInput = Bitmap.createScaledBitmap(bitmap, DETECT_W, DETECT_H, true);
            byte[] detectRgb = bitmapToRgb(detectInput, DETECT_W, DETECT_H);
            float[] detectOut = nativeInfer(detectHandle, detectRgb);
            if (detectOut == null) return "{\"detected\":false,\"error\":\"detection failed\"}";

            // 解析 boxes [N,4] 和 scores [N,1]，做 NMS
            // 输出格式: boxes shape [1,N,4] (y1x1y2x2), scores shape [1,1,N]
            RectF bestBox = parseBestBox(detectOut, DETECT_W, DETECT_H);
            if (bestBox == null) return "{\"detected\":false}";

            // 裁剪手部区域（加 10% padding）
            float padX = (bestBox.right - bestBox.left) * 0.1f;
            float padY = (bestBox.bottom - bestBox.top) * 0.1f;
            int cropX = Math.max(0, (int)((bestBox.left   - padX) / DETECT_W * origW));
            int cropY = Math.max(0, (int)((bestBox.top    - padY) / DETECT_H * origH));
            int cropW = Math.min(origW - cropX, (int)((bestBox.right  - bestBox.left + padX*2) / DETECT_W * origW));
            int cropH = Math.min(origH - cropY, (int)((bestBox.bottom - bestBox.top  + padY*2) / DETECT_H * origH));
            if (cropW <= 0 || cropH <= 0) return "{\"detected\":false}";

            Bitmap handCrop = Bitmap.createBitmap(bitmap, cropX, cropY, cropW, cropH);

            // ---- Stage 2: 关键点推理 ----
            Bitmap lmInput = Bitmap.createScaledBitmap(handCrop, LANDMARK_SIZE, LANDMARK_SIZE, true);
            byte[] lmRgb = bitmapToRgb(lmInput, LANDMARK_SIZE, LANDMARK_SIZE);
            float[] lmOut = nativeInfer(landmarkHandle, lmRgb);
            if (lmOut == null) return "{\"detected\":false,\"error\":\"landmark failed\"}";

            return buildResult(lmOut, bestBox, cropX, cropY, cropW, cropH, origW, origH);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    private byte[] bitmapToRgb(Bitmap bitmap, int w, int h) {
        int[] pixels = new int[w * h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < pixels.length; i++) {
            rgb[i*3]   = (byte)((pixels[i] >> 16) & 0xFF);
            rgb[i*3+1] = (byte)((pixels[i] >> 8)  & 0xFF);
            rgb[i*3+2] = (byte)(pixels[i]          & 0xFF);
        }
        return rgb;
    }

    /**
     * 解析检测输出，返回置信度最高的手部 box
     * boxes: [1, N, 4] (y1,x1,y2,x2 格式) → 展平后长度 N*4
     * scores: [1, 1, N] → 展平后长度 N
     * 两个输出拼接在一起: 先 boxes 再 scores
     */
    private RectF parseBestBox(float[] output, int imgW, int imgH) {
        // 尝试推断 N
        // output 总长 = N*4 + N = N*5
        if (output.length % 5 != 0) return null;
        int N = output.length / 5;

        int boxOffset   = 0;
        int scoreOffset = N * 4;

        float bestScore = 0.3f; // 置信度阈值
        int bestIdx = -1;
        for (int i = 0; i < N; i++) {
            float score = output[scoreOffset + i];
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        if (bestIdx < 0) return null;

        // y1,x1,y2,x2 格式，坐标已归一化 [0,1]
        float y1 = output[boxOffset + bestIdx*4]     * imgH;
        float x1 = output[boxOffset + bestIdx*4 + 1] * imgW;
        float y2 = output[boxOffset + bestIdx*4 + 2] * imgH;
        float x2 = output[boxOffset + bestIdx*4 + 3] * imgW;
        return new RectF(x1, y1, x2, y2);
    }

    private static final String[] KP_NAMES = {
        "wrist",
        "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
        "index_mcp","index_pip","index_dip","index_tip",
        "middle_mcp","middle_pip","middle_dip","middle_tip",
        "ring_mcp","ring_pip","ring_dip","ring_tip",
        "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"
    };

    /**
     * 构建最终 JSON
     * landmark 输出: [63] = 21 * 3 (x, y, z)，坐标归一化到 224x224
     * 还原到原始图像坐标
     */
    private String buildResult(float[] lmOut,
            RectF box, int cropX, int cropY, int cropW, int cropH,
            int origW, int origH) {

        StringBuilder sb = new StringBuilder();
        sb.append("{\"detected\":true,\"keypoints\":[");

        for (int k = 0; k < 21; k++) {
            float lx = lmOut[k*3]   / LANDMARK_SIZE; // 0~1 within crop
            float ly = lmOut[k*3+1] / LANDMARK_SIZE;

            // 还原到原图坐标（归一化）
            float x = (cropX + lx * cropW) / origW;
            float y = (cropY + ly * cropH) / origH;

            if (k > 0) sb.append(",");
            sb.append(String.format("{\"name\":\"%s\",\"x\":%.4f,\"y\":%.4f}",
                KP_NAMES[k], x, y));
        }

        // 手腕坐标也输出检测框信息
        float boxX = box.left / DETECT_W;
        float boxY = box.top  / DETECT_H;
        float boxW = (box.right - box.left) / DETECT_W;
        float boxH = (box.bottom - box.top) / DETECT_H;

        sb.append(String.format("],\"box\":{\"x\":%.4f,\"y\":%.4f,\"w\":%.4f,\"h\":%.4f}}",
            boxX, boxY, boxW, boxH));
        return sb.toString();
    }

    public void release() {
        if (detectHandle   != 0) { nativeRelease(detectHandle);   detectHandle   = 0; }
        if (landmarkHandle != 0) { nativeRelease(landmarkHandle); landmarkHandle = 0; }
        initialized = false;
    }

    private native long    nativeInit(byte[] modelData);
    private native float[] nativeInfer(long handle, byte[] rgbData);
    private native void    nativeRelease(long handle);
}
