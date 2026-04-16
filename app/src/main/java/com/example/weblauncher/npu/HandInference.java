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

    /** 直接接收 RGB 字节（Camera2 采集），纯字节操作，不经过 Bitmap */
    public String inferRgb(byte[] rgbData, int origW, int origH) {
        if (!initialized) return "{\"error\":\"HandInference not initialized\"}";
        try {
            // Stage 1: resize 在 C++ 层完成
            float[] detectOut = nativeInferResized(detectHandle, rgbData, origW, origH, DETECT_W, DETECT_H);
            if (detectOut == null) return "{\"detected\":false,\"error\":\"detection failed\"}";

            int nOut = (int) detectOut[0];
            StringBuilder dbg = new StringBuilder("detect nOut=" + nOut);
            for (int i = 0; i < nOut; i++) dbg.append(" size").append(i).append("=").append((int)detectOut[1+i]);
            Log.i(TAG, dbg.toString());

            RectF bestBox = parseBestBox(detectOut, DETECT_W, DETECT_H);
            if (bestBox == null) return "{\"detected\":false}";

            // Stage 2: 裁剪手部区域（字节级），缩放到 LANDMARK_SIZE
            float padX = (bestBox.right - bestBox.left) * 0.1f;
            float padY = (bestBox.bottom - bestBox.top) * 0.1f;
            int cropX = Math.max(0, (int)((bestBox.left   - padX) / DETECT_W * origW));
            int cropY = Math.max(0, (int)((bestBox.top    - padY) / DETECT_H * origH));
            int cropW = Math.min(origW - cropX, (int)((bestBox.right  - bestBox.left + padX*2) / DETECT_W * origW));
            int cropH = Math.min(origH - cropY, (int)((bestBox.bottom - bestBox.top  + padY*2) / DETECT_H * origH));
            if (cropW <= 0 || cropH <= 0) return "{\"detected\":false}";

            byte[] cropRgb = cropRgb(rgbData, origW, cropX, cropY, cropW, cropH);
            float[] lmOut = nativeInferResized(landmarkHandle, cropRgb, cropW, cropH, LANDMARK_SIZE, LANDMARK_SIZE);
            if (lmOut == null) return "{\"detected\":false,\"error\":\"landmark failed\"}";

            return buildResult(lmOut, bestBox, cropX, cropY, cropW, cropH, origW, origH);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    /** 纯字节 nearest-neighbor resize */
    private byte[] resizeRgb(byte[] src, int srcW, int srcH, int dstW, int dstH) {
        byte[] dst = new byte[dstW * dstH * 3];
        float sx = (float) srcW / dstW;
        float sy = (float) srcH / dstH;
        for (int dy = 0; dy < dstH; dy++) {
            int srcy = Math.min(srcH - 1, (int)(dy * sy));
            for (int dx = 0; dx < dstW; dx++) {
                int srcx = Math.min(srcW - 1, (int)(dx * sx));
                int si = (srcy * srcW + srcx) * 3;
                int di = (dy * dstW + dx) * 3;
                dst[di]   = src[si];
                dst[di+1] = src[si+1];
                dst[di+2] = src[si+2];
            }
        }
        return dst;
    }

    /** 从 RGB 字节数组中裁剪区域 */
    private byte[] cropRgb(byte[] src, int srcW, int x, int y, int w, int h) {
        byte[] dst = new byte[w * h * 3];
        for (int row = 0; row < h; row++) {
            int si = ((y + row) * srcW + x) * 3;
            int di = row * w * 3;
            System.arraycopy(src, si, dst, di, w * 3);
        }
        return dst;
    }

    public String infer(String base64Image) {
        if (!initialized) return "{\"error\":\"HandInference not initialized\"}";
        try {
            byte[] imageBytes = Base64.decode(
                base64Image.replaceAll("^data:image/[^;]+;base64,", ""), Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            if (bitmap == null) return "{\"error\":\"Invalid image\"}";
            return inferBitmap(bitmap, bitmap.getWidth(), bitmap.getHeight());
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    private String inferBitmap(Bitmap bitmap, int origW, int origH) {
        try {

            // ---- Stage 1: 手部检测 ----
            Bitmap detectInput = Bitmap.createScaledBitmap(bitmap, DETECT_W, DETECT_H, true);
            byte[] detectRgb = bitmapToRgb(detectInput, DETECT_W, DETECT_H);
            float[] detectOut = nativeInfer(detectHandle, detectRgb);
            if (detectOut == null) return "{\"detected\":false,\"error\":\"detection failed\"}";

            // 打印输出结构供调试
            int nOut = (int) detectOut[0];
            StringBuilder dbg = new StringBuilder("detect nOut=" + nOut);
            for (int i = 0; i < nOut; i++) dbg.append(" size").append(i).append("=").append((int)detectOut[1+i]);
            Log.i(TAG, dbg.toString());

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
    private RectF parseBestBox(float[] raw, int imgW, int imgH) {
        // JNI header: [nOutput, size0, size1, ..., data0..., data1..., ...]
        int nOutput = (int) raw[0];
        int headerSize = 1 + nOutput;
        int[] sizes = new int[nOutput];
        for (int i = 0; i < nOutput; i++) sizes[i] = (int) raw[1 + i];

        // 打印前几个值帮助理解格式
        StringBuilder sb = new StringBuilder("detect out values: ");
        int dataStart = headerSize;
        for (int i = 0; i < Math.min(10, raw.length - dataStart); i++) {
            sb.append(String.format("%.4f ", raw[dataStart + i]));
        }
        Log.i(TAG, sb.toString());
        Log.i(TAG, "total raw len=" + raw.length + " nOutput=" + nOutput + " sizes[0]=" + sizes[0]);

        // Gold-YOLO 输出: 通常 2 个输出
        // out0: boxes [N, 4] 归一化坐标
        // out1: scores [N, num_classes]
        // 或者单输出 [N, 5] (x1,y1,x2,y2,score)
        if (nOutput == 1) {
            // 单输出 [N, 5]: x1y1x2y2 + score
            int N = sizes[0] / 5;
            float bestScore = -1f;
            float topScore = -1f;
            int bestIdx = -1;
            for (int i = 0; i < N; i++) {
                float score = raw[dataStart + i * 5 + 4];
                if (score > topScore) topScore = score;
                if (score > 0.1f && score > bestScore) { bestScore = score; bestIdx = i; }
            }
            Log.i(TAG, String.format("nOutput=1 N=%d topScore=%.4f bestScore=%.4f bestIdx=%d", N, topScore, bestScore, bestIdx));
            if (bestIdx < 0) return null;
            float x1 = raw[dataStart + bestIdx*5]     * imgW;
            float y1 = raw[dataStart + bestIdx*5 + 1] * imgH;
            float x2 = raw[dataStart + bestIdx*5 + 2] * imgW;
            float y2 = raw[dataStart + bestIdx*5 + 3] * imgH;
            Log.i(TAG, String.format("Best box: (%.1f,%.1f)-(%.1f,%.1f) score=%.3f", x1,y1,x2,y2,bestScore));
            return new RectF(x1, y1, x2, y2);
        } else if (nOutput >= 2) {
            // out0=boxes[N,4] 绝对像素坐标(相对于检测输入尺寸 DETECT_W x DETECT_H)
            // out1=scores[N] raw logit，需要 sigmoid
            int dataStart1 = dataStart + sizes[0];
            int N = sizes[0] / 4;
            float bestScore = -1f;
            float topScore = -1f;
            int bestIdx = -1;
            for (int i = 0; i < N && i < sizes[1]; i++) {
                float score = sigmoid(raw[dataStart1 + i]);
                if (score > topScore) topScore = score;
                if (score > 0.3f && score > bestScore) { bestScore = score; bestIdx = i; }
            }
            Log.i(TAG, String.format("nOutput=%d N=%d topScore=%.4f bestScore=%.4f bestIdx=%d", nOutput, N, topScore, bestScore, bestIdx));
            if (bestIdx < 0) return null;
            // boxes: 绝对像素坐标，直接使用，不乘以 imgW/imgH
            float x1 = raw[dataStart + bestIdx*4];
            float y1 = raw[dataStart + bestIdx*4 + 1];
            float x2 = raw[dataStart + bestIdx*4 + 2];
            float y2 = raw[dataStart + bestIdx*4 + 3];
            Log.i(TAG, String.format("Best box: (%.1f,%.1f)-(%.1f,%.1f) score=%.3f", x1,y1,x2,y2,bestScore));
            return new RectF(x1, y1, x2, y2);
        }
        return null;
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
    private String buildResult(float[] lmRaw,
            RectF box, int cropX, int cropY, int cropW, int cropH,
            int origW, int origH) {

        // 解析 JNI header: [nOutput, size0, ..., data0..., data1..., ...]
        int nLmOutput = (int) lmRaw[0];
        int lmHeaderSize = 1 + nLmOutput;
        int lmDataStart = lmHeaderSize;

        // Log landmark output info
        StringBuilder lmDbg = new StringBuilder("landmark nOut=" + nLmOutput);
        for (int i = 0; i < nLmOutput; i++) lmDbg.append(" size").append(i).append("=").append((int)lmRaw[1+i]);
        Log.i(TAG, lmDbg.toString());

        // landmark 数据从 lmDataStart 开始
        // 期望: 21 * 3 = 63 个 float (x, y, z)，单位为 LANDMARK_SIZE 像素
        float[] lmOut = new float[lmRaw.length - lmDataStart];
        System.arraycopy(lmRaw, lmDataStart, lmOut, 0, lmOut.length);

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

    private float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    public void release() {
        if (detectHandle   != 0) { nativeRelease(detectHandle);   detectHandle   = 0; }
        if (landmarkHandle != 0) { nativeRelease(landmarkHandle); landmarkHandle = 0; }
        initialized = false;
    }

    /** YUV_420_888 直接推理（检测阶段），跳过 Java YUV→RGB */
    public String inferYuv(byte[] yData, byte[] uData, byte[] vData,
                           int srcW, int srcH,
                           int yRowStride, int uvRowStride, int uvPixelStride) {
        if (!initialized) return "{\"error\":\"HandInference not initialized\"}";
        try {
            // Stage 1: YUV → resize → detection
            float[] detectOut = nativeInferYuv(detectHandle, yData, uData, vData,
                srcW, srcH, yRowStride, uvRowStride, uvPixelStride, DETECT_W, DETECT_H);
            if (detectOut == null) return "{\"detected\":false,\"error\":\"detection failed\"}";

            int nOut = (int) detectOut[0];
            StringBuilder dbg = new StringBuilder("detect nOut=" + nOut);
            for (int i = 0; i < nOut; i++) dbg.append(" size").append(i).append("=").append((int)detectOut[1+i]);
            Log.i(TAG, dbg.toString());

            RectF bestBox = parseBestBox(detectOut, DETECT_W, DETECT_H);
            if (bestBox == null) return "{\"detected\":false}";

            // Stage 2: 需要 RGB 做裁剪，用 C++ YUV→RGB 转换
            byte[] rgb = nativeYuvToRgb(yData, uData, vData, srcW, srcH,
                yRowStride, uvRowStride, uvPixelStride, srcW, srcH);

            float padX = (bestBox.right - bestBox.left) * 0.1f;
            float padY = (bestBox.bottom - bestBox.top) * 0.1f;
            int cropX = Math.max(0, (int)((bestBox.left   - padX) / DETECT_W * srcW));
            int cropY = Math.max(0, (int)((bestBox.top    - padY) / DETECT_H * srcH));
            int cropW = Math.min(srcW - cropX, (int)((bestBox.right  - bestBox.left + padX*2) / DETECT_W * srcW));
            int cropH = Math.min(srcH - cropY, (int)((bestBox.bottom - bestBox.top  + padY*2) / DETECT_H * srcH));
            if (cropW <= 0 || cropH <= 0) return "{\"detected\":false}";

            byte[] cropRgb = cropRgb(rgb, srcW, cropX, cropY, cropW, cropH);
            float[] lmOut = nativeInferResized(landmarkHandle, cropRgb, cropW, cropH, LANDMARK_SIZE, LANDMARK_SIZE);
            if (lmOut == null) return "{\"detected\":false,\"error\":\"landmark failed\"}";

            return buildResult(lmOut, bestBox, cropX, cropY, cropW, cropH, srcW, srcH);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    private native long    nativeInit(byte[] modelData);
    private native float[] nativeInfer(long handle, byte[] rgbData);
    private native float[] nativeInferResized(long handle, byte[] rgbData, int srcW, int srcH, int dstW, int dstH);
    private native float[] nativeInferYuv(long handle,
        byte[] yData, byte[] uData, byte[] vData,
        int srcW, int srcH, int yRowStride, int uvRowStride, int uvPixelStride,
        int dstW, int dstH);
    private native byte[]  nativeYuvToRgb(
        byte[] yData, byte[] uData, byte[] vData,
        int srcW, int srcH, int yRowStride, int uvRowStride, int uvPixelStride,
        int dstW, int dstH);
    private native void    nativeRelease(long handle);
}
