package com.example.weblauncher.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;
import java.io.InputStream;

public class RknnInference {

    private static final String TAG = "RknnInference";
    private static final String MODEL_FILE = "yolov8n-pose.rknn";
    private static final int INPUT_SIZE = 640;
    private static final float OBJ_THRESH = 0.25f;
    private static final float NMS_THRESH  = 0.45f;

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

    public RknnInference(Context context) { this.context = context; }

    public boolean init() {
        try {
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

    /** 直接接收 RGB 字节（Camera2 采集），resize 在 C++ 层完成 */
    public String inferRgb(byte[] rgbData, int origW, int origH) {
        if (!initialized) return "{\"error\":\"RKNN not initialized\"}";
        try {
            // nativeInferResized 返回: [padTop, padLeft, scale, nOutput, size0..., data0...]
            float[] raw = nativeInferResized(rknnHandle, rgbData, origW, origH, INPUT_SIZE, INPUT_SIZE);
            if (raw == null) return "{\"error\":\"Inference failed\"}";
            int padTop  = (int) raw[0];
            int padLeft = (int) raw[1];
            float scale = raw[2];
            int[] padInfo = new int[]{padTop, INPUT_SIZE - (int)(origH*scale) - padTop,
                                      padLeft, INPUT_SIZE - (int)(origW*scale) - padLeft};
            // 去掉前 3 个 meta float，传给 buildResult
            float[] outputs = new float[raw.length - 3];
            System.arraycopy(raw, 3, outputs, 0, outputs.length);
            return buildResult(outputs, origW, origH, padInfo);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    /**
     * 纯字节 letterbox resize：等比缩放 + 灰色填充，不经过 Bitmap
     * padInfo: [padTop, padBottom, padLeft, padRight]
     */
    private byte[] letterboxRgb(byte[] src, int srcW, int srcH,
                                  int dstW, int dstH, int[] padInfo) {
        float scale = Math.min((float) dstW / srcW, (float) dstH / srcH);
        int scaledW = Math.round(srcW * scale);
        int scaledH = Math.round(srcH * scale);
        int padLeft = (dstW - scaledW) / 2;
        int padTop  = (dstH - scaledH) / 2;
        if (padInfo != null) {
            padInfo[0] = padTop;
            padInfo[1] = dstH - scaledH - padTop;
            padInfo[2] = padLeft;
            padInfo[3] = dstW - scaledW - padLeft;
        }

        byte gray = (byte) 0x38; // 灰色填充 (0x38 = 56)
        byte[] dst = new byte[dstW * dstH * 3];
        // 先填灰色
        for (int i = 0; i < dst.length; i += 3) {
            dst[i] = gray; dst[i+1] = gray; dst[i+2] = gray;
        }

        // 最近邻缩放（比双线性快 ~4x，对 NPU 推理精度影响极小）
        float scaleInv = 1.0f / scale;
        for (int dy = 0; dy < scaledH; dy++) {
            int srcy = Math.min(srcH - 1, (int)(dy * scaleInv));
            for (int dx = 0; dx < scaledW; dx++) {
                int srcx = Math.min(srcW - 1, (int)(dx * scaleInv));
                int si = (srcy * srcW + srcx) * 3;
                int di = ((padTop + dy) * dstW + (padLeft + dx)) * 3;
                dst[di]   = src[si];
                dst[di+1] = src[si+1];
                dst[di+2] = src[si+2];
            }
        }
        return dst;
    }

    public String infer(String base64Image) {
        if (!initialized) return "{\"error\":\"RKNN not initialized\"}";
        try {
            byte[] imageBytes = Base64.decode(
                base64Image.replaceAll("^data:image/[^;]+;base64,", ""), Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            if (bitmap == null) return "{\"error\":\"Invalid image\"}";

            // letterbox resize 到 640x640
            int[] padInfo = new int[4]; // padTop, padBottom, padLeft, padRight
            Bitmap letterboxed = letterboxResize(bitmap, INPUT_SIZE, INPUT_SIZE, padInfo);
            byte[] rgbData = bitmapToRgb(letterboxed, INPUT_SIZE, INPUT_SIZE);

            float[] outputs = nativeInfer(rknnHandle, rgbData);
            if (outputs == null) return "{\"error\":\"Inference failed\"}";

            return buildResult(outputs, bitmap.getWidth(), bitmap.getHeight(), padInfo);
        } catch (Exception e) {
            Log.e(TAG, "Infer error: " + e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    // letterbox: 等比缩放，剩余部分补灰边
    private Bitmap letterboxResize(Bitmap src, int targetW, int targetH, int[] padInfo) {
        float scale = Math.min((float) targetW / src.getWidth(), (float) targetH / src.getHeight());
        int scaledW = Math.round(src.getWidth() * scale);
        int scaledH = Math.round(src.getHeight() * scale);
        int padLeft  = (targetW - scaledW) / 2;
        int padTop   = (targetH - scaledH) / 2;

        if (padInfo != null) {
            padInfo[0] = padTop;
            padInfo[1] = targetH - scaledH - padTop;
            padInfo[2] = padLeft;
            padInfo[3] = targetW - scaledW - padLeft;
        }

        Bitmap scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true);
        Bitmap result = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(result);
        canvas.drawColor(0xFF383838); // 灰色填充
        canvas.drawBitmap(scaled, padLeft, padTop, null);
        return result;
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
     * 解析 yolov8n-pose 的 4 个输出:
     *   outputs[0]: [1, 65, 80, 80]  stride=8
     *   outputs[1]: [1, 65, 40, 40]  stride=16
     *   outputs[2]: [1, 65, 20, 20]  stride=32
     *   outputs[3]: [1, 51, 8400]    keypoints (17*3)
     *
     * JNI 返回格式: [n_output, size0, size1, size2, size3, data0..., data1..., data2..., data3...]
     */
    private String buildResult(float[] raw, int origW, int origH, int[] padInfo) {
        // 解析 header
        int nOutput = (int) raw[0];
        if (nOutput < 4) return "{\"detected\":false,\"error\":\"unexpected n_output=" + nOutput + "\"}";

        int[] sizes = new int[nOutput];
        int dataStart = 1 + nOutput;
        for (int i = 0; i < nOutput; i++) sizes[i] = (int) raw[1 + i];

        // 拆分各输出
        float[][] outs = new float[nOutput][];
        int offset = dataStart;
        for (int i = 0; i < nOutput; i++) {
            outs[i] = new float[sizes[i]];
            System.arraycopy(raw, offset, outs[i], 0, sizes[i]);
            offset += sizes[i];
        }

        // keypoints: [1, 51, 8400] -> [8400, 17, 3]
        float[] kpFlat = outs[3]; // 51*8400 = 428400

        // 处理三个 stride head
        int[][] strideCfg = {{8, 80, 80}, {16, 40, 40}, {32, 20, 20}};
        int[] indexOffsets = {0, 80*80, 80*80 + 40*40};

        float bestConf = OBJ_THRESH;
        int bestAnchorIdx = -1;
        float[] bestBox = null;

        for (int si = 0; si < 3; si++) {
            int stride = strideCfg[si][0];
            int gridH  = strideCfg[si][1];
            int gridW  = strideCfg[si][2];
            int gridSize = gridH * gridW;
            float[] feat = outs[si]; // [65 * gridSize]

            for (int h = 0; h < gridH; h++) {
                for (int w = 0; w < gridW; w++) {
                    int pos = h * gridW + w;
                    // conf 在第 64 通道: feat[64 * gridSize + pos]
                    float conf = sigmoid(feat[64 * gridSize + pos]);
                    if (conf > bestConf) {
                        bestConf = conf;
                        bestAnchorIdx = indexOffsets[si] + pos;

                        // DFL box decode: channels 0..63 -> 4 * 16
                        float[] dfl = new float[64];
                        for (int c = 0; c < 64; c++) dfl[c] = feat[c * gridSize + pos];

                        float x1 = (w + 0.5f) - dflDecode(dfl, 0);
                        float y1 = (h + 0.5f) - dflDecode(dfl, 1);
                        float x2 = (w + 0.5f) + dflDecode(dfl, 2);
                        float y2 = (h + 0.5f) + dflDecode(dfl, 3);

                        bestBox = new float[]{x1 * stride, y1 * stride, x2 * stride, y2 * stride};
                    }
                }
            }
        }

        if (bestAnchorIdx < 0 || bestBox == null) return "{\"detected\":false,\"confidence\":0}";

        // 去掉 letterbox padding，还原到原图坐标
        int padTop  = padInfo[0];
        int padLeft = padInfo[2];
        float scale = Math.min((float) INPUT_SIZE / origW, (float) INPUT_SIZE / origH);

        float nx1 = Math.max(0, (bestBox[0] - padLeft) / scale / origW);
        float ny1 = Math.max(0, (bestBox[1] - padTop)  / scale / origH);
        float nx2 = Math.min(1, (bestBox[2] - padLeft) / scale / origW);
        float ny2 = Math.min(1, (bestBox[3] - padTop)  / scale / origH);

        // keypoints: kpFlat shape [51, 8400], 即 [17*3, 8400]
        // kp[k][j] = kpFlat[k*3+j, anchorIdx] = kpFlat[(k*3+j)*8400 + anchorIdx]
        int totalAnchors = 8400;
        String[] kpNames = {"nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"};

        StringBuilder kpJson = new StringBuilder("[");
        for (int k = 0; k < 17; k++) {
            float kx = kpFlat[(k*3)   * totalAnchors + bestAnchorIdx];
            float ky = kpFlat[(k*3+1) * totalAnchors + bestAnchorIdx];
            float ks = sigmoid(kpFlat[(k*3+2) * totalAnchors + bestAnchorIdx]);

            // 还原关键点坐标
            float nkx = Math.max(0, Math.min(1, (kx - padLeft) / scale / origW));
            float nky = Math.max(0, Math.min(1, (ky - padTop)  / scale / origH));

            if (k > 0) kpJson.append(",");
            kpJson.append(String.format("{\"name\":\"%s\",\"x\":%.4f,\"y\":%.4f,\"score\":%.4f}",
                kpNames[k], nkx, nky, ks));
        }
        kpJson.append("]");

        return String.format(
            "{\"detected\":true,\"confidence\":%.4f,\"box\":{\"x\":%.4f,\"y\":%.4f,\"x2\":%.4f,\"y2\":%.4f},\"keypoints\":%s}",
            bestConf, nx1, ny1, nx2, ny2, kpJson.toString());
    }

    // DFL decode: softmax 后加权求和，得到一个方向的距离
    private float dflDecode(float[] dfl, int group) {
        // group: 0=left, 1=top, 2=right, 3=bottom，每组 16 个 bins
        float[] bins = new float[16];
        for (int i = 0; i < 16; i++) bins[i] = dfl[group * 16 + i];
        float[] sm = softmax(bins);
        float val = 0;
        for (int i = 0; i < 16; i++) val += i * sm[i];
        return val;
    }

    private float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    private float[] softmax(float[] x) {
        float max = x[0];
        for (float v : x) if (v > max) max = v;
        float sum = 0;
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) { out[i] = (float) Math.exp(x[i] - max); sum += out[i]; }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }

    public void release() {
        if (rknnHandle != 0) { nativeRelease(rknnHandle); rknnHandle = 0; initialized = false; }
    }

    /** YUV_420_888 直接推理，C++ 完成 decode，直接返回 JSON */
    public String inferYuv(byte[] yData, byte[] uData, byte[] vData,
                           int srcW, int srcH,
                           int yRowStride, int uvRowStride, int uvPixelStride) {
        if (!initialized) return "{\"error\":\"RKNN not initialized\"}";
        try {
            return nativeInferYuv(rknnHandle, yData, uData, vData,
                srcW, srcH, yRowStride, uvRowStride, uvPixelStride, INPUT_SIZE, INPUT_SIZE);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    private native long    nativeInit(byte[] modelData);
    private native float[] nativeInfer(long handle, byte[] rgbData);
    private native float[] nativeInferResized(long handle, byte[] rgbData, int srcW, int srcH, int dstW, int dstH);
    private native String  nativeInferYuv(long handle,
        byte[] yData, byte[] uData, byte[] vData,
        int srcW, int srcH, int yRowStride, int uvRowStride, int uvPixelStride,
        int dstW, int dstH);
    private native void    nativeRelease(long handle);
}
