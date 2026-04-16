package com.example.weblauncher.npu;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.hardware.camera2.*;
import java.io.ByteArrayOutputStream;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

/**
 * Camera2 持续采帧 → NPU 推理 → 结果广播
 * 使用 YUV_420_888，直接在 Android 端做 YUV→RGB，省去 base64 传输
 */
public class CameraInferenceManager {

    private static final String TAG = "CameraInference";
    private static final int CAPTURE_W = 640;
    private static final int CAPTURE_H = 480;

    private final Context context;
    private final RknnInference poseInfer;
    private final HandInference handInfer;
    private final NpuWebSocketServer wsServer;

    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private ImageReader imageReader;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private HandlerThread inferThread;
    private Handler inferHandler;

    private volatile boolean inferBusy = false;
    private volatile String mode = "pose"; // "pose" | "hand" | "both"

    public CameraInferenceManager(Context context,
                                   RknnInference poseInfer,
                                   HandInference handInfer,
                                   NpuWebSocketServer wsServer) {
        this.context   = context;
        this.poseInfer = poseInfer;
        this.handInfer = handInfer;
        this.wsServer  = wsServer;
    }

    public void setMode(String mode) { this.mode = mode; }

    @SuppressLint("MissingPermission")
    public void start() {
        cameraThread = new HandlerThread("CameraThread");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());

        inferThread = new HandlerThread("InferThread");
        inferThread.start();
        inferHandler = new Handler(inferThread.getLooper());

        imageReader = ImageReader.newInstance(CAPTURE_W, CAPTURE_H,
                ImageFormat.YUV_420_888, 2);
        imageReader.setOnImageAvailableListener(this::onImageAvailable, cameraHandler);

        CameraManager cm = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        try {
            String cameraId = findBackCamera(cm);
            cm.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    startPreview();
                }
                @Override
                public void onDisconnected(CameraDevice camera) { camera.close(); }
                @Override
                public void onError(CameraDevice camera, int error) {
                    Log.e(TAG, "Camera error: " + error);
                    camera.close();
                }
            }, cameraHandler);
        } catch (Exception e) {
            Log.e(TAG, "openCamera failed: " + e.getMessage());
        }
    }

    private String findBackCamera(CameraManager cm) throws CameraAccessException {
        for (String id : cm.getCameraIdList()) {
            CameraCharacteristics c = cm.getCameraCharacteristics(id);
            Integer facing = c.get(CameraCharacteristics.LENS_FACING);
            if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) return id;
        }
        return cm.getCameraIdList()[0];
    }

    private void startPreview() {
        try {
            CaptureRequest.Builder builder = cameraDevice.createCaptureRequest(
                    CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(imageReader.getSurface());
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);

            cameraDevice.createCaptureSession(
                java.util.Collections.singletonList(imageReader.getSurface()),
                new CameraCaptureSession.StateCallback() {
                    @Override
                    public void onConfigured(CameraCaptureSession session) {
                        captureSession = session;
                        try {
                            session.setRepeatingRequest(builder.build(), null, cameraHandler);
                            Log.i(TAG, "Camera preview started " + CAPTURE_W + "x" + CAPTURE_H);
                        } catch (CameraAccessException e) {
                            Log.e(TAG, "setRepeatingRequest: " + e.getMessage());
                        }
                    }
                    @Override
                    public void onConfigureFailed(CameraCaptureSession session) {
                        Log.e(TAG, "Camera configure failed");
                    }
                }, cameraHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "startPreview: " + e.getMessage());
        }
    }

    private void onImageAvailable(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null) return;
        if (inferBusy) { image.close(); return; }

        inferBusy = true;

        // 在 camera 线程提取 YUV plane 数据，然后关闭 Image
        Image.Plane[] planes = image.getPlanes();
        java.nio.ByteBuffer yBuf  = planes[0].getBuffer();
        java.nio.ByteBuffer uBuf  = planes[1].getBuffer();
        java.nio.ByteBuffer vBuf  = planes[2].getBuffer();
        final byte[] yData = new byte[yBuf.remaining()];
        final byte[] uData = new byte[uBuf.remaining()];
        final byte[] vData = new byte[vBuf.remaining()];
        yBuf.get(yData);
        uBuf.get(uData);
        vBuf.get(vData);
        final int yRowStride   = planes[0].getRowStride();
        final int uvRowStride  = planes[1].getRowStride();
        final int uvPixelStride = planes[1].getPixelStride();
        image.close();

        inferHandler.post(() -> {
            try {
                runInferenceYuv(yData, uData, vData, yRowStride, uvRowStride, uvPixelStride);
            } catch (Exception e) {
                Log.e(TAG, "infer error: " + e.getMessage());
            } finally {
                inferBusy = false;
            }
        });
    }

    private void runInferenceYuv(byte[] yData, byte[] uData, byte[] vData,
                                  int yRowStride, int uvRowStride, int uvPixelStride) {
        long t0 = System.currentTimeMillis();
        String result;
        String currentMode = mode;

        if ("pose".equals(currentMode)) {
            result = poseInfer.inferYuv(yData, uData, vData,
                CAPTURE_W, CAPTURE_H, yRowStride, uvRowStride, uvPixelStride);
        } else if ("hand".equals(currentMode)) {
            result = handInfer.inferYuv(yData, uData, vData,
                CAPTURE_W, CAPTURE_H, yRowStride, uvRowStride, uvPixelStride);
        } else {
            String pose = poseInfer.inferYuv(yData, uData, vData,
                CAPTURE_W, CAPTURE_H, yRowStride, uvRowStride, uvPixelStride);
            String hand = handInfer.inferYuv(yData, uData, vData,
                CAPTURE_W, CAPTURE_H, yRowStride, uvRowStride, uvPixelStride);
            result = "{\"pose\":" + pose + ",\"hand\":" + hand + "}";
        }

        long msInfer = System.currentTimeMillis() - t0;
        if (result.endsWith("}")) {
            result = result.substring(0, result.length()-1) + ",\"inferMs\":" + msInfer + "}";
        }

        long t1 = System.currentTimeMillis();
        // 用 C++ 做 YUV→RGB，缩小到 JPEG_W×JPEG_H 再编码
        byte[] smallRgb = nativeYuvToRgb(yData, uData, vData,
            CAPTURE_W, CAPTURE_H, yRowStride, uvRowStride, uvPixelStride, JPEG_W, JPEG_H);
        byte[] jpeg = rgbToJpeg(smallRgb, JPEG_W, JPEG_H);
        long msJpeg = System.currentTimeMillis() - t1;

        Log.i(TAG, "infer=" + msInfer + "ms jpeg=" + msJpeg + "ms total=" + (msInfer+msJpeg) + "ms");
        wsServer.broadcastFrame(result, jpeg);
    }

    private static final int JPEG_W = 160;
    private static final int JPEG_H = 120;

    /** RGB byte[] (w×h) → JPEG，输入已是目标尺寸，无需再 resize */
    private byte[] rgbToJpeg(byte[] rgb, int w, int h) {
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(w * h * 4);
        for (int i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
            buf.put(j,   rgb[i]);
            buf.put(j+1, rgb[i+1]);
            buf.put(j+2, rgb[i+2]);
            buf.put(j+3, (byte) 0xFF);
        }
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        buf.rewind();
        bmp.copyPixelsFromBuffer(buf);
        ByteArrayOutputStream out = new ByteArrayOutputStream(w * h / 3);
        bmp.compress(Bitmap.CompressFormat.JPEG, 65, out);
        bmp.recycle();
        return out.toByteArray();
    }

    /** YUV_420_888 → RGB，在 C++ 完成，同时 resize 到 dstW×dstH */
    private native byte[] nativeYuvToRgb(
        byte[] yData, byte[] uData, byte[] vData,
        int srcW, int srcH, int yRowStride, int uvRowStride, int uvPixelStride,
        int dstW, int dstH);

    static {
        try {
            System.loadLibrary("rknn_jni");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load librknn_jni.so: " + e.getMessage());
        }
    }

    public void stop() {
        try {
            if (captureSession != null) { captureSession.close(); captureSession = null; }
            if (cameraDevice  != null) { cameraDevice.close();   cameraDevice   = null; }
            if (imageReader   != null) { imageReader.close();    imageReader    = null; }
        } catch (Exception e) {
            Log.e(TAG, "stop: " + e.getMessage());
        }
        if (cameraThread != null) cameraThread.quitSafely();
        if (inferThread  != null) inferThread.quitSafely();
    }
}
