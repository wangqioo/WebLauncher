package com.example.weblauncher.npu;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;

/**
 * 后台 NPU 推理服务
 * - HTTP 8080: 兼容旧的 base64 接口
 * - WebSocket 8081: Camera2 直采，推理结果主动推送
 */
public class NpuInferenceService extends Service {

    private static final String TAG = "NpuInferenceService";

    private NpuHttpServer       httpServer;
    private NpuWebSocketServer  wsServer;
    private CameraInferenceManager cameraManager;
    private RknnInference       rknnInference;
    private HandInference       handInference;

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "NPU inference service starting...");
        initNpu();
        startHttpServer();
        startWebSocketServer();
        startCameraInference();
    }

    private void initNpu() {
        try {
            rknnInference = new RknnInference(getApplicationContext());
            boolean ok = rknnInference.init();
            Log.i(TAG, "RKNN pose init: " + (ok ? "SUCCESS" : "FAILED"));
        } catch (Exception e) {
            Log.e(TAG, "RKNN init error: " + e.getMessage());
        }
        try {
            handInference = new HandInference(getApplicationContext());
            boolean ok = handInference.init();
            Log.i(TAG, "Hand inference init: " + (ok ? "SUCCESS" : "FAILED"));
        } catch (Exception e) {
            Log.e(TAG, "Hand inference init error: " + e.getMessage());
        }
    }

    private void startHttpServer() {
        try {
            httpServer = new NpuHttpServer(8080, rknnInference, handInference);
            httpServer.start();
            Log.i(TAG, "HTTP server started on port 8080");
        } catch (Exception e) {
            Log.e(TAG, "HTTP server start error: " + e.getMessage());
        }
    }

    private void startWebSocketServer() {
        try {
            wsServer = new NpuWebSocketServer(8081);
            wsServer.start();
            Log.i(TAG, "WebSocket server started on port 8081");
        } catch (Exception e) {
            Log.e(TAG, "WebSocket server start error: " + e.getMessage());
        }
    }

    private void startCameraInference() {
        try {
            cameraManager = new CameraInferenceManager(
                getApplicationContext(), rknnInference, handInference, wsServer);
            cameraManager.start();
            if (httpServer != null) httpServer.setCameraManager(cameraManager);
            Log.i(TAG, "Camera inference started");
        } catch (Exception e) {
            Log.e(TAG, "Camera inference start error: " + e.getMessage());
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraManager != null) cameraManager.stop();
        if (wsServer      != null) wsServer.stop();
        if (httpServer    != null) httpServer.stop();
        if (rknnInference != null) rknnInference.release();
        if (handInference != null) handInference.release();
        Log.i(TAG, "NPU inference service stopped");
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) { return null; }
}
