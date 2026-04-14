package com.example.weblauncher.npu;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;

/**
 * 后台 NPU 推理服务
 * 内嵌 HTTP 服务器，网页通过 localhost:8080 调用 NPU 推理
 */
public class NpuInferenceService extends Service {

    private static final String TAG = "NpuInferenceService";
    private NpuHttpServer httpServer;
    private RknnInference rknnInference;

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "NPU inference service starting...");
        initNpu();
        startHttpServer();
    }

    private void initNpu() {
        try {
            rknnInference = new RknnInference(getApplicationContext());
            boolean ok = rknnInference.init();
            Log.i(TAG, "RKNN init: " + (ok ? "SUCCESS" : "FAILED"));
        } catch (Exception e) {
            Log.e(TAG, "RKNN init error: " + e.getMessage());
        }
    }

    private void startHttpServer() {
        try {
            httpServer = new NpuHttpServer(8080, rknnInference);
            httpServer.start();
            Log.i(TAG, "HTTP server started on port 8080");
        } catch (Exception e) {
            Log.e(TAG, "HTTP server start error: " + e.getMessage());
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (httpServer != null) httpServer.stop();
        if (rknnInference != null) rknnInference.release();
        Log.i(TAG, "NPU inference service stopped");
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
