package com.example.weblauncher.npu;

import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 轻量级 HTTP 服务器
 * 监听 localhost:8080，提供 NPU 推理 API
 *
 * API:
 *   POST /detect   Body: {"image": "base64..."}  返回手势识别结果
 *   GET  /health   返回服务状态
 */
public class NpuHttpServer {

    private static final String TAG = "NpuHttpServer";
    private final int port;
    private final RknnInference rknn;
    private ServerSocket serverSocket;
    private boolean running = false;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);

    public NpuHttpServer(int port, RknnInference rknn) {
        this.port = port;
        this.rknn = rknn;
    }

    public void start() throws IOException {
        serverSocket = new ServerSocket(port);
        running = true;
        new Thread(() -> {
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    executor.submit(() -> handleRequest(client));
                } catch (IOException e) {
                    if (running) Log.e(TAG, "Accept error: " + e.getMessage());
                }
            }
        }).start();
    }

    private void handleRequest(Socket client) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(client.getInputStream()));
            OutputStream out = client.getOutputStream();

            // 读取请求行
            String requestLine = reader.readLine();
            if (requestLine == null) { client.close(); return; }

            String method = requestLine.split(" ")[0];
            String path = requestLine.split(" ").length > 1 ? requestLine.split(" ")[1] : "/";

            // 读取 headers
            int contentLength = 0;
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                if (line.toLowerCase().startsWith("content-length:")) {
                    contentLength = Integer.parseInt(line.split(":")[1].trim());
                }
            }

            // 读取 body
            String body = "";
            if (contentLength > 0) {
                char[] buf = new char[contentLength];
                reader.read(buf, 0, contentLength);
                body = new String(buf);
            }

            // 路由处理
            String responseBody;
            if (path.equals("/health")) {
                responseBody = "{\"status\":\"ok\",\"npu\":\"RK3576\",\"version\":\"1.0\"}";
            } else if (path.equals("/detect") && method.equals("POST")) {
                responseBody = handleDetect(body);
            } else {
                responseBody = "{\"error\":\"Not found\"}";
            }

            // 返回响应（带 CORS 头，允许网页跨域访问）
            String response = "HTTP/1.1 200 OK\r\n" +
                "Content-Type: application/json\r\n" +
                "Access-Control-Allow-Origin: *\r\n" +
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" +
                "Access-Control-Allow-Headers: Content-Type\r\n" +
                "Content-Length: " + responseBody.getBytes().length + "\r\n" +
                "\r\n" +
                responseBody;

            out.write(response.getBytes());
            out.flush();
            client.close();

        } catch (Exception e) {
            Log.e(TAG, "Request error: " + e.getMessage());
        }
    }

    private String handleDetect(String body) {
        try {
            // 简单解析 JSON，提取 image 字段
            int start = body.indexOf("\"image\"");
            if (start == -1) return "{\"error\":\"Missing image field\"}";

            int colonIdx = body.indexOf(":", start);
            int quoteStart = body.indexOf("\"", colonIdx) + 1;
            int quoteEnd = body.lastIndexOf("\"");
            if (quoteStart <= 0 || quoteEnd <= quoteStart) {
                return "{\"error\":\"Invalid image data\"}";
            }

            String base64Image = body.substring(quoteStart, quoteEnd);
            long t0 = System.currentTimeMillis();
            String result = rknn.infer(base64Image);
            long elapsed = System.currentTimeMillis() - t0;

            // 在结果里附加推理耗时
            return result.replace("}", ",\"inferMs\":" + elapsed + "}");
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    public void stop() {
        running = false;
        executor.shutdown();
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) {}
    }
}
