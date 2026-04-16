package com.example.weblauncher.npu;

import android.util.Log;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;

/**
 * 轻量级 WebSocket 服务器（RFC 6455）
 * 端口 8081，只做广播：NPU 推理结果 JSON 推送给所有连接的客户端
 */
public class NpuWebSocketServer {

    private static final String TAG = "NpuWebSocketServer";
    private static final String WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

    private final int port;
    private ServerSocket serverSocket;
    private boolean running = false;
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final Set<ClientHandler> clients = Collections.synchronizedSet(new HashSet<>());

    // 最新结果缓存，新客户端连接时立刻发一条
    private volatile String lastJson = null;

    public NpuWebSocketServer(int port) {
        this.port = port;
    }

    public void start() throws IOException {
        serverSocket = new ServerSocket(port);
        running = true;
        executor.submit(() -> {
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    executor.submit(new ClientHandler(client));
                } catch (IOException e) {
                    if (running) Log.e(TAG, "accept: " + e.getMessage());
                }
            }
        });
        Log.i(TAG, "WebSocket server listening on :" + port);
    }

    /** 广播 JSON 到所有已握手的客户端 */
    public void broadcast(String json) {
        lastJson = json;
        synchronized (clients) {
            Iterator<ClientHandler> it = clients.iterator();
            while (it.hasNext()) {
                ClientHandler c = it.next();
                if (!c.send(json)) it.remove();
            }
        }
    }

    /**
     * 广播帧包：二进制格式
     * [4字节 JSON长度(big-endian)] [JSON字节] [JPEG字节]
     */
    public void broadcastFrame(String json, byte[] jpeg) {
        lastJson = json;
        try {
            byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);
            ByteArrayOutputStream buf = new ByteArrayOutputStream(4 + jsonBytes.length + jpeg.length);
            int jlen = jsonBytes.length;
            buf.write((jlen >> 24) & 0xFF);
            buf.write((jlen >> 16) & 0xFF);
            buf.write((jlen >>  8) & 0xFF);
            buf.write( jlen        & 0xFF);
            buf.write(jsonBytes);
            buf.write(jpeg);
            byte[] frame = buf.toByteArray();
            synchronized (clients) {
                Iterator<ClientHandler> it = clients.iterator();
                while (it.hasNext()) {
                    ClientHandler c = it.next();
                    if (!c.sendBinary(frame)) it.remove();
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "broadcastFrame: " + e.getMessage());
        }
    }

    public void stop() {
        running = false;
        executor.shutdownNow();
        try { if (serverSocket != null) serverSocket.close(); } catch (IOException ignored) {}
    }

    // ── 单个客户端处理 ──────────────────────────────────────────────
    private class ClientHandler implements Runnable {
        private final Socket socket;
        private OutputStream out;
        private volatile boolean ready = false;

        ClientHandler(Socket socket) { this.socket = socket; }

        @Override
        public void run() {
            try {
                socket.setTcpNoDelay(true);
                InputStream in   = socket.getInputStream();
                out = socket.getOutputStream();

                // HTTP 握手
                BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
                Map<String, String> headers = new HashMap<>();
                String line;
                while ((line = reader.readLine()) != null && !line.isEmpty()) {
                    int colon = line.indexOf(':');
                    if (colon > 0) headers.put(
                        line.substring(0, colon).trim().toLowerCase(),
                        line.substring(colon + 1).trim());
                }

                String key = headers.get("sec-websocket-key");
                if (key == null) { socket.close(); return; }

                String accept = Base64.getEncoder().encodeToString(
                    MessageDigest.getInstance("SHA-1")
                        .digest((key + WS_MAGIC).getBytes(StandardCharsets.UTF_8)));

                String response = "HTTP/1.1 101 Switching Protocols\r\n"
                    + "Upgrade: websocket\r\n"
                    + "Connection: Upgrade\r\n"
                    + "Access-Control-Allow-Origin: *\r\n"
                    + "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
                out.write(response.getBytes(StandardCharsets.UTF_8));
                out.flush();

                ready = true;
                clients.add(this);
                Log.i(TAG, "Client connected, total=" + clients.size());

                // 立刻推最新结果
                if (lastJson != null) send(lastJson);

                // 保持连接，读 ping / close 帧
                byte[] buf = new byte[256];
                while (!socket.isClosed()) {
                    int b0 = in.read();
                    if (b0 < 0) break;
                    int b1 = in.read();
                    if (b1 < 0) break;
                    int opcode = b0 & 0x0F;
                    if (opcode == 0x8) break; // close
                    // 读完帧（不处理内容，只是消耗掉）
                    long payloadLen = b1 & 0x7F;
                    if (payloadLen == 126) {
                        payloadLen = ((in.read() & 0xFF) << 8) | (in.read() & 0xFF);
                    } else if (payloadLen == 127) {
                        payloadLen = 0;
                        for (int i = 0; i < 8; i++) payloadLen = (payloadLen << 8) | (in.read() & 0xFF);
                    }
                    boolean masked = (b1 & 0x80) != 0;
                    if (masked) in.readNBytes(4); // mask key
                    long remaining = payloadLen;
                    while (remaining > 0) {
                        int read = in.read(buf, 0, (int) Math.min(remaining, buf.length));
                        if (read < 0) break;
                        remaining -= read;
                    }
                }
            } catch (Exception e) {
                Log.d(TAG, "Client disconnected: " + e.getMessage());
            } finally {
                clients.remove(this);
                try { socket.close(); } catch (IOException ignored) {}
                Log.i(TAG, "Client removed, total=" + clients.size());
            }
        }

        /** 发送文本帧，失败返回 false */
        boolean send(String text) {
            if (!ready || socket.isClosed()) return false;
            try {
                byte[] payload = text.getBytes(StandardCharsets.UTF_8);
                ByteArrayOutputStream frame = new ByteArrayOutputStream();
                frame.write(0x81); // FIN + text opcode
                int len = payload.length;
                if (len <= 125) {
                    frame.write(len);
                } else if (len <= 65535) {
                    frame.write(126);
                    frame.write((len >> 8) & 0xFF);
                    frame.write(len & 0xFF);
                } else {
                    frame.write(127);
                    for (int i = 7; i >= 0; i--) frame.write((int)((len >> (i * 8)) & 0xFF));
                }
                frame.write(payload);
                synchronized (out) {
                    out.write(frame.toByteArray());
                    out.flush();
                }
                return true;
            } catch (IOException e) {
                return false;
            }
        }

        /** 发送二进制帧，失败返回 false */
        boolean sendBinary(byte[] payload) {
            if (!ready || socket.isClosed()) return false;
            try {
                ByteArrayOutputStream frame = new ByteArrayOutputStream();
                frame.write(0x82); // FIN + binary opcode
                int len = payload.length;
                if (len <= 125) {
                    frame.write(len);
                } else if (len <= 65535) {
                    frame.write(126);
                    frame.write((len >> 8) & 0xFF);
                    frame.write(len & 0xFF);
                } else {
                    frame.write(127);
                    for (int i = 7; i >= 0; i--) frame.write((int)((len >> (i * 8)) & 0xFF));
                }
                frame.write(payload);
                synchronized (out) {
                    out.write(frame.toByteArray());
                    out.flush();
                }
                return true;
            } catch (IOException e) {
                return false;
            }
        }
    }
}
