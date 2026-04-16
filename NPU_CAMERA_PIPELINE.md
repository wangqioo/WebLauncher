# NPU 推理管道优化记录

设备：Rockchip RK3576 / Android 14  
模型：yolov8n-pose（人体姿态，17关键点）+ Gold-YOLO 手部检测 + MediaPipe 手部关键点  
目标：Camera2 采集 → NPU 推理 → WebSocket 推流，尽量提高帧率

---

## 最终架构

```
Camera2 (YUV_420_888, 640×480)
    ↓ JNI: YUV→letterbox RGB + NPU 推理（C++，一步完成）
RKNN NPU (yolov8n-pose / hand_detection)
    ↓ C++ decode → JSON
NpuWebSocketServer (port 8081, binary frame)
    ↓ ws://localhost:8081 (adb forward)
前端 HTML (canvas 渲染关键点 + JPEG 预览)
```

**Binary WebSocket 帧格式：**

```
[4 bytes big-endian: JSON length][JSON bytes][JPEG bytes]
```

---

## 优化历程与各阶段耗时

### 阶段 0：基线（HTTP base64）

架构：前端发 base64 图片 → HTTP POST `/detect` → Java 解码 → NPU 推理 → 返回 JSON

| 指标 | 数值 |
|------|------|
| 帧率 | ~2fps |
| 瓶颈 | HTTP 往返 + base64 编解码 |

---

### 阶段 1：Camera2 + WebSocket

用 Android 端 Camera2 直接采帧，省去前端摄像头和 base64 传输。  
实现了 RFC 6455 WebSocket 服务器（Java 手写），binary frame 携带 JSON + JPEG。

| 指标 | 数值 |
|------|------|
| 帧率 | ~4fps（250ms/帧） |
| infer | ~150ms |
| jpeg | ~50ms |
| 瓶颈 | Java YUV→RGB 像素循环（307K 像素）+ JPEG 640×480 编码 |

---

### 阶段 2：JPEG 缩小编码

JPEG 编码前先 resize 到 320×240，减少编码时间。

| 指标 | 数值 |
|------|------|
| jpeg | **12ms**（从 50ms 降到 12ms）|
| total | ~240ms |

---

### 阶段 3：resize 移到 C++（nativeInferResized）

在 JNI 层用 C++ nearest-neighbor resize（letterbox），避免 Java 像素循环。

**结果：** infer 从 150ms **增加**到 260ms（回归！）

**原因：** Java YUV→RGB 循环（307K 像素 × 3 浮点运算）才是瓶颈，C++ resize 没有解决这个问题，反而因为多了一次 1.2MB 数据的 JNI 拷贝而更慢。

---

### 阶段 4：YUV→RGB 移到 C++（nativeInferYuv）

直接把 YUV 三个 plane 的 byte[] 传给 JNI，在 C++ 完成 YUV→letterbox RGB，消除 Java 的 307K 像素循环。

C++ 计时结果：

| 步骤 | 耗时 |
|------|------|
| GetByteArrayElements | 0ms |
| YUV→letterbox RGB | 70ms |
| NPU inputsSet | 23ms |
| NPU run | 57ms |
| NPU outputsGet | 15ms |
| JNI jfloatArray 拷贝 | 6ms |
| Java buildResult | ~100ms |
| **C++ 合计** | **171ms** |
| **Java total** | **270ms** |

**Java buildResult 100ms 原因：** NPU 返回 974,400 个 float（3.8MB），全部拷贝到 Java，再在 Java 做 DFL decode + NMS + keypoint 解析。

---

### 阶段 5：pose decode 移到 C++（poseDecodeJson）

在 C++ 直接完成 YOLOv8-pose 的 DFL decode、置信度筛选、关键点坐标还原、JSON 构建，`nativeInferYuv` 直接返回 `jstring`，不再把 974K float 传给 Java。

| 步骤 | 耗时 |
|------|------|
| YUV→letterbox RGB | 70ms |
| NPU run（含 inputsSet + outputsGet）| 84ms |
| C++ poseDecodeJson | **1ms** |
| **C++ 合计** | **155ms** |
| Java total（含 nativeYuvToRgb for JPEG）| **190ms** |
| JPEG 编码（320×240）| 19ms |
| **端到端 total** | **~210ms** |

---

### 阶段 6：整数定点数替代浮点 YUV 转换

把 YUV→RGB 的浮点系数换成 Q13 定点整数近似：

```
1.370705 → 11277/8192
0.337633 → 2765/8192
0.698001 → 5726/8192
1.732446 → 14216/8192
```

同时把内循环 `dy * inv_fp + srcY` 改为行指针直接索引，减少乘法。

**结果：** YUV 转换从 70ms 降到 **60-65ms**（提升 ~10%）。  
瓶颈在内存带宽，整数运算效果有限。

---

### 阶段 7：JPEG 分辨率改为 160×120

| 指标 | 数值 |
|------|------|
| JPEG 编码 | **6ms**（从 19ms 降到 6ms）|
| **最终 total** | **~190ms ≈ 5fps** |

---

### 尝试：降低采集分辨率到 320×240

**预期：** YUV→letterbox 循环减少到 76800 次，convert 从 60ms 降到 15ms。

**实际：** convert 仍然 60ms，无变化。

**原因：** letterbox 的目标是 640×640（NPU 需要），scale = min(640/320, 640/240) = 2.0，scaledH = 480，scaledW = 640，循环次数完全一样（480×640 = 307K），只是从源读的分辨率降低了，但写入目标仍然是 307K 像素。

**结论：** 降采集分辨率对 letterbox 无效，已回滚到 640×480。

---

### 尝试：Push 模式（遍历源像素）

**思路：** 改为遍历源像素 320×240 = 76K 次，每个源像素"推送"到对应的目标区域（1 个源像素对应 scale² 个目标像素）。

**结果：** convert 约 57ms，略低于 pull 模式的 63ms，但收益不显著。  
**原因：** 目标写入是随机跳跃（缓存不友好），整体内存带宽仍然是瓶颈。

**结论：** 保留 push 模式（稍快），但不是本质优化。

---

## 硬件瓶颈分析

当前各步骤不可压缩的耗时：

| 步骤 | 耗时 | 优化空间 |
|------|------|---------|
| YUV→letterbox（307K 像素，内存带宽限制）| ~60ms | 极小 |
| NPU inputsSet（1.2MB 数据 DMA 传输）| ~23ms | 无 |
| NPU run（硬件推理）| ~57ms | 无（换模型除外）|
| JPEG 编码（160×120 @ Q65）| ~6ms | 极小 |
| JNI + Java 调度开销 | ~30ms | 小 |
| **理论下限** | **~176ms** | **~5.7fps** |

实际稳定值约 **190-215ms（4.7-5.3fps）**。

---

## 关键 Bug 修复记录

### 手部检测始终返回 `detected: false`

**原因 1：** `parseBestBox` 直接用 raw logit 比较阈值，未做 sigmoid。  
Gold-YOLO scores 输出是 raw logit，值域大约 -10 到 +10，直接与 0.3 比较永远不会触发。  
**修复：** `float score = sigmoid(raw[dataStart1 + i])`

**原因 2：** box 坐标乘以了 `imgW/imgH`。Gold-YOLO 输出的 box 是绝对像素坐标（0~640/480），不是归一化坐标。  
**修复：** 去掉 `* imgW` 乘法，直接用 `raw[dataStart + bestIdx*4]`

### adb 端口冲突（EADDRINUSE）

**症状：** 用 `adb reverse tcp:8080 tcp:8080` 后，Android 端口被 adbd 占用，app 无法绑定 8080。  
**原因：** `adb reverse` 是设备→PC 方向，adbd 在设备上占用该端口。  
**修复：** 改用 `adb forward tcp:8080 tcp:8080`（PC 端口转发到设备），app 正常绑定。

### Body pose 被手部检测覆盖

**症状：** 切换到 pose 模式后，推理结果仍然是手部数据。  
**原因：** `CameraInferenceManager` 初始 `mode = "hand"` 硬编码，前端 WebSocket 连接时没有发送当前 mode。  
**修复：** 默认 mode 改为 `"pose"`；WebSocket `onopen` 时调用 `notifyMode(mode)` 同步状态。

### buildResult 解析错误（landmark 输出）

**症状：** 手部关键点坐标全部归零或异常。  
**原因：** JNI 返回的 float[] 带有 header `[nOutput, size0, size1, ...]`，Java 的 `buildResult` 直接从 index 0 读，把 header 当数据。  
**修复：** 解析 header 后，把 `lmRaw[lmDataStart:]` 拷贝到单独数组再处理。

---

## 最终文件结构

```
app/src/main/
├── cpp/
│   └── rknn_jni.cpp          # JNI：YUV→RGB、letterbox、推理、pose decode、JSON
├── java/.../npu/
│   ├── RknnInference.java    # pose 推理（inferYuv 直接返回 JSON）
│   ├── HandInference.java    # 手部检测 + landmark（inferYuv）
│   ├── CameraInferenceManager.java  # Camera2 采帧 → 推理 → WebSocket
│   ├── NpuWebSocketServer.java      # RFC 6455 WebSocket 服务（port 8081）
│   ├── NpuHttpServer.java    # HTTP 服务（port 8080，mode 切换）
│   └── NpuInferenceService.java     # Android Service，统一管理上述组件
└── assets/
    ├── yolov8n-pose.rknn
    ├── hand_detection.rknn
    └── hand_landmark.rknn
```

---

## 前端接入

```bash
adb forward tcp:8080 tcp:8080
adb forward tcp:8081 tcp:8081
```

打开 `npu-validator.html`，或在 WebView 中访问前端页面，通过 `ws://localhost:8081` 接收推理帧。

---

*记录时间：2026-04-16*  
*设备：RK3576 / Android 14 / RKNN SDK 2.3.2*
