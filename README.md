# WebLauncher

运行在 Rockchip RK3576 Android 设备上的网页启动器，内置 NPU 推理服务，让网页可以直接调用设备的 6 TOPS NPU 进行 AI 推理。

---

## 功能

- **网页桌面**：全屏沉浸式，支持添加/删除网页应用，5列图标网格布局
- **WebView 优化**：硬件加速、桌面版 UA、自动授权摄像头/麦克风、WebAssembly SIMD
- **NPU 推理服务**：后台 HTTP 服务（localhost:8080），网页通过 fetch 调用 RK3576 NPU
- **人体姿态检测**：YOLOv8n-Pose，全身 17 个关键点
- **手部关键点检测**：Gold-YOLO + MediaPipe Landmark，手部 21 个关键点

---

## 设备要求

| 项目 | 要求 |
|------|------|
| 芯片 | Rockchip RK3576 |
| 系统 | Android 14 |
| NPU | 6 TOPS（RKNN SDK 2.3.2） |
| ABI | arm64-v8a |

---

## 项目结构

```
app/src/main/
├── java/com/example/weblauncher/
│   ├── LauncherActivity.java     # 主界面，网格桌面
│   ├── WebActivity.java          # WebView 页面，全屏+权限
│   ├── WebApp.java               # 数据模型
│   ├── AppAdapter.java           # RecyclerView 适配器
│   └── npu/
│       ├── NpuInferenceService.java  # 后台 Service，管理模型生命周期
│       ├── NpuHttpServer.java        # HTTP 服务器，监听 :8080
│       ├── RknnInference.java        # 人体姿态推理（yolov8n-pose）
│       └── HandInference.java        # 手部关键点推理（两阶段）
├── cpp/
│   ├── rknn_jni.cpp              # JNI 桥接，调用 librknnrt.so
│   ├── CMakeLists.txt
│   └── include/rknn_api.h
├── assets/
│   ├── yolov8n-pose.rknn         # 人体姿态模型（10MB）
│   ├── hand_detection.rknn       # 手部检测模型（13MB）
│   └── hand_landmark.rknn        # 手部关键点模型（5.7MB）
└── jniLibs/arm64-v8a/
    └── librknnrt.so              # RKNN 运行时（8MB）
```

---

## 构建

### 环境要求

- Android Studio 或命令行 Gradle
- JDK 17+
- Android SDK（API 34）
- NDK 25.1.8937393
- CMake 3.22+

### 步骤

```bash
git clone https://github.com/wangqioo/WebLauncher.git
cd WebLauncher

# 配置 SDK 路径（改成你自己的路径）
echo "sdk.dir=/Users/你的用户名/Library/Android/sdk" > local.properties

./gradlew assembleDebug
```

APK 输出路径：`app/build/outputs/apk/debug/app-debug.apk`

### 安装到设备

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## NPU 推理接口

APP 启动后会在后台自动运行 HTTP 服务，网页通过 `fetch` 直接调用。

### 接口一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/health` | 检查服务状态 |
| POST | `/detect` | 人体姿态，17个关键点 |
| POST | `/detect/hand` | 手部关键点，21个关键点 |

### 快速调用

```javascript
// 检查服务
const { status } = await fetch('http://localhost:8080/health').then(r => r.json());

// 人体姿态
const pose = await fetch('http://localhost:8080/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.8) })
}).then(r => r.json());

// 手部关键点
const hand = await fetch('http://localhost:8080/detect/hand', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.8) })
}).then(r => r.json());
```

详细接口文档、字段说明、完整示例代码（骨架绘制、手势识别）见 [NPU_API.md](./NPU_API.md)。

---

## 架构说明

### 为什么需要 NPU 推理服务

RK3576 的 GPU（Mali-G52，~50 GFLOPS）性能有限，MediaPipe 在 WebView 里通过 WebGL 做推理会严重卡顿。NPU 有 6 TOPS 算力，单次推理耗时 15~25ms，是 GPU 方案的 10 倍以上。

```
网页 (WebView)
    ↕  fetch('http://localhost:8080/detect')
NpuHttpServer (Android Service, :8080)
    ↕  JNI
rknn_jni.cpp
    ↕  librknnrt.so
NPU (6 TOPS)
```

### 手部推理两阶段流程

```
输入图像
    ↓
hand_detection.rknn (640×480)
    → 找到手的位置（bounding box）
    ↓ 裁剪 + 加 10% padding
hand_landmark.rknn (224×224)
    → 提取 21 个手指关键点
    ↓
返回归一化坐标（相对原始图像）
```

---

## 相关文档

- [NPU_API.md](./NPU_API.md) — 网页调用接口完整文档，含示例代码
- [NPU_OPTIMIZATION.md](./NPU_OPTIMIZATION.md) — 性能优化方案评估（A~E 五套方案）
