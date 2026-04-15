# RK3576 NPU 推理 API 文档

WebLauncher 在设备后台运行一个本地 HTTP 服务，网页可以通过 `fetch` 调用这个接口，把图像交给 RK3576 的 6 TOPS NPU 做推理，完全绕开 WebGL/WebAssembly 的性能限制。

---

## 基本信息

| 项目 | 值 |
|------|----|
| 地址 | `http://localhost:8080` |
| 协议 | HTTP/1.1 |
| 跨域 | 已开启 CORS，网页可以直接 fetch，无需额外配置 |
| 模型 | YOLOv8n-Pose（人体姿态，17 个关键点） |
| 输入尺寸 | 640×640（服务端自动缩放，传任意尺寸图像均可） |

---

## 接口列表

### GET /health — 检查服务状态

确认 NPU 服务是否在运行，建议页面初始化时先调用一次。

**请求**

```
GET http://localhost:8080/health
```

**响应**

```json
{
  "status": "ok",
  "npu": "RK3576",
  "version": "1.0"
}
```

**示例代码**

```javascript
const res = await fetch('http://localhost:8080/health');
const data = await res.json();
if (data.status === 'ok') {
  console.log('NPU 服务正常');
}
```

---

### POST /detect — 人体姿态推理

传入一张图像，返回画面中置信度最高的一个人的 17 个关键点坐标。

**请求**

```
POST http://localhost:8080/detect
Content-Type: application/json

{
  "image": "<base64 字符串>"
}
```

- `image` 字段支持两种格式：
  - 带前缀：`data:image/jpeg;base64,/9j/4AAQ...`
  - 纯 base64：`/9j/4AAQ...`

**响应 — 检测到人体**

```json
{
  "detected": true,
  "confidence": 0.9123,
  "inferMs": 18,
  "keypoints": [
    { "name": "nose",           "x": 0.512, "y": 0.134, "score": 0.981 },
    { "name": "left_eye",       "x": 0.531, "y": 0.118, "score": 0.973 },
    { "name": "right_eye",      "x": 0.492, "y": 0.117, "score": 0.969 },
    { "name": "left_ear",       "x": 0.558, "y": 0.127, "score": 0.921 },
    { "name": "right_ear",      "x": 0.465, "y": 0.126, "score": 0.918 },
    { "name": "left_shoulder",  "x": 0.601, "y": 0.245, "score": 0.954 },
    { "name": "right_shoulder", "x": 0.421, "y": 0.243, "score": 0.951 },
    { "name": "left_elbow",     "x": 0.643, "y": 0.381, "score": 0.887 },
    { "name": "right_elbow",    "x": 0.378, "y": 0.379, "score": 0.882 },
    { "name": "left_wrist",     "x": 0.672, "y": 0.498, "score": 0.861 },
    { "name": "right_wrist",    "x": 0.349, "y": 0.495, "score": 0.858 },
    { "name": "left_hip",       "x": 0.578, "y": 0.521, "score": 0.931 },
    { "name": "right_hip",      "x": 0.443, "y": 0.519, "score": 0.928 },
    { "name": "left_knee",      "x": 0.589, "y": 0.682, "score": 0.843 },
    { "name": "right_knee",     "x": 0.432, "y": 0.680, "score": 0.840 },
    { "name": "left_ankle",     "x": 0.594, "y": 0.841, "score": 0.812 },
    { "name": "right_ankle",    "x": 0.427, "y": 0.839, "score": 0.809 }
  ]
}
```

**响应 — 未检测到人体**

```json
{
  "detected": false,
  "confidence": 0
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `detected` | boolean | 是否检测到人体 |
| `confidence` | number | 检测框的置信度，0~1 |
| `inferMs` | number | NPU 推理耗时（毫秒） |
| `keypoints` | array | 17 个关键点，顺序固定（见下表） |
| `keypoints[].name` | string | 关键点名称 |
| `keypoints[].x` | number | 归一化 x 坐标，0~1（相对于输入图像宽度） |
| `keypoints[].y` | number | 归一化 y 坐标，0~1（相对于输入图像高度） |
| `keypoints[].score` | number | 该关键点的可见度置信度，0~1 |

**17 个关键点顺序**

| 索引 | 名称 | 部位 |
|------|------|------|
| 0 | nose | 鼻子 |
| 1 | left_eye | 左眼 |
| 2 | right_eye | 右眼 |
| 3 | left_ear | 左耳 |
| 4 | right_ear | 右耳 |
| 5 | left_shoulder | 左肩 |
| 6 | right_shoulder | 右肩 |
| 7 | left_elbow | 左肘 |
| 8 | right_elbow | 右肘 |
| 9 | left_wrist | 左手腕 |
| 10 | right_wrist | 右手腕 |
| 11 | left_hip | 左髋 |
| 12 | right_hip | 右髋 |
| 13 | left_knee | 左膝 |
| 14 | right_knee | 右膝 |
| 15 | left_ankle | 左踝 |
| 16 | right_ankle | 右踝 |

> 注意：left/right 是从**被检测者视角**定义的，不是摄像头视角。

---

## 完整使用示例

### 从摄像头实时推理

```javascript
const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// 启动摄像头
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
video.srcObject = stream;
await video.play();

let inferring = false;

async function inferLoop() {
  if (!inferring) {
    inferring = true;

    // 把当前帧画到 canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // 转 base64
    const base64 = canvas.toDataURL('image/jpeg', 0.8);

    try {
      const res = await fetch('http://localhost:8080/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      });
      const data = await res.json();

      if (data.detected) {
        console.log(`推理耗时 ${data.inferMs}ms，置信度 ${data.confidence.toFixed(2)}`);
        drawKeypoints(data.keypoints, canvas.width, canvas.height);
      }
    } catch (e) {
      console.warn('NPU 服务不可用', e);
    }

    inferring = false;
  }

  requestAnimationFrame(inferLoop);
}

inferLoop();
```

### 绘制关键点骨架

```javascript
function drawKeypoints(keypoints, width, height) {
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);

  // 骨架连接关系
  const skeleton = [
    [0,1],[0,2],[1,3],[2,4],           // 头部
    [5,6],[5,7],[7,9],[6,8],[8,10],    // 上肢
    [5,11],[6,12],[11,12],             // 躯干
    [11,13],[13,15],[12,14],[14,16]    // 下肢
  ];

  // 画连线
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  for (const [a, b] of skeleton) {
    const kpA = keypoints[a];
    const kpB = keypoints[b];
    if (kpA.score > 0.5 && kpB.score > 0.5) {
      ctx.beginPath();
      ctx.moveTo(kpA.x * width, kpA.y * height);
      ctx.lineTo(kpB.x * width, kpB.y * height);
      ctx.stroke();
    }
  }

  // 画关键点
  ctx.fillStyle = '#ff0000';
  for (const kp of keypoints) {
    if (kp.score > 0.5) {
      ctx.beginPath();
      ctx.arc(kp.x * width, kp.y * height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
```

### 检查服务可用性（页面初始化时）

```javascript
async function checkNpu() {
  try {
    const res = await fetch('http://localhost:8080/health', { signal: AbortSignal.timeout(1000) });
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

const npuAvailable = await checkNpu();
if (npuAvailable) {
  console.log('使用 NPU 推理');
} else {
  console.log('NPU 不可用，回退到 MediaPipe');
}
```

---

## 错误响应

| 情况 | 响应 |
|------|------|
| 缺少 image 字段 | `{"error":"Missing image field"}` |
| 图像 base64 无效 | `{"error":"Invalid image"}` |
| NPU 未初始化 | `{"error":"RKNN not initialized"}` |
| 路径不存在 | `{"error":"Not found"}` |

---

## 注意事项

1. **仅在 WebLauncher 内运行的网页可以访问此接口**，外部网络无法访问 localhost。
2. **每次只返回一个人**（置信度最高的检测框），多人场景只处理最突出的一个。
3. **推理耗时参考**：RK3576 NPU 单次推理约 15~25ms，加上图像传输约 20~40ms 总延迟，可实现 25fps 以上的实时处理。
4. **score < 0.5 的关键点**建议忽略，表示该关键点被遮挡或置信度不足。
5. 坐标 `x`、`y` 是归一化值（0~1），使用时乘以画布实际宽高即可还原像素坐标。
