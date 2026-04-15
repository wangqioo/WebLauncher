# RK3576 NPU 推理 API 文档

WebLauncher 在设备后台运行一个本地 HTTP 服务，网页可以通过 `fetch` 调用这个接口，把图像交给 RK3576 的 6 TOPS NPU 做推理，完全绕开 WebGL/WebAssembly 的性能限制。

---

## 基本信息

| 项目 | 值 |
|------|----|
| 地址 | `http://localhost:8080` |
| 协议 | HTTP/1.1 |
| 跨域 | 已开启 CORS，网页可以直接 fetch，无需任何配置 |

### 已部署模型

| 接口 | 模型 | 用途 | 输入尺寸 |
|------|------|------|---------|
| `POST /detect` | YOLOv8n-Pose | 人体姿态，17个关键点 | 640×640（自动缩放） |
| `POST /detect/hand` | Gold-YOLO + MediaPipe Landmark | 手部关键点，21个关键点 | 640×480 → 224×224（两阶段，自动处理） |

---

## 接口详情

### GET /health — 检查服务状态

页面初始化时先调用，确认 NPU 服务是否在线。

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

---

### POST /detect — 人体姿态（17关键点）

检测画面中置信度最高的一个人，返回全身 17 个关键点坐标。

**适用场景：** 体感游戏、舞蹈跟随、健身动作检测、全身互动

**请求**
```
POST http://localhost:8080/detect
Content-Type: application/json

{ "image": "<base64>" }
```

`image` 字段支持两种格式：
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
{ "detected": false, "confidence": 0 }
```

**17 个关键点索引**

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

> left/right 是从**被检测者视角**定义的，不是摄像头视角。

---

### POST /detect/hand — 手部关键点（21关键点）

两阶段推理：先用 Gold-YOLO 找到手的位置，再对手部区域提取 21 个精细关键点。

**适用场景：** 手势识别、手指控制、虚拟触控、手语识别

**请求**
```
POST http://localhost:8080/detect/hand
Content-Type: application/json

{ "image": "<base64>" }
```

**响应 — 检测到手**
```json
{
  "detected": true,
  "inferMs": 35,
  "box": { "x": 0.31, "y": 0.22, "w": 0.18, "h": 0.24 },
  "keypoints": [
    { "name": "wrist",       "x": 0.412, "y": 0.581 },
    { "name": "thumb_cmc",   "x": 0.401, "y": 0.548 },
    { "name": "thumb_mcp",   "x": 0.388, "y": 0.511 },
    { "name": "thumb_ip",    "x": 0.371, "y": 0.482 },
    { "name": "thumb_tip",   "x": 0.358, "y": 0.459 },
    { "name": "index_mcp",   "x": 0.418, "y": 0.498 },
    { "name": "index_pip",   "x": 0.421, "y": 0.461 },
    { "name": "index_dip",   "x": 0.423, "y": 0.431 },
    { "name": "index_tip",   "x": 0.425, "y": 0.408 },
    { "name": "middle_mcp",  "x": 0.438, "y": 0.495 },
    { "name": "middle_pip",  "x": 0.441, "y": 0.456 },
    { "name": "middle_dip",  "x": 0.443, "y": 0.424 },
    { "name": "middle_tip",  "x": 0.445, "y": 0.399 },
    { "name": "ring_mcp",    "x": 0.456, "y": 0.499 },
    { "name": "ring_pip",    "x": 0.459, "y": 0.463 },
    { "name": "ring_dip",    "x": 0.461, "y": 0.434 },
    { "name": "ring_tip",    "x": 0.463, "y": 0.411 },
    { "name": "pinky_mcp",   "x": 0.471, "y": 0.506 },
    { "name": "pinky_pip",   "x": 0.474, "y": 0.474 },
    { "name": "pinky_dip",   "x": 0.476, "y": 0.449 },
    { "name": "pinky_tip",   "x": 0.478, "y": 0.430 }
  ]
}
```

**响应 — 未检测到手**
```json
{ "detected": false }
```

**21 个关键点结构**

```
手腕 (0)
  拇指：cmc(1) → mcp(2) → ip(3)  → tip(4)
  食指：mcp(5) → pip(6) → dip(7)  → tip(8)
  中指：mcp(9) → pip(10)→ dip(11) → tip(12)
  无名：mcp(13)→ pip(14)→ dip(15) → tip(16)
  小指：mcp(17)→ pip(18)→ dip(19) → tip(20)
```

| 缩写 | 含义 |
|------|------|
| cmc | 腕掌关节（拇指根部） |
| mcp | 掌指关节（手指与手掌连接处） |
| pip | 近端指间关节（手指第一个弯曲处） |
| dip | 远端指间关节（手指第二个弯曲处） |
| tip | 指尖 |

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `detected` | boolean | 是否检测到手 |
| `inferMs` | number | 两阶段总推理耗时（毫秒） |
| `box.x/y` | number | 手部检测框左上角，归一化坐标 0~1 |
| `box.w/h` | number | 手部检测框宽高，归一化 0~1 |
| `keypoints[].name` | string | 关键点名称 |
| `keypoints[].x/y` | number | 归一化坐标 0~1，相对于原始图像 |

---

## 完整代码示例

### 通用工具函数

以下函数在所有示例中复用，建议放在公共 JS 文件里。

```javascript
// 检查 NPU 服务是否可用（超时 1 秒）
async function checkNpu() {
  try {
    const res = await fetch('http://localhost:8080/health', {
      signal: AbortSignal.timeout(1000)
    });
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

// 把 video/canvas 元素的当前帧转成 base64
function frameToBase64(source, quality = 0.8) {
  const canvas = document.createElement('canvas');
  canvas.width  = source.videoWidth  || source.width;
  canvas.height = source.videoHeight || source.height;
  canvas.getContext('2d').drawImage(source, 0, 0);
  return canvas.toDataURL('image/jpeg', quality);
}

// 通用推理入口，返回解析后的 JSON 或 null
async function npuInfer(endpoint, base64Image) {
  try {
    const res = await fetch(`http://localhost:8080${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });
    return await res.json();
  } catch {
    return null;
  }
}
```

---

### 示例一：人体姿态实时推理 + 骨架绘制

```html
<video id="video" autoplay playsinline></video>
<canvas id="overlay"></canvas>

<script>
const video   = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx     = overlay.getContext('2d');

// 骨架连接关系（关键点索引对）
const SKELETON = [
  [0,1],[0,2],[1,3],[2,4],          // 头部
  [5,6],[5,7],[7,9],[6,8],[8,10],   // 上肢
  [5,11],[6,12],[11,12],            // 躯干
  [11,13],[13,15],[12,14],[14,16]   // 下肢
];

async function startPose() {
  const npuOk = await checkNpu();
  if (!npuOk) { alert('NPU 服务未启动'); return; }

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;

  let busy = false;
  async function loop() {
    if (!busy) {
      busy = true;
      const base64 = frameToBase64(video);
      const data   = await npuInfer('/detect', base64);

      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (data?.detected) {
        drawPoseSkeleton(data.keypoints, overlay.width, overlay.height);
      }
      busy = false;
    }
    requestAnimationFrame(loop);
  }
  loop();
}

function drawPoseSkeleton(keypoints, w, h) {
  // 骨架连线
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  for (const [a, b] of SKELETON) {
    const kpA = keypoints[a], kpB = keypoints[b];
    if (kpA.score > 0.5 && kpB.score > 0.5) {
      ctx.beginPath();
      ctx.moveTo(kpA.x * w, kpA.y * h);
      ctx.lineTo(kpB.x * w, kpB.y * h);
      ctx.stroke();
    }
  }
  // 关键点圆点
  ctx.fillStyle = '#ff4444';
  for (const kp of keypoints) {
    if (kp.score > 0.5) {
      ctx.beginPath();
      ctx.arc(kp.x * w, kp.y * h, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

startPose();
</script>
```

---

### 示例二：手部关键点实时推理 + 骨架绘制

```html
<video id="video" autoplay playsinline></video>
<canvas id="overlay"></canvas>

<script>
const video   = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx     = overlay.getContext('2d');

// 手部骨架连接（关键点索引对）
const HAND_SKELETON = [
  [0,1],[1,2],[2,3],[3,4],      // 拇指
  [0,5],[5,6],[6,7],[7,8],      // 食指
  [0,9],[9,10],[10,11],[11,12], // 中指
  [0,13],[13,14],[14,15],[15,16],// 无名指
  [0,17],[17,18],[18,19],[19,20] // 小指
];

async function startHand() {
  const npuOk = await checkNpu();
  if (!npuOk) { alert('NPU 服务未启动'); return; }

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;

  let busy = false;
  async function loop() {
    if (!busy) {
      busy = true;
      const base64 = frameToBase64(video);
      const data   = await npuInfer('/detect/hand', base64);

      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (data?.detected) {
        drawHandSkeleton(data.keypoints, overlay.width, overlay.height);

        // 可选：绘制检测框
        const b = data.box;
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 1;
        ctx.strokeRect(b.x * overlay.width, b.y * overlay.height,
                       b.w * overlay.width, b.h * overlay.height);
      }
      busy = false;
    }
    requestAnimationFrame(loop);
  }
  loop();
}

function drawHandSkeleton(keypoints, w, h) {
  // 连线
  ctx.strokeStyle = '#00aaff';
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_SKELETON) {
    const kpA = keypoints[a], kpB = keypoints[b];
    ctx.beginPath();
    ctx.moveTo(kpA.x * w, kpA.y * h);
    ctx.lineTo(kpB.x * w, kpB.y * h);
    ctx.stroke();
  }
  // 关键点
  for (const kp of keypoints) {
    // 指尖用红色，其他用白色
    ctx.fillStyle = kp.name.endsWith('_tip') ? '#ff4444' : '#ffffff';
    ctx.beginPath();
    ctx.arc(kp.x * w, kp.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

startHand();
</script>
```

---

### 示例三：基于手部关键点的手势识别

拿到 21 个关键点后，可以用几何规则判断手势，不需要额外的分类模型。

```javascript
// 判断某根手指是否伸直
// 原理：指尖 y 坐标 < 掌指关节 y 坐标（屏幕向下为正，伸直时指尖在上方）
function isFingerExtended(keypoints, fingerName) {
  const kp = name => keypoints.find(k => k.name === name);
  const tip = kp(`${fingerName}_tip`);
  const mcp = kp(`${fingerName}_mcp`);
  if (!tip || !mcp) return false;
  return tip.y < mcp.y - 0.03; // 留 3% 的容差
}

function isThumbExtended(keypoints) {
  const kp = name => keypoints.find(k => k.name === name);
  const tip = kp('thumb_tip');
  const cmc = kp('thumb_cmc');
  if (!tip || !cmc) return false;
  // 拇指水平伸展：tip.x 远离手掌中心
  const wrist = kp('wrist');
  return Math.abs(tip.x - wrist.x) > 0.06;
}

// 识别常见手势
function recognizeGesture(keypoints) {
  const index  = isFingerExtended(keypoints, 'index');
  const middle = isFingerExtended(keypoints, 'middle');
  const ring   = isFingerExtended(keypoints, 'ring');
  const pinky  = isFingerExtended(keypoints, 'pinky');
  const thumb  = isThumbExtended(keypoints);

  if (!index && !middle && !ring && !pinky) return 'fist';       // 握拳
  if (index && middle && ring && pinky)     return 'open';       // 张开手掌
  if (index && !middle && !ring && !pinky)  return 'point';      // 指向
  if (index && middle && !ring && !pinky)   return 'peace';      // 剪刀手 ✌️
  if (thumb && !index && !middle && !ring && !pinky) return 'thumb_up'; // 点赞
  if (!thumb && index && !middle && !ring && pinky)  return 'rock';     // 摇滚 🤘
  return 'unknown';
}

// 使用方式
async function detectGesture(videoElement) {
  const base64 = frameToBase64(videoElement);
  const data   = await npuInfer('/detect/hand', base64);
  if (!data?.detected) return null;

  const gesture = recognizeGesture(data.keypoints);
  console.log(`手势: ${gesture}，推理耗时: ${data.inferMs}ms`);
  return gesture;
}
```

---

### 示例四：同时运行人体 + 手部推理

两个接口并发请求，总延迟取决于较慢的那个。

```javascript
async function detectAll(videoElement) {
  const base64 = frameToBase64(videoElement);

  // 并发发送两个请求
  const [poseData, handData] = await Promise.all([
    npuInfer('/detect',      base64),
    npuInfer('/detect/hand', base64)
  ]);

  return {
    pose: poseData?.detected ? poseData.keypoints : null,
    hand: handData?.detected ? handData.keypoints : null,
    inferMs: Math.max(poseData?.inferMs ?? 0, handData?.inferMs ?? 0)
  };
}
```

---

## 开发注意事项

### 1. 必须在 WebLauncher 内运行

`localhost:8080` 只有在 WebLauncher APP 加载的网页里才能访问。普通浏览器或 Chrome 打开的网页无法连接，因为 localhost 指向设备本身而不是 PC。

### 2. 推理频率建议

不要每帧都等推理结果，应该用"上一帧推理完成后再发下一帧"的模式（即上面示例中的 `busy` 锁），避免请求积压导致延迟越来越高。

```javascript
// 推荐：推理完成后再发下一帧
let busy = false;
function loop() {
  if (!busy) {
    busy = true;
    infer().then(() => { busy = false; });
  }
  requestAnimationFrame(loop);
}

// 不推荐：每帧都发请求（会积压）
function loop() {
  infer(); // 不等待结果直接发下一帧
  requestAnimationFrame(loop);
}
```

### 3. 图像质量 vs 速度

`canvas.toDataURL('image/jpeg', quality)` 的 quality 参数影响传输大小：

| quality | 文件大小 | 推荐场景 |
|---------|---------|---------|
| 0.9 | 较大 | 精度要求高 |
| 0.8 | 中等 | **推荐默认值** |
| 0.6 | 较小 | 追求速度、弱网 |

### 4. 坐标系说明

所有坐标都是**归一化值（0~1）**，以输入图像的左上角为原点，x 向右，y 向下。

```javascript
// 还原为像素坐标
const pixelX = kp.x * canvas.width;
const pixelY = kp.y * canvas.height;
```

### 5. 手部推理延迟

`/detect/hand` 是两阶段推理（先检测后提取），单次耗时约 30~50ms，比人体姿态慢一倍。如果只需要手势识别不需要实时骨架，可以降低推理频率（每 3 帧推理一次）：

```javascript
let frameCount = 0;
function loop() {
  frameCount++;
  if (frameCount % 3 === 0 && !busy) {
    // 每 3 帧推理一次手部
    busy = true;
    npuInfer('/detect/hand', frameToBase64(video))
      .then(data => { handleResult(data); busy = false; });
  }
  requestAnimationFrame(loop);
}
```

### 6. NPU 不可用时的降级处理

建议页面加载时先 check，NPU 不可用时回退到 MediaPipe（精度相同但性能较低）。

```javascript
const npuAvailable = await checkNpu();

if (npuAvailable) {
  // 使用 NPU
  startNpuLoop();
} else {
  // 回退到 MediaPipe
  startMediaPipeLoop();
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
| 手部推理失败 | `{"detected":false,"error":"detection failed"}` |
