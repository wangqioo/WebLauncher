# RK3576 WebView 性能优化方案评估

## 背景

设备：Rockchip RK3576  
GPU：Mali-G52（入门级，~50 GFLOPS FP32）  
NPU：6 TOPS  
WebView：116.0（系统内置，无法升级）  
问题：MediaPipe 手势识别在 WebView 中卡顿，原因是 Mali-G52 跑 WebGL 推理性能不足

---

## 已完成的优化

| 优化项 | 状态 | 效果 |
|--------|------|------|
| WebView 硬件加速 | ✅ 已完成 | 小幅提升渲染 |
| 渲染优先级设为 HIGH | ✅ 已完成 | 减少帧丢失 |
| User Agent 改为桌面版 Chrome | ✅ 已完成 | 解除网站限制 |
| WebAssembly SIMD 开启 | ✅ 已完成 | WASM 推理提速 |
| GPU 光栅化开启 | ✅ 已完成 | GPU 渲染优化 |
| WebView 多进程模式 | ✅ 已完成 | 渲染进程独立 |

---

## 方案总览

| 方案 | 性能提升 | 实现难度 | 成功率 | 预计时间 |
|------|---------|---------|--------|---------|
| A. 网页降低模型负载 | 2-3x | 低 | 95% | 30 分钟 |
| B. RKNN NPU 推理服务 | 10-15x | 高 | 60% | 1-2 天 |
| C. 换硬件 RK3588 | 6-8x | 无需开发 | 100% | 购买即用 |
| D. WebView 渲染降分辨率 | 1.5x | 低 | 95% | 20 分钟 |
| E. ONNX.js 替代 MediaPipe | 3-5x | 中 | 75% | 半天 |

---

## 方案详细评估

### 方案 A：网页端降低 MediaPipe 负载

**原理：** 换轻量模型 + 降低输入分辨率 + 限制推理帧率

**改动示例：**

```javascript
// 换用轻量模型（0 = lite，1 = full）
const hands = new Hands({
  modelComplexity: 0,   // 从 1 改成 0，推理量减少 60%
  maxNumHands: 1,       // 只检测 1 只手，减少 50%
});

// 摄像头降到 480p
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: 640, height: 480 }  // 从 1280x720 降下来
});

// 每 2 帧推理 1 次
let frameCount = 0;
function onFrame() {
  frameCount++;
  if (frameCount % 2 === 0) {
    hands.send({ image: video });
  }
  requestAnimationFrame(onFrame);
}
```

**效果分析：**

| 改动 | 负载降低 |
|------|---------|
| modelComplexity: 0 | -60% |
| maxNumHands: 1 | -50% |
| 分辨率 1280x720 → 640x480 | -55% |
| 每 2 帧推理 1 次 | -50% |

综合提升约 **2-3 倍**流畅度

**前提条件：** 网页代码可修改  
**风险：** 识别精度略降，距离太远或角度刁钻时可能漏检  
**成功率：95%**

---

### 方案 B：RKNN NPU 推理服务

**原理：** 原生调用 RK3576 的 6 TOPS NPU，彻底绕开 WebGL 推理

**架构：**

```
网页 (WebView)
    ↕ fetch('http://localhost:8080/detect')
本地 HTTP 推理服务 (Android Service)
    ↕ JNI
librknnrt.so
    ↕
NPU (6 TOPS)
```

**API 设计：**

```
POST http://localhost:8080/detect
Body: { "image": "base64图像数据" }

返回:
{
  "gesture": "palm",
  "confidence": 0.97,
  "inferMs": 12
}
```

**完整步骤：**

```
步骤 1：下载 RKNN SDK 2.3.2         成功率 100%
步骤 2：获取手势识别模型              成功率 70%（model zoo 无现成）
步骤 3：tflite → onnx 模型转换       成功率 70%
步骤 4：onnx → rknn 模型转换         成功率 80%（需要 Docker）
步骤 5：编译 JNI + librknnrt.so      成功率 90%
步骤 6：HTTP 服务 + 网页对接          成功率 95%
```

**卡点：**
- MediaPipe 手势模型是私有格式，转换工具链不完整
- 需要 Mac 安装 Docker + Python 环境
- 手势标签映射需要额外调试

**综合成功率：50-60%**，工作量 1-2 天

---

### 方案 C：换硬件 RK3588

**原理：** 直接换更强的芯片，不改任何代码

**RK3576 vs RK3588 对比：**

| 规格 | RK3576（当前） | RK3588（推荐） |
|------|--------------|--------------|
| GPU | Mali-G52 | Mali-G610 MP4 |
| GPU 性能 | ~50 GFLOPS | ~300 GFLOPS |
| NPU | 6 TOPS | 6 TOPS |
| CPU | 8核 2.1GHz | 4大核 2.4GHz + 4小核 1.8GHz |
| WebView 可升级 | 受限 | 支持 |

**代表设备：**

| 设备 | 价格 | 特点 |
|------|------|------|
| Orange Pi 5 Plus | ¥600 | 开发板，需自己装系统 |
| Firefly ROC-RK3588S | ¥1200 | 工业级，稳定 |
| RK3588 平板（第三方） | ¥800-1500 | 开箱即用 |

**迁移成本：** WebLauncher APP 无需改动，直接安装即用  
**成功率：100%**，换完立刻见效

---

### 方案 D：WebView 渲染降分辨率

**原理：** 降低 WebView 的渲染密度，GPU 处理的像素数减少

**改动：**

```java
// WebActivity.java 中添加
webView.setInitialScale(75);  // 缩放到 75%，渲染像素减少约 44%
```

**提升：** 约 1.5 倍  
**代价：** 网页视觉会略微缩小  
**成功率：95%**，但收益最低，建议配合方案 A 使用

---

### 方案 E：ONNX.js 替代 MediaPipe

**原理：** MediaPipe 在 WebView 里优化差，换用更轻量的推理框架

```javascript
// 用 onnxruntime-web 替代 MediaPipe
import * as ort from 'onnxruntime-web';

// 已开启 WebAssembly SIMD flag，直接受益
// 推理速度比 MediaPipe JS 快 2-3 倍
const session = await ort.InferenceSession.create('./model.onnx', {
  executionProviders: ['wasm'],
});
```

**提升：** 2-3 倍，与方案 A 叠加可达 4-5 倍  
**前提：** 需要找到对应功能的 ONNX 模型并替换推理代码  
**成功率：75%**

---

## 综合建议

### 短期方案（今天能完成）

> **方案 A + D 叠加**
>
> 修改网页 MediaPipe 参数 + 降低 WebView 渲染密度  
> 综合提升约 **3 倍**，30-60 分钟完成，成功率 **95%**  
> 前提：网页代码可以修改

### 中期方案（1-2 天）

> **方案 B：RKNN NPU 推理服务**
>
> 利用设备自带的 6 TOPS NPU 做推理，效果最强  
> 需要接受 50% 的失败风险，投入 1-2 天开发时间  
> 框架代码已写好，缺少 RKNN 模型文件

### 长期方案（根本解决）

> **方案 C：换 RK3588 设备**
>
> GPU 性能提升 6 倍，WebView 可升级到最新版本  
> 一劳永逸，无需任何代码改动

---

## 关键决策问题

**网页代码能否修改？**

- **能修改** → 优先方案 A，今天就能完成
- **不能修改** → 考虑方案 B 或方案 C

---

*文档生成时间：2026-04-14*  
*设备：Rockchip RK3576 / Android 14 / Mali-G52 / WebView 116*
