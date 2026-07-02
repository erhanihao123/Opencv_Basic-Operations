# Web端图像标注工具 - 设计文档

## 1. 架构设计

### 1.1 整体架构
采用前后端分离的模块化架构，前端使用 Streamlit 构建用户界面，后端使用 OpenCV 封装图像处理逻辑。

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit 前端                         │
│  ┌───────────────┐    ┌──────────────────────────────────┐  │
│  │   侧边栏      │    │            主区域                │  │
│  │  - 模块选择   │    │  ┌───────────┐  ┌───────────┐    │  │
│  │  - 功能选择   │    │  │   原图     │  │ 处理结果   │    │  │
│  │  - 参数调节   │    │  │   显示区   │  │   显示区   │    │  │
│  │  - 暂存面板   │    │  └───────────┘  └───────────┘    │  │
│  │  - 下载按钮   │    │                                  │  │
│  └───────────────┘    └──────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ API 调用
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend 后端                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              image_processor.py                        │ │
│  │  ┌───────────┐ ┌───────────┐ ┌─────────────────────┐   │ │
│  │  │ 检测模块  │ │ 基础处理  │ │ 图像增强与形态学     │   │ │
│  │  │ Canny/人脸│ │ 色彩/几何 │ │ 形态学/直方图       │   │ │
│  │  │ /目标检测 │ │ /阈值/滤波│ │                     │   │ │
│  │  └───────────┘ └───────────┘ └─────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块划分

| 模块 | 文件路径 | 职责描述 |
| :--- | :--- | :--- |
| 主入口 | `streamlit_app.py` | Streamlit 应用入口，负责 UI 渲染和用户交互 |
| 图像处理 | `backend/image_processor.py` | 封装 OpenCV 图像处理逻辑（检测、基础处理、增强） |
| 单元测试 | `test/test_processor.py` | 测试图像处理基础函数 |
| 部署配置 | `deploy/` | Docker、Nginx、Docker Compose 配置文件 |

---

## 2. 页面布局设计

### 2.1 整体布局

```
┌─────────────────────────────────────────────────────────────────┐
│                     顶部标题栏                                   │
│         Web端图像标注工具 - Image Annotation Tool                │
├──────────────────────┬───────────────────────────────────────────┤
│                      │                                           │
│                      │                                           │
│   ┌───────────────┐  │   ┌──────────────────┐ ┌───────────────┐  │
│   │               │  │   │                  │ │               │  │
│   │   文件上传    │  │   │     原图显示     │ │   处理结果    │  │
│   │               │  │   │                  │ │               │  │
│   │   模块选择    │  │   │                  │ │               │  │
│   │   (按钮组)    │  │   │                  │ │               │  │
│   │               │  │   │                  │ │               │  │
│   │   功能选择    │  │   │                  │ │               │  │
│   │   (单选)      │  │   │                  │ │               │  │
│   │               │  │   └──────────────────┘ └───────────────┘  │
│   │   参数调节    │  │                                           │
│   │   滑动条      │  │                                           │
│   │               │  │   ┌──────────────────────────────────┐   │
│   │   暂存面板    │  │   │         暂存图片列表              │   │
│   │               │  │   └──────────────────────────────────┘   │
│   └───────────────┘  │                                           │
│                      │                                           │
│         侧边栏(25%)  │              主区域(75%)                   │
└──────────────────────┴───────────────────────────────────────────┘
```

### 2.2 侧边栏布局

| 区域 | 组件 | 说明 |
| :--- | :--- | :--- |
| 配置面板 | `st.header` | 语言选择、文件上传 |
| 文件上传区 | `st.file_uploader` | 支持 JPG/PNG 格式 |
| 模块选择区 | `st.button` | 三个模块切换按钮 |
| 功能选择区 | `st.radio` | 根据模块动态显示功能选项 |
| 参数调节区 | `st.slider` | 根据选择的功能动态显示对应参数 |
| 暂存面板 | `st.columns` | 显示暂存图片列表 |
| 下载按钮区 | `st.download_button` | 下载处理后的图片 |

### 2.3 参数调节区设计

**检测模块**

**边缘检测 (Canny)**：
- 阈值1 (Threshold 1)：滑动条，范围 0-255，默认 100
- 阈值2 (Threshold 2)：滑动条，范围 0-255，默认 200

**人脸检测 (Haar Cascades)**：
- 缩放因子 (Scale Factor)：滑动条，范围 1.01-1.2，默认 1.05
- 最小邻域数 (Min Neighbors)：滑动条，范围 1-15，默认 3
- 最小人脸尺寸 (Min Size)：滑动条，范围 30-150，默认 30
- 要求检测到眼睛 (Require Eyes)：复选框，默认 False
- 要求肤色检测 (Require Skin)：复选框，默认 False
- 级联分类器置信度 (Confidence)：滑动条，范围 0.0-1.0，默认 0.3

**目标检测 (DNN)**：
- 置信度阈值 (Confidence)：滑动条，范围 0.1-0.9，默认 0.3

**基础图像处理模块**

**色彩空间转换**：
- 目标色彩空间：单选按钮，灰度/HSV/RGB

**几何变换**：
- 变换类型：单选按钮，缩放/旋转/平移/仿射变换
- 缩放比例：滑动条，范围 0.1-3.0，默认 1.0
- 旋转角度：滑动条，范围 0-360，默认 45
- 水平偏移：滑动条，范围 -200-200，默认 50
- 垂直偏移：滑动条，范围 -200-200，默认 50

**阈值化**：
- 阈值类型：单选按钮，二值化/反二值化/截断/归零/自适应/Otsu
- 阈值：滑动条，范围 0-255，默认 127
- 块大小：滑动条，范围 3-31，默认 11
- 常数C：滑动条，范围 -20-20，默认 2

**平滑/滤波**：
- 滤波类型：单选按钮，高斯模糊/中值滤波/均值模糊/双边滤波
- 核大小：滑动条，范围 1-21，默认 5
- 邻域直径：滑动条，范围 1-25，默认 9
- 颜色sigma：滑动条，范围 1-200，默认 75
- 空间sigma：滑动条，范围 1-200，默认 75

**图像增强与形态学模块**

**形态学变换**：
- 运算类型：单选按钮，腐蚀/膨胀/开运算/闭运算
- 核大小：滑动条，范围 1-15，默认 5
- 迭代次数：滑动条，范围 1-5，默认 1

**直方图均衡化**：
- 无参数

---

## 3. 数据流设计

### 3.1 数据流转流程

```
用户上传图片
    │
    ▼
Streamlit 接收图片文件
    │
    ▼
转换为 OpenCV 格式 (BGR)
    │
    ▼
根据用户选择调用对应的处理函数
    │
    ├─→ 检测模块
    │   ├─→ canny_edge_detection()
    │   ├─→ face_detection()
    │   └─→ object_detection()
    │
    ├─→ 基础处理模块
    │   ├─→ color_space_conversion()
    │   ├─→ geometric_transform()
    │   ├─→ image_threshold()
    │   └─→ image_smoothing()
    │
    └─→ 增强模块
        ├─→ morphological_transform()
        └─→ histogram_equalization()
    │
    ▼
返回处理后的图像和标注信息
    │
    ▼
转换为 Streamlit 显示格式 (RGB)
    │
    ▼
在右侧结果区显示处理后的图片
```

### 3.2 Session State 管理

Streamlit 使用 `st.session_state` 管理应用状态：

| 状态变量 | 类型 | 说明 |
| :--- | :--- | :--- |
| `language` | str | 当前语言（zh/en） |
| `cv_image` | numpy.ndarray | OpenCV 格式的图片 |
| `processed_image` | numpy.ndarray | 处理后的图片 |
| `processed_image_bytes` | bytes | 处理后图片的字节流 |
| `active_module` | str | 当前激活的模块（detection/basic/enhance） |
| `selected_function` | str | 当前选择的处理功能 |
| `saved_images` | list | 暂存图片列表 |
| `using_temp_image` | bool | 是否使用暂存图片作为原图 |
| `processor` | ImageProcessor | 图像处理实例 |

---

## 4. 模型加载策略

### 4.1 Haar Cascade 模型
- 使用 OpenCV 自带的预训练模型
- 默认路径：`cv2.data.haarcascades`
- 模型文件：多种级联分类器（frontalface_alt2, frontalface_default, frontalface_alt, frontalface_alt_tree）
- 加载策略：多分类器融合，提高检测准确率

### 4.2 DNN 目标检测模型
- 使用 MobileNet-SSD 预训练模型
- 模型文件：
  - `MobileNetSSD_deploy.prototxt`
  - `MobileNetSSD_deploy.caffemodel`
- 加载策略：
  1. 检查本地是否存在模型文件
  2. 如果不存在，从多源下载（GitHub + jsDelivr）
  3. 下载失败自动重试（最多3次）
  4. 使用 `cv2.dnn.readNetFromCaffe()` 加载模型
  5. 启动时预加载，提升首次请求响应速度

### 4.3 模型存储位置

```
backend/
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
└── image_processor.py
```

---

## 5. 接口设计

### 5.1 图像处理接口

| 函数名 | 参数 | 返回值 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `load_image(file_bytes)` | file_bytes: bytes | numpy.ndarray | 加载图片并转换为 OpenCV 格式 |
| `canny_edge_detection(image, threshold1, threshold2)` | image: ndarray, threshold1: int, threshold2: int | numpy.ndarray | Canny 边缘检测 |
| `face_detection(image, **kwargs)` | image: ndarray, min_neighbors: int, min_size: tuple, scale_factor: float, require_eyes: bool, require_skin: bool, confidence_threshold: float | numpy.ndarray | Haar 人脸检测并标注 |
| `object_detection(image, confidence_threshold)` | image: ndarray, confidence_threshold: float | numpy.ndarray | DNN 目标检测并标注 |
| `color_space_conversion(image, target_space)` | image: ndarray, target_space: str | numpy.ndarray | 色彩空间转换 |
| `geometric_transform(image, transform_type, **kwargs)` | image: ndarray, transform_type: str, **kwargs | numpy.ndarray | 几何变换 |
| `image_threshold(image, threshold_type, **kwargs)` | image: ndarray, threshold_type: str, **kwargs | numpy.ndarray | 图像阈值化 |
| `image_smoothing(image, filter_type, **kwargs)` | image: ndarray, filter_type: str, **kwargs | numpy.ndarray | 图像平滑/滤波 |
| `morphological_transform(image, morph_type, **kwargs)` | image: ndarray, morph_type: str, **kwargs | numpy.ndarray | 形态学变换 |
| `histogram_equalization(image)` | image: ndarray | numpy.ndarray | 直方图均衡化 |
| `image_to_bytes(image)` | image: numpy.ndarray | bytes | 将图片转换为可下载的字节流 |
| `bgr_to_rgb(image)` | image: numpy.ndarray | numpy.ndarray | BGR 转 RGB |
| `process_image(image, function_type, **kwargs)` | image: ndarray, function_type: str, **kwargs | numpy.ndarray | 统一图像处理入口 |

---

## 6. 设计原则

### 6.1 UI 设计原则
- **简洁性**：界面布局清晰，操作流程直观
- **一致性**：使用统一的配色方案和交互模式
- **响应式**：适应不同屏幕尺寸
- **模块化**：功能按模块组织，便于用户理解和使用

### 6.2 代码设计原则
- **模块化**：后端逻辑与前端展示分离
- **可复用性**：图像处理函数独立封装，便于测试和复用
- **健壮性**：处理异常情况，如图片加载失败、模型文件缺失等
- **可观测性**：完善的日志记录机制，便于问题排查

---

## 7. 技术细节

### 7.1 图片格式转换
- 上传的图片 → OpenCV (BGR)：`cv2.imdecode()`
- OpenCV (BGR) → Streamlit 显示 (RGB)：`cv2.cvtColor()`
- 处理后的图片 → 下载格式：`cv2.imencode()`

### 7.2 标注可视化
- 边缘检测：直接显示边缘图像
- 人脸检测：在原图上绘制绿色矩形框
- 目标检测：在原图上绘制红色矩形框，并标注类别名称和置信度

### 7.3 实时更新机制
- 使用 Streamlit 的 `st.session_state` 存储参数状态
- 参数变化时自动触发重新运行，实现实时预览

### 7.4 暂存功能实现
- 使用 `st.session_state.saved_images` 存储暂存图片列表
- 支持将暂存图片作为新原图继续处理
- 支持清空暂存列表

---

## 8. 部署设计

### 8.1 Docker 部署
- 使用 Python 3.9-slim 作为基础镜像
- 安装依赖时使用清华镜像源
- 直接复制模型文件，避免从 GitHub 下载
- 添加健康检查

### 8.2 Docker Compose 部署
- Streamlit 服务：运行图像处理应用
- Nginx 服务：反向代理，处理 HTTP 请求
- 专用网络：隔离容器通信
- 数据持久化：模型文件和日志挂载到宿主机

### 8.3 Nginx 配置
- 反向代理指向 Streamlit 服务
- 配置 WebSocket 支持
- 添加安全头
- 配置静态资源缓存