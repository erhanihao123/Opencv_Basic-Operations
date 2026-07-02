# Web端图像标注工具 - 接口文档

## 1. 概述

本文档描述了 `backend/image_processor.py` 中 `ImageProcessor` 类的所有公开接口。

## 2. 类定义

### 2.1 ImageProcessor

图像处理核心类，封装了边缘检测、人脸检测、目标检测以及丰富的图像处理功能。

```python
class ImageProcessor:
    def __init__(self)
```

**初始化说明**：
- 自动创建模型目录
- 加载 Haar 级联人脸检测模型（多分类器融合）
- 预加载 MobileNet-SSD 目标检测模型（如需下载则自动下载）
- 初始化日志系统

---

## 3. 接口详细说明

### 3.1 load_image

**功能描述**：加载图片文件并转换为 OpenCV 格式

**方法签名**：
```python
def load_image(self, file_bytes: bytes) -> numpy.ndarray or None
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| file_bytes | bytes | 图片文件的字节数据 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | OpenCV BGR 格式的图像数组，形状为 (height, width, 3) |
| None | 加载失败时返回 |

**示例**：
```python
with open("image.jpg", "rb") as f:
    file_bytes = f.read()

processor = ImageProcessor()
image = processor.load_image(file_bytes)
print(image.shape)  # (480, 640, 3)
```

---

### 3.2 canny_edge_detection

**功能描述**：使用 Canny 算法进行边缘检测

**方法签名**：
```python
def canny_edge_detection(self, image: numpy.ndarray, threshold1: int, threshold2: int) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| threshold1 | int | 低阈值，范围 0-255 |
| threshold2 | int | 高阈值，范围 0-255 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 边缘检测结果图像（BGR 格式），边缘为白色 |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
edges = processor.canny_edge_detection(image, threshold1=100, threshold2=200)
```

---

### 3.3 face_detection

**功能描述**：使用 Haar 级联分类器检测人脸，支持多分类器融合、眼睛检测、肤色过滤

**方法签名**：
```python
def face_detection(self, image: numpy.ndarray, min_neighbors: int = 6, min_size: tuple = (50, 50),
                   scale_factor: float = 1.1, require_eyes: bool = True, require_skin: bool = True,
                   confidence_threshold: float = 0.5) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| image | numpy.ndarray | - | 输入图像（BGR 格式） |
| min_neighbors | int | 6 | 每个候选矩形应保留的邻近数 |
| min_size | tuple | (50, 50) | 最小检测尺寸 |
| scale_factor | float | 1.1 | 图像缩放比例 |
| require_eyes | bool | True | 是否要求检测到眼睛 |
| require_skin | bool | True | 是否要求肤色检测 |
| confidence_threshold | float | 0.5 | 级联分类器置信度 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 在原图上标注了人脸位置的图像（BGR 格式），人脸用绿色矩形框标注 |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
result = processor.face_detection(image, min_neighbors=3, min_size=(30, 30), scale_factor=1.05)
```

---

### 3.4 object_detection

**功能描述**：使用 MobileNet-SSD 模型进行目标检测

**方法签名**：
```python
def object_detection(self, image: numpy.ndarray, confidence_threshold: float = 0.3) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| confidence_threshold | float | 置信度阈值，范围 0.1-0.9，低于此值的检测结果将被过滤 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 在原图上标注了检测目标的图像（BGR 格式），目标用红色矩形框标注，并显示类别名称和置信度 |

**支持的目标类别**：
```
0: background
1: aeroplane
2: bicycle
3: bird
4: boat
5: bottle
6: bus
7: car
8: cat
9: chair
10: cow
11: diningtable
12: dog
13: horse
14: motorbike
15: person
16: pottedplant
17: sheep
18: sofa
19: train
20: tvmonitor
```

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
result = processor.object_detection(image, confidence_threshold=0.5)
```

---

### 3.5 color_space_conversion

**功能描述**：色彩空间转换

**方法签名**：
```python
def color_space_conversion(self, image: numpy.ndarray, target_space: str) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| target_space | str | 目标色彩空间："gray"、"hsv"、"rgb" |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 转换后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 转换为灰度
gray_image = processor.color_space_conversion(image, "gray")

# 转换为 HSV
hsv_image = processor.color_space_conversion(image, "hsv")

# 转换为 RGB
rgb_image = processor.color_space_conversion(image, "rgb")
```

---

### 3.6 geometric_transform

**功能描述**：几何变换

**方法签名**：
```python
def geometric_transform(self, image: numpy.ndarray, transform_type: str, **kwargs) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| transform_type | str | 变换类型："resize"、"rotate"、"translate"、"affine" |
| **kwargs | dict | 额外参数，根据变换类型不同而变化 |

**kwargs 参数说明**：
| transform_type | 可选 kwargs |
| :--- | :--- |
| "resize" | scale (float, 默认 1.0) |
| "rotate" | angle (int, 默认 0) |
| "translate" | tx (int, 默认 0), ty (int, 默认 0) |
| "affine" | 无额外参数 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 变换后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 缩放（0.5倍）
resized = processor.geometric_transform(image, "resize", scale=0.5)

# 旋转（45度）
rotated = processor.geometric_transform(image, "rotate", angle=45)

# 平移
translated = processor.geometric_transform(image, "translate", tx=50, ty=30)

# 仿射变换
affined = processor.geometric_transform(image, "affine")
```

---

### 3.7 image_threshold

**功能描述**：图像阈值化

**方法签名**：
```python
def image_threshold(self, image: numpy.ndarray, threshold_type: str, **kwargs) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| threshold_type | str | 阈值类型："binary"、"binary_inv"、"truncate"、"to_zero"、"adaptive"、"otsu" |
| **kwargs | dict | 额外参数，根据阈值类型不同而变化 |

**kwargs 参数说明**：
| threshold_type | 可选 kwargs |
| :--- | :--- |
| "binary" | thresh (int, 默认 127) |
| "binary_inv" | thresh (int, 默认 127) |
| "truncate" | thresh (int, 默认 127) |
| "to_zero" | thresh (int, 默认 127) |
| "adaptive" | block_size (int, 默认 11), C (int, 默认 2) |
| "otsu" | 无额外参数 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 阈值化后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 二值化
binary = processor.image_threshold(image, "binary", thresh=127)

# 自适应阈值
adaptive = processor.image_threshold(image, "adaptive", block_size=11, C=2)

# Otsu 自动阈值
otsu = processor.image_threshold(image, "otsu")
```

---

### 3.8 image_smoothing

**功能描述**：图像平滑/滤波

**方法签名**：
```python
def image_smoothing(self, image: numpy.ndarray, filter_type: str, **kwargs) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| filter_type | str | 滤波类型："gaussian"、"median"、"blur"、"bilateral" |
| **kwargs | dict | 额外参数，根据滤波类型不同而变化 |

**kwargs 参数说明**：
| filter_type | 可选 kwargs |
| :--- | :--- |
| "gaussian" | ksize (int, 默认 5) |
| "median" | ksize (int, 默认 5) |
| "blur" | ksize (int, 默认 5) |
| "bilateral" | d (int, 默认 9), sigma_color (int, 默认 75), sigma_space (int, 默认 75) |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 滤波后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 高斯模糊
gaussian = processor.image_smoothing(image, "gaussian", ksize=5)

# 中值滤波
median = processor.image_smoothing(image, "median", ksize=5)

# 双边滤波
bilateral = processor.image_smoothing(image, "bilateral", d=9, sigma_color=75, sigma_space=75)
```

---

### 3.9 morphological_transform

**功能描述**：形态学变换

**方法签名**：
```python
def morphological_transform(self, image: numpy.ndarray, morph_type: str, **kwargs) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| morph_type | str | 运算类型："erode"、"dilate"、"opening"、"closing" |
| **kwargs | dict | 额外参数 |

**kwargs 参数说明**：
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| kernel_size | int | 5 | 核大小 |
| iterations | int | 1 | 迭代次数 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 形态学变换后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 腐蚀
eroded = processor.morphological_transform(image, "erode", kernel_size=5, iterations=1)

# 膨胀
dilated = processor.morphological_transform(image, "dilate", kernel_size=5, iterations=1)

# 开运算
opening = processor.morphological_transform(image, "opening", kernel_size=5, iterations=1)

# 闭运算
closing = processor.morphological_transform(image, "closing", kernel_size=5, iterations=1)
```

---

### 3.10 histogram_equalization

**功能描述**：直方图均衡化，增强图像对比度

**方法签名**：
```python
def histogram_equalization(self, image: numpy.ndarray) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 直方图均衡化后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
equalized = processor.histogram_equalization(image)
```

---

### 3.11 resize_image

**功能描述**：调整图像大小，保持宽高比

**方法签名**：
```python
def resize_image(self, image: numpy.ndarray, max_dimension: int = 1024) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| max_dimension | int | 最大维度（宽或高），默认为 1024 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 调整大小后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
resized = processor.resize_image(image, max_dimension=800)
```

---

### 3.12 image_to_bytes

**功能描述**：将图像转换为可下载的字节流

**方法签名**：
```python
def image_to_bytes(self, image: numpy.ndarray) -> bytes or None
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| bytes | PNG 格式的图像字节流 |
| None | 编码失败时返回 |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)
result = processor.canny_edge_detection(image, 100, 200)
img_bytes = processor.image_to_bytes(result)

with open("output.png", "wb") as f:
    f.write(img_bytes)
```

---

### 3.13 bgr_to_rgb

**功能描述**：将图像从 BGR 格式转换为 RGB 格式

**方法签名**：
```python
def bgr_to_rgb(self, image: numpy.ndarray) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | RGB 格式的图像 |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)  # BGR 格式
rgb_image = processor.bgr_to_rgb(image)  # RGB 格式
```

---

### 3.14 is_object_model_loaded

**功能描述**：检查目标检测模型是否已加载

**方法签名**：
```python
def is_object_model_loaded(self) -> bool
```

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| bool | True 表示模型已加载，False 表示未加载 |

---

### 3.15 get_supported_classes

**功能描述**：获取目标检测支持的类别列表

**方法签名**：
```python
def get_supported_classes(self) -> list
```

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| list | 支持的类别名称列表（不包含 background） |

---

### 3.16 process_image

**功能描述**：统一的图像处理入口，根据功能类型调用对应的处理方法

**方法签名**：
```python
def process_image(self, image: numpy.ndarray, function_type: str, **kwargs) -> numpy.ndarray
```

**参数**：
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image | numpy.ndarray | 输入图像（BGR 格式） |
| function_type | str | 处理功能类型 |
| **kwargs | dict | 额外参数，根据 function_type 不同而变化 |

**function_type 及 kwargs 参数说明**：
| function_type | 功能描述 | 可选 kwargs |
| :--- | :--- | :--- |
| "original" | 原图（不处理） | 无 |
| "canny" | Canny 边缘检测 | threshold1 (int), threshold2 (int) |
| "face" | Haar 人脸检测 | min_neighbors (int), min_size (int/tuple), scale_factor (float), require_eyes (bool), require_skin (bool), confidence_threshold (float) |
| "object" | DNN 目标检测 | confidence (float) |
| "color_space" | 色彩空间转换 | target_space (str) |
| "geometric" | 几何变换 | transform_type (str), scale (float), angle (int), tx (int), ty (int) |
| "threshold" | 图像阈值化 | threshold_type (str), thresh (int), block_size (int), C (int) |
| "smoothing" | 图像平滑/滤波 | filter_type (str), ksize (int), d (int), sigma_color (int), sigma_space (int) |
| "morphology" | 形态学变换 | morph_type (str), kernel_size (int), iterations (int) |
| "histogram" | 直方图均衡化 | 无 |

**返回值**：
| 类型 | 说明 |
| :--- | :--- |
| numpy.ndarray | 处理后的图像（BGR 格式） |

**示例**：
```python
processor = ImageProcessor()
image = processor.load_image(file_bytes)

# 边缘检测
result = processor.process_image(image, "canny", threshold1=100, threshold2=200)

# 人脸检测
result = processor.process_image(image, "face", min_neighbors=3, require_eyes=True)

# 目标检测
result = processor.process_image(image, "object", confidence=0.5)

# 色彩空间转换
result = processor.process_image(image, "color_space", target_space="gray")

# 几何变换
result = processor.process_image(image, "geometric", transform_type="resize", scale=0.5)

# 阈值化
result = processor.process_image(image, "threshold", threshold_type="binary", thresh=127)

# 平滑/滤波
result = processor.process_image(image, "smoothing", filter_type="gaussian", ksize=5)

# 形态学变换
result = processor.process_image(image, "morphology", morph_type="erode", kernel_size=5)

# 直方图均衡化
result = processor.process_image(image, "histogram")

# 原图（不处理）
result = processor.process_image(image, "original")
```

---

## 4. 错误处理

| 错误场景 | 处理方式 |
| :--- | :--- |
| 人脸检测模型加载失败 | 记录错误日志，face_detection 返回原图 |
| 目标检测模型加载失败 | 记录错误日志，object_detection 返回原图 |
| 模型文件下载失败 | 记录错误日志，自动重试，object_detection 返回原图 |
| 图片加载失败 | load_image 返回 None |
| 图像编码失败 | image_to_bytes 返回 None |
| 图像处理失败 | 记录错误日志，返回原图 |

---

## 5. 模型文件

### 5.1 人脸检测模型
- 文件名：多种 Haar Cascade 分类器
- 来源：OpenCV 自带
- 路径：`cv2.data.haarcascades`
- 加载方式：自动加载，多分类器融合

### 5.2 目标检测模型
- 文件名：`MobileNetSSD_deploy.prototxt`
- 文件名：`MobileNetSSD_deploy.caffemodel`
- 来源：MobileNet-SSD 官方仓库（GitHub + jsDelivr 镜像）
- 路径：`backend/models/`
- 加载方式：检查本地文件 → 自动下载（多源 + 重试）→ 加载模型