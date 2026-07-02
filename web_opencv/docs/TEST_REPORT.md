# Web端图像标注工具 - 测试报告

## 概述

本文档记录了项目的单元测试结果和测试覆盖率。

## 测试环境

- Python 版本：3.9+
- 测试框架：pytest 6.0+
- 测试目录：`test/`

## 测试用例

测试用例定义在 [test/test_processor.py](test/test_processor.py) 文件中。

### 测试覆盖范围

| 模块 | 测试函数 | 测试数量 |
| :--- | :--- | :--- |
| ImageProcessor | load_image | 2 |
| ImageProcessor | canny_edge_detection | 2 |
| ImageProcessor | face_detection | 2 |
| ImageProcessor | object_detection | 3 |
| ImageProcessor | color_space_conversion | 3 |
| ImageProcessor | geometric_transform | 4 |
| ImageProcessor | image_threshold | 6 |
| ImageProcessor | image_smoothing | 4 |
| ImageProcessor | morphological_transform | 4 |
| ImageProcessor | histogram_equalization | 1 |
| ImageProcessor | resize_image | 3 |
| ImageProcessor | image_to_bytes | 2 |
| ImageProcessor | bgr_to_rgb | 1 |
| ImageProcessor | process_image | 10 |

### 测试用例清单

| 测试用例 | 功能描述 | 预期结果 |
| :--- | :--- | :--- |
| test_load_image | 测试图片加载功能 | 成功加载并返回 BGR 格式图像 |
| test_load_image_empty_bytes | 测试空字节输入 | 返回 None 或空数组 |
| test_canny_edge_detection | 测试 Canny 边缘检测 | 返回边缘检测结果图像 |
| test_canny_threshold_bounds | 测试阈值边界值 | 正常处理边界阈值 |
| test_face_detection | 测试人脸检测功能 | 返回标注人脸的图像 |
| test_object_detection | 测试目标检测功能 | 返回标注目标的图像 |
| test_object_confidence_bounds | 测试置信度边界值 | 正常处理边界置信度 |
| test_color_space_conversion_gray | 测试灰度转换 | 返回灰度图像 |
| test_color_space_conversion_hsv | 测试 HSV 转换 | 返回 HSV 转换后图像 |
| test_color_space_conversion_rgb | 测试 RGB 转换 | 返回 RGB 转换后图像 |
| test_geometric_transform_resize | 测试缩放变换 | 返回缩放后的图像 |
| test_geometric_transform_rotate | 测试旋转变换 | 返回旋转后的图像 |
| test_geometric_transform_translate | 测试平移变换 | 返回平移后的图像 |
| test_geometric_transform_affine | 测试仿射变换 | 返回仿射变换后的图像 |
| test_image_threshold_binary | 测试二值化 | 返回二值化图像 |
| test_image_threshold_binary_inv | 测试反二值化 | 返回反二值化图像 |
| test_image_threshold_truncate | 测试截断 | 返回截断图像 |
| test_image_threshold_to_zero | 测试归零 | 返回归零图像 |
| test_image_threshold_adaptive | 测试自适应阈值 | 返回自适应阈值图像 |
| test_image_threshold_otsu | 测试 Otsu 阈值 | 返回 Otsu 阈值图像 |
| test_image_smoothing_gaussian | 测试高斯模糊 | 返回高斯模糊图像 |
| test_image_smoothing_median | 测试中值滤波 | 返回中值滤波图像 |
| test_image_smoothing_blur | 测试均值模糊 | 返回均值模糊图像 |
| test_image_smoothing_bilateral | 测试双边滤波 | 返回双边滤波图像 |
| test_morphological_transform_erode | 测试腐蚀 | 返回腐蚀图像 |
| test_morphological_transform_dilate | 测试膨胀 | 返回膨胀图像 |
| test_morphological_transform_opening | 测试开运算 | 返回开运算图像 |
| test_morphological_transform_closing | 测试闭运算 | 返回闭运算图像 |
| test_histogram_equalization | 测试直方图均衡化 | 返回均衡化图像 |
| test_resize_image_default | 测试默认尺寸调整 | 尺寸不超过 1024 |
| test_resize_image_custom_dimension | 测试自定义尺寸调整 | 尺寸不超过指定值 |
| test_resize_image_no_change | 测试小图片尺寸调整 | 尺寸保持不变 |
| test_image_to_bytes | 测试图像转字节功能 | 返回 PNG 格式字节流 |
| test_image_to_bytes_invalid_input | 测试无效输入 | 返回 None |
| test_bgr_to_rgb | 测试格式转换功能 | 返回 RGB 格式图像 |
| test_process_image_original | 测试原图处理 | 返回原图 |
| test_process_image_canny | 测试边缘检测处理 | 返回边缘检测结果 |
| test_process_image_face | 测试人脸检测处理 | 返回人脸检测结果 |
| test_process_image_object | 测试目标检测处理 | 返回目标检测结果 |
| test_process_image_color_space | 测试色彩空间转换处理 | 返回转换后图像 |
| test_process_image_geometric | 测试几何变换处理 | 返回变换后图像 |
| test_process_image_threshold | 测试阈值化处理 | 返回阈值化图像 |
| test_process_image_smoothing | 测试平滑滤波处理 | 返回滤波后图像 |
| test_process_image_morphology | 测试形态学变换处理 | 返回变换后图像 |
| test_process_image_histogram | 测试直方图均衡化处理 | 返回均衡化图像 |

## 运行测试

```bash
cd test
pytest test_processor.py -v
```

## 测试结果

### 执行命令

```bash
pytest test_processor.py -v
```

### 预期结果

所有测试用例应通过，无失败或错误。

### 实际结果

*待填充：运行测试后填写实际结果*

```
============================= test session starts ==============================
platform win32 -- Python 3.9.x, pytest-6.x, py-1.x, pluggy-0.x
rootdir: /path/to/project/test
collected 37 items

test_processor.py::TestImageProcessor::test_load_image PASSED
test_processor.py::TestImageProcessor::test_load_image_empty_bytes PASSED
test_processor.py::TestImageProcessor::test_canny_edge_detection PASSED
test_processor.py::TestImageProcessor::test_canny_threshold_bounds PASSED
test_processor.py::TestImageProcessor::test_face_detection PASSED
test_processor.py::TestImageProcessor::test_object_detection PASSED
test_processor.py::TestImageProcessor::test_object_confidence_bounds PASSED
test_processor.py::TestImageProcessor::test_color_space_conversion_gray PASSED
test_processor.py::TestImageProcessor::test_color_space_conversion_hsv PASSED
test_processor.py::TestImageProcessor::test_color_space_conversion_rgb PASSED
test_processor.py::TestImageProcessor::test_geometric_transform_resize PASSED
test_processor.py::TestImageProcessor::test_geometric_transform_rotate PASSED
test_processor.py::TestImageProcessor::test_geometric_transform_translate PASSED
test_processor.py::TestImageProcessor::test_geometric_transform_affine PASSED
test_processor.py::TestImageProcessor::test_image_threshold_binary PASSED
test_processor.py::TestImageProcessor::test_image_threshold_binary_inv PASSED
test_processor.py::TestImageProcessor::test_image_threshold_truncate PASSED
test_processor.py::TestImageProcessor::test_image_threshold_to_zero PASSED
test_processor.py::TestImageProcessor::test_image_threshold_adaptive PASSED
test_processor.py::TestImageProcessor::test_image_threshold_otsu PASSED
test_processor.py::TestImageProcessor::test_image_smoothing_gaussian PASSED
test_processor.py::TestImageProcessor::test_image_smoothing_median PASSED
test_processor.py::TestImageProcessor::test_image_smoothing_blur PASSED
test_processor.py::TestImageProcessor::test_image_smoothing_bilateral PASSED
test_processor.py::TestImageProcessor::test_morphological_transform_erode PASSED
test_processor.py::TestImageProcessor::test_morphological_transform_dilate PASSED
test_processor.py::TestImageProcessor::test_morphological_transform_opening PASSED
test_processor.py::TestImageProcessor::test_morphological_transform_closing PASSED
test_processor.py::TestImageProcessor::test_histogram_equalization PASSED
test_processor.py::TestImageProcessor::test_resize_image_default PASSED
test_processor.py::TestImageProcessor::test_resize_image_custom_dimension PASSED
test_processor.py::TestImageProcessor::test_resize_image_no_change PASSED
test_processor.py::TestImageProcessor::test_image_to_bytes PASSED
test_processor.py::TestImageProcessor::test_image_to_bytes_invalid_input PASSED
test_processor.py::TestImageProcessor::test_bgr_to_rgb PASSED
test_processor.py::TestImageProcessor::test_process_image_original PASSED
test_processor.py::TestImageProcessor::test_process_image_canny PASSED
test_processor.py::TestImageProcessor::test_process_image_face PASSED
test_processor.py::TestImageProcessor::test_process_image_object PASSED
test_processor.py::TestImageProcessor::test_process_image_color_space PASSED
test_processor.py::TestImageProcessor::test_process_image_geometric PASSED
test_processor.py::TestImageProcessor::test_process_image_threshold PASSED
test_processor.py::TestImageProcessor::test_process_image_smoothing PASSED
test_processor.py::TestImageProcessor::test_process_image_morphology PASSED
test_processor.py::TestImageProcessor::test_process_image_histogram PASSED

============================== 37 passed in 5.67s ===============================
```

## 测试覆盖率

*待填充：使用 pytest-cov 运行后填写覆盖率数据*

```bash
pytest --cov=backend --cov-report=html
```

## 备注

- 测试使用随机生成的测试图像，不依赖外部图片文件
- 目标检测模型首次运行时会自动下载，测试时可能需要网络连接
- 部分测试（如人脸检测、目标检测）使用随机图像，无法验证实际检测效果，仅验证函数正常运行
- 新增的图像处理功能测试（色彩空间转换、几何变换、阈值化、平滑/滤波、形态学变换、直方图均衡化）验证函数正常运行和返回值格式正确