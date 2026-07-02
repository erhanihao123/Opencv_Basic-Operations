import streamlit as st
import cv2
import numpy as np
import time

from backend.image_processor import ImageProcessor

CLASS_NAMES_CN = {
    "aeroplane": "飞机", "bicycle": "自行车", "bird": "鸟", "boat": "船",
    "bottle": "瓶子", "bus": "公交车", "car": "汽车", "cat": "猫",
    "chair": "椅子", "cow": "牛", "diningtable": "餐桌", "dog": "狗",
    "horse": "马", "motorbike": "摩托车", "person": "人", "pottedplant": "盆栽",
    "sheep": "羊", "sofa": "沙发", "train": "火车", "tvmonitor": "电视"
}

TEXT = {
    "zh": {
        "title": "Web端图像标注工具",
        "subtitle": "基于 Streamlit 和 OpenCV 的图像标注工具，支持边缘检测、人脸检测和目标检测",
        "config_panel": "配置面板",
        "language": "语言",
        "upload": "上传图片",
        "select_function": "选择处理功能",
        "original": "原图",
        "canny": "边缘检测 (Canny)",
        "face": "人脸检测 (Haar)",
        "object": "目标检测 (DNN)",
        "canny_params": "Canny 参数",
        "threshold1": "阈值1",
        "threshold2": "阈值2",
        "face_params": "人脸检测参数",
        "scale_factor": "缩放因子",
        "scale_factor_help": "较小的值检测更全面，但可能增加误检",
        "min_neighbors": "最小邻域数",
        "min_neighbors_help": "较大的值减少误检，但可能漏检",
        "min_size": "最小人脸尺寸(像素)",
        "min_size_help": "过滤太小的检测结果",
        "require_eyes": "要求检测到眼睛",
        "require_eyes_help": "启用后，只有检测到眼睛的区域才会被标记为人脸",
        "require_skin": "要求肤色检测",
        "require_skin_help": "启用后，只有肤色区域才会被标记为人脸",
        "confidence_threshold": "级联分类器置信度",
        "confidence_threshold_help": "需要多少个分类器确认才标记为人脸",
        "object_params": "目标检测参数",
        "confidence": "置信度阈值",
        "confidence_help": "较低的值检测更多目标，但可能增加误检",
        "supported_classes": "支持检测的目标类别",
        "model_loading": "模型加载中...",
        "download_result": "下载结果图",
        "download_hint": "处理图片后可下载结果",
        "original_image": "原图",
        "processed_result": "处理结果",
        "upload_hint": "请上传一张图片",
        "result_hint": "上传图片后将显示处理结果",
        "loading_model": "正在加载目标检测模型，请稍候...",
        "module_selection": "模块选择",
        "module_detection": "📊 检测模块",
        "module_basic": "🛠️ 基础图像处理",
        "module_enhance": "✨ 图像增强与形态学",
        "temp_save": "暂存结果",
        "temp_save_panel": "临时保存区",
        "temp_save_hint": "点击暂存按钮保存当前处理结果，点击已保存的图片可将其作为新的原图继续处理",
        "no_saved_images": "暂无暂存图片",
        "clear_saved": "清空暂存",
        "use_as_original": "用作原图",
        "color_space": "色彩空间转换",
        "color_space_params": "色彩空间参数",
        "target_space": "目标色彩空间",
        "space_gray": "灰度 (Gray)",
        "space_hsv": "HSV",
        "space_rgb": "RGB",
        "geometric": "几何变换",
        "geometric_params": "几何变换参数",
        "transform_type": "变换类型",
        "transform_resize": "缩放",
        "transform_rotate": "旋转",
        "transform_translate": "平移",
        "transform_affine": "仿射变换",
        "param_scale": "缩放比例",
        "param_angle": "旋转角度",
        "param_tx": "水平偏移",
        "param_ty": "垂直偏移",
        "threshold": "图像阈值化",
        "threshold_params": "阈值化参数",
        "threshold_type": "阈值类型",
        "type_binary": "二值化",
        "type_binary_inv": "反二值化",
        "type_truncate": "截断",
        "type_to_zero": "归零",
        "type_adaptive": "自适应阈值",
        "type_otsu": "Otsu自动阈值",
        "param_thresh": "阈值",
        "param_block_size": "块大小",
        "param_C": "常数C",
        "smoothing": "平滑/滤波",
        "smoothing_params": "滤波参数",
        "filter_type": "滤波类型",
        "filter_gaussian": "高斯模糊",
        "filter_median": "中值滤波",
        "filter_blur": "均值模糊",
        "filter_bilateral": "双边滤波",
        "param_ksize": "核大小",
        "param_d": "邻域直径",
        "param_sigma_color": "颜色sigma",
        "param_sigma_space": "空间sigma",
        "morphology": "形态学变换",
        "morphology_params": "形态学参数",
        "morph_type": "运算类型",
        "morph_erode": "腐蚀",
        "morph_dilate": "膨胀",
        "morph_opening": "开运算",
        "morph_closing": "闭运算",
        "param_kernel_size": "核大小",
        "param_iterations": "迭代次数",
        "histogram": "直方图均衡化",
    },
    "en": {
        "title": "Web Image Annotation Tool",
        "subtitle": "Image annotation tool based on Streamlit and OpenCV, supporting edge detection, face detection and object detection",
        "config_panel": "Configuration Panel",
        "language": "Language",
        "upload": "Upload Image",
        "select_function": "Select Function",
        "original": "Original",
        "canny": "Edge Detection (Canny)",
        "face": "Face Detection (Haar)",
        "object": "Object Detection (DNN)",
        "canny_params": "Canny Parameters",
        "threshold1": "Threshold 1",
        "threshold2": "Threshold 2",
        "face_params": "Face Detection Parameters",
        "scale_factor": "Scale Factor",
        "scale_factor_help": "Smaller values detect more thoroughly but may increase false positives",
        "min_neighbors": "Min Neighbors",
        "min_neighbors_help": "Larger values reduce false positives but may miss detections",
        "min_size": "Min Face Size (pixels)",
        "min_size_help": "Filter out detections that are too small",
        "require_eyes": "Require Eye Detection",
        "require_eyes_help": "When enabled, only regions with detected eyes are marked as faces",
        "require_skin": "Require Skin Detection",
        "require_skin_help": "When enabled, only skin-colored regions are marked as faces",
        "confidence_threshold": "Cascade Classifier Confidence",
        "confidence_threshold_help": "Minimum fraction of classifiers that must confirm a face",
        "object_params": "Object Detection Parameters",
        "confidence": "Confidence Threshold",
        "confidence_help": "Lower values detect more objects but may increase false positives",
        "supported_classes": "Supported Object Classes",
        "model_loading": "Model loading...",
        "download_result": "Download Result",
        "download_hint": "Process an image to enable download",
        "original_image": "Original Image",
        "processed_result": "Processed Result",
        "upload_hint": "Please upload an image",
        "result_hint": "Processed result will appear here after uploading",
        "loading_model": "Loading object detection model, please wait...",
        "module_selection": "Module Selection",
        "module_detection": "📊 Detection Module",
        "module_basic": "🛠️ Basic Image Processing",
        "module_enhance": "✨ Enhancement & Morphology",
        "temp_save": "Save Temp",
        "temp_save_panel": "Temp Save Panel",
        "temp_save_hint": "Click save to store the current result. Click a saved image to use it as the new original for further processing.",
        "no_saved_images": "No saved images",
        "clear_saved": "Clear All",
        "use_as_original": "Use as Original",
        "color_space": "Color Space Conversion",
        "color_space_params": "Color Space Parameters",
        "target_space": "Target Color Space",
        "space_gray": "Grayscale",
        "space_hsv": "HSV",
        "space_rgb": "RGB",
        "geometric": "Geometric Transform",
        "geometric_params": "Geometric Transform Parameters",
        "transform_type": "Transform Type",
        "transform_resize": "Resize",
        "transform_rotate": "Rotate",
        "transform_translate": "Translate",
        "transform_affine": "Affine",
        "param_scale": "Scale Factor",
        "param_angle": "Rotation Angle",
        "param_tx": "Horizontal Offset",
        "param_ty": "Vertical Offset",
        "threshold": "Image Thresholding",
        "threshold_params": "Threshold Parameters",
        "threshold_type": "Threshold Type",
        "type_binary": "Binary",
        "type_binary_inv": "Binary Inverse",
        "type_truncate": "Truncate",
        "type_to_zero": "To Zero",
        "type_adaptive": "Adaptive",
        "type_otsu": "Otsu",
        "param_thresh": "Threshold Value",
        "param_block_size": "Block Size",
        "param_C": "Constant C",
        "smoothing": "Smoothing / Filtering",
        "smoothing_params": "Filter Parameters",
        "filter_type": "Filter Type",
        "filter_gaussian": "Gaussian Blur",
        "filter_median": "Median Filter",
        "filter_blur": "Mean Blur",
        "filter_bilateral": "Bilateral Filter",
        "param_ksize": "Kernel Size",
        "param_d": "Neighborhood Diameter",
        "param_sigma_color": "Color Sigma",
        "param_sigma_space": "Space Sigma",
        "morphology": "Morphological Transform",
        "morphology_params": "Morphology Parameters",
        "morph_type": "Operation Type",
        "morph_erode": "Erosion",
        "morph_dilate": "Dilation",
        "morph_opening": "Opening",
        "morph_closing": "Closing",
        "param_kernel_size": "Kernel Size",
        "param_iterations": "Iterations",
        "histogram": "Histogram Equalization",
    }
}


def get_class_display_name(class_name, lang):
    if lang == "zh":
        cn = CLASS_NAMES_CN.get(class_name)
        if cn:
            return f"{cn} ({class_name})"
    return class_name


def build_detection_params(t, lang, processor):
    params = {}
    function_options = [
        (t["original"], "original"),
        (t["canny"], "canny"),
        (t["face"], "face"),
        (t["object"], "object"),
    ]
    selected = st.radio(
        t["select_function"],
        function_options,
        format_func=lambda x: x[0],
        index=0,
        key="selected_function",
        horizontal=True,
    )

    if selected[1] == "canny":
        st.subheader(t["canny_params"])
        params["threshold1"] = st.slider(t["threshold1"], 0, 255, 100, key="canny_threshold1")
        params["threshold2"] = st.slider(t["threshold2"], 0, 255, 200, key="canny_threshold2")
    elif selected[1] == "face":
        st.subheader(t["face_params"])
        params["scale_factor"] = st.slider(t["scale_factor"], 1.01, 1.2, 1.05, 0.01, key="face_scale_factor", help=t["scale_factor_help"])
        params["min_neighbors"] = st.slider(t["min_neighbors"], 1, 15, 3, key="face_min_neighbors", help=t["min_neighbors_help"])
        params["min_size"] = st.slider(t["min_size"], 30, 150, 30, key="face_min_size", help=t["min_size_help"])
        params["require_eyes"] = st.checkbox(t["require_eyes"], value=False, key="face_require_eyes", help=t["require_eyes_help"])
        params["require_skin"] = st.checkbox(t["require_skin"], value=False, key="face_require_skin", help=t["require_skin_help"])
        params["confidence_threshold"] = st.slider(t["confidence_threshold"], 0.0, 1.0, 0.3, 0.1, key="face_confidence", help=t["confidence_threshold_help"])
    elif selected[1] == "object":
        st.subheader(t["object_params"])
        params["confidence"] = st.slider(t["confidence"], 0.1, 0.9, 0.3, 0.05, key="confidence_threshold", help=t["confidence_help"])
        st.markdown("---")
        st.subheader(t["supported_classes"])
        supported_classes = processor.get_supported_classes()
        cols = st.columns(2)
        for i, class_name in enumerate(supported_classes):
            with cols[i % 2]:
                st.write(f"- {get_class_display_name(class_name, lang)}")

    return selected[1], params


def build_basic_params(t):
    params = {}
    function_options = [
        (t["color_space"], "color_space"),
        (t["geometric"], "geometric"),
        (t["threshold"], "threshold"),
        (t["smoothing"], "smoothing"),
    ]
    selected = st.radio(
        t["select_function"],
        function_options,
        format_func=lambda x: x[0],
        index=0,
        key="selected_basic_function",
        horizontal=True,
    )

    if selected[1] == "color_space":
        st.subheader(t["color_space_params"])
        space_options = [
            (t["space_gray"], "gray"),
            (t["space_hsv"], "hsv"),
            (t["space_rgb"], "rgb"),
        ]
        chosen = st.radio(t["target_space"], space_options, format_func=lambda x: x[0], index=0, key="target_space", horizontal=True)
        params["target_space"] = chosen[1]

    elif selected[1] == "geometric":
        st.subheader(t["geometric_params"])
        transform_options = [
            (t["transform_resize"], "resize"),
            (t["transform_rotate"], "rotate"),
            (t["transform_translate"], "translate"),
            (t["transform_affine"], "affine"),
        ]
        chosen = st.radio(t["transform_type"], transform_options, format_func=lambda x: x[0], index=0, key="transform_type", horizontal=True)
        params["transform_type"] = chosen[1]

        if chosen[1] == "resize":
            params["scale"] = st.slider(t["param_scale"], 0.1, 3.0, 1.0, 0.1, key="geo_scale")
        elif chosen[1] == "rotate":
            params["angle"] = st.slider(t["param_angle"], 0, 360, 45, key="geo_angle")
        elif chosen[1] == "translate":
            params["tx"] = st.slider(t["param_tx"], -200, 200, 50, key="geo_tx")
            params["ty"] = st.slider(t["param_ty"], -200, 200, 50, key="geo_ty")

    elif selected[1] == "threshold":
        st.subheader(t["threshold_params"])
        type_options = [
            (t["type_binary"], "binary"),
            (t["type_binary_inv"], "binary_inv"),
            (t["type_truncate"], "truncate"),
            (t["type_to_zero"], "to_zero"),
            (t["type_adaptive"], "adaptive"),
            (t["type_otsu"], "otsu"),
        ]
        chosen = st.radio(t["threshold_type"], type_options, format_func=lambda x: x[0], index=0, key="threshold_type", horizontal=True)
        params["threshold_type"] = chosen[1]

        if chosen[1] in ("binary", "binary_inv", "truncate", "to_zero"):
            params["thresh"] = st.slider(t["param_thresh"], 0, 255, 127, key="thresh_val")
        elif chosen[1] == "adaptive":
            params["block_size"] = st.slider(t["param_block_size"], 3, 31, 11, 2, key="block_size")
            params["C"] = st.slider(t["param_C"], -20, 20, 2, key="adaptive_C")

    elif selected[1] == "smoothing":
        st.subheader(t["smoothing_params"])
        filter_options = [
            (t["filter_gaussian"], "gaussian"),
            (t["filter_median"], "median"),
            (t["filter_blur"], "blur"),
            (t["filter_bilateral"], "bilateral"),
        ]
        chosen = st.radio(t["filter_type"], filter_options, format_func=lambda x: x[0], index=0, key="filter_type", horizontal=True)
        params["filter_type"] = chosen[1]

        if chosen[1] in ("gaussian", "median", "blur"):
            params["ksize"] = st.slider(t["param_ksize"], 1, 21, 5, 2, key="smooth_ksize")
        elif chosen[1] == "bilateral":
            params["d"] = st.slider(t["param_d"], 1, 25, 9, key="bilateral_d")
            params["sigma_color"] = st.slider(t["param_sigma_color"], 1, 200, 75, key="bilateral_sc")
            params["sigma_space"] = st.slider(t["param_sigma_space"], 1, 200, 75, key="bilateral_ss")

    return selected[1], params


def build_enhance_params(t):
    params = {}
    function_options = [
        (t["morphology"], "morphology"),
        (t["histogram"], "histogram"),
    ]
    selected = st.radio(
        t["select_function"],
        function_options,
        format_func=lambda x: x[0],
        index=0,
        key="selected_enhance_function",
        horizontal=True,
    )

    if selected[1] == "morphology":
        st.subheader(t["morphology_params"])
        morph_options = [
            (t["morph_erode"], "erode"),
            (t["morph_dilate"], "dilate"),
            (t["morph_opening"], "opening"),
            (t["morph_closing"], "closing"),
        ]
        chosen = st.radio(t["morph_type"], morph_options, format_func=lambda x: x[0], index=0, key="morph_type", horizontal=True)
        params["morph_type"] = chosen[1]
        params["kernel_size"] = st.slider(t["param_kernel_size"], 1, 15, 5, 2, key="morph_kernel")
        params["iterations"] = st.slider(t["param_iterations"], 1, 5, 1, key="morph_iter")

    return selected[1], params


def main():
    if 'language' not in st.session_state:
        st.session_state.language = 'zh'

    lang = st.session_state.language
    t = TEXT[lang]

    st.set_page_config(
        page_title=t["title"],
        page_icon="🖼️",
        layout="wide"
    )

    st.title(f" {t['title']}")
    st.markdown(t["subtitle"])

    st.markdown("""
        <style>
        .temp-column-scroll {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
            position: sticky;
            top: 20px;
        }
        .temp-column-scroll::-webkit-scrollbar {
            width: 6px;
        }
        .temp-column-scroll::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        .temp-column-scroll::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        .temp-column-scroll::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        </style>
    """, unsafe_allow_html=True)

    if 'processor' not in st.session_state:
        st.session_state.processor = ImageProcessor()

    processor = st.session_state.processor

    with st.sidebar:
        st.header(t["config_panel"])

        lang_options = [("中文", "zh"), ("English", "en")]
        selected_lang = st.radio(
            t["language"],
            lang_options,
            format_func=lambda x: x[0],
            index=0 if lang == "zh" else 1,
            key="language_selector",
            horizontal=True
        )
        if selected_lang[1] != lang:
            st.session_state.language = selected_lang[1]
            st.rerun()

        st.markdown("---")

        uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and not st.session_state.get('using_temp_image', False):
            file_bytes = uploaded_file.read()
            cv_image = processor.load_image(file_bytes)
            cv_image = processor.resize_image(cv_image)
            st.session_state.cv_image = cv_image
            st.session_state.using_temp_image = False

            if 'saved_images' not in st.session_state:
                st.session_state.saved_images = []

            original_in_saved = any(s.get('name') == t["original"] for s in st.session_state.saved_images)
            if not original_in_saved:
                st.session_state.saved_images.insert(0, {
                    'image': cv_image.copy(),
                    'bytes': processor.image_to_bytes(cv_image),
                    'timestamp': time.time(),
                    'name': t["original"]
                })
        elif uploaded_file is None:
            st.session_state.using_temp_image = False
            for key in ["cv_image", "processed_image", "processed_image_bytes", "saved_images"]:
                if key in st.session_state:
                    del st.session_state[key]

        st.markdown("---")

        if 'active_module' not in st.session_state:
            st.session_state.active_module = 'detection'

        selected_function = "original"
        params = {}

        module_buttons = [
            (t["module_detection"], 'detection'),
            (t["module_basic"], 'basic'),
            (t["module_enhance"], 'enhance')
        ]
        for label, module_key in module_buttons:
            btn_type = "primary" if st.session_state.active_module == module_key else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"module_{module_key}"):
                st.session_state.active_module = module_key
                st.rerun()

        st.markdown("---")

        if st.session_state.active_module == 'detection':
            selected_function, params = build_detection_params(t, lang, processor)
        elif st.session_state.active_module == 'basic':
            selected_function, params = build_basic_params(t)
        elif st.session_state.active_module == 'enhance':
            selected_function, params = build_enhance_params(t)

        st.markdown("---")

    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

    has_saved_images = 'saved_images' in st.session_state and len(st.session_state.saved_images) > 0
    show_temp_panel = st.session_state.get('show_temp_panel', True)

    if has_saved_images and show_temp_panel:
        col1, col2, col3 = st.columns([1, 1, 0.4])
    else:
        col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(t["original_image"])
        if "cv_image" in st.session_state:
            rgb_image = processor.bgr_to_rgb(st.session_state.cv_image)
            st.image(rgb_image)
        else:
            st.info(t["upload_hint"])

    with col2:
        st.subheader(t["processed_result"])
        if "cv_image" in st.session_state:
            if selected_function == "object" and not processor.is_object_model_loaded():
                with st.spinner(t["loading_model"]):
                    processed_image = processor.process_image(
                        st.session_state.cv_image,
                        selected_function,
                        **params
                    )
            else:
                processed_image = processor.process_image(
                    st.session_state.cv_image,
                    selected_function,
                    **params
                )

            st.session_state.processed_image = processed_image
            img_bytes = processor.image_to_bytes(processed_image)
            if img_bytes:
                st.session_state.processed_image_bytes = img_bytes
            rgb_processed = processor.bgr_to_rgb(processed_image)
            st.image(rgb_processed)

            st.markdown("---")
            if st.button(f"💾 {t['temp_save']}", use_container_width=True):
                if 'saved_images' not in st.session_state:
                    st.session_state.saved_images = []
                st.session_state.saved_images.append({
                    'image': processed_image.copy(),
                    'bytes': img_bytes,
                    'timestamp': time.time(),
                    'name': f"{t['processed_result']} {len(st.session_state.saved_images)}"
                })
                st.success(f"已暂存！当前共 {len(st.session_state.saved_images)} 张图片")
        else:
            st.info(t["result_hint"])

    if has_saved_images and show_temp_panel:
        with col3:
            st.markdown('<div id="temp-save-scroll" class="temp-column-scroll">', unsafe_allow_html=True)
            
            if st.button(f"▼ {t['temp_save_panel']}", use_container_width=True, key="toggle_temp_panel"):
                st.session_state.show_temp_panel = False
                st.rerun()
            
            st.caption(t["temp_save_hint"])

            if st.button(f"🗑️ {t['clear_saved']}", use_container_width=True, key="clear_saved_btn"):
                st.session_state.saved_images = []
                st.session_state.using_temp_image = False
                st.rerun()

            st.markdown("---")

            for idx, saved in enumerate(st.session_state.saved_images):
                st.markdown(f"**{saved['name']}**")
                rgb_saved = processor.bgr_to_rgb(saved['image'])
                st.image(rgb_saved, width=150)

                if st.button(f"️ {t['use_as_original']}", key=f"use_saved_{idx}", use_container_width=True):
                    st.session_state.cv_image = saved['image'].copy()
                    st.session_state.using_temp_image = True
                    st.rerun()

                if saved.get('bytes') and saved['name'] != t["original"]:
                    st.download_button(
                        label=f"📥 {t['download_result']}",
                        data=saved['bytes'],
                        file_name=f"{saved['name']}.png",
                        mime="image/png",
                        key=f"download_saved_{idx}",
                        use_container_width=True
                    )

                st.markdown("---")

            st.markdown('</div>', unsafe_allow_html=True)

    if has_saved_images and not show_temp_panel:
        if st.button(f"📋 {t['temp_save_panel']} ({len(st.session_state.saved_images)})", key="show_temp_panel_btn"):
            st.session_state.show_temp_panel = True
            st.rerun()


if __name__ == "__main__":
    main()
