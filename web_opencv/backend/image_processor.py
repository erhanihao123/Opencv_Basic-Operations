import cv2
import numpy as np
import os
import urllib.request
import socket
import logging
import time


log_dir = os.getenv('LOG_DIR', os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'image_processor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from .oss_storage import OSSStorage
    HAS_OSS = True
except ImportError:
    HAS_OSS = False
    logger.warning("OSS storage module not available, OSS functionality will be disabled")


MODEL_URLS = {
    'prototxt': [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        "https://cdn.jsdelivr.net/gh/chuanqi305/MobileNet-SSD@master/deploy.prototxt"
    ],
    'caffemodel': [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
        "https://cdn.jsdelivr.net/gh/chuanqi305/MobileNet-SSD@master/mobilenet_iter_73000.caffemodel"
    ],
    'haarcascade_frontalface_default': [
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_default.xml"
    ],
    'haarcascade_frontalface_alt2': [
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    ],
    'haarcascade_frontalface_alt': [
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt.xml"
    ],
    'haarcascade_frontalface_alt_tree': [
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml",
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
    ],
    'haarcascade_eye': [
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_eye.xml"
    ]
}


class ImageProcessor:
    def __init__(self):
        self.face_cascades = []
        self.eye_cascade = None
        self.object_net = None
        self.class_names = None
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.oss_storage = OSSStorage() if HAS_OSS else None
        self._ensure_model_dir()
        self._load_face_cascades()
        self._load_eye_cascade()
        self._preload_object_model()

    def _ensure_model_dir(self):
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            logger.info(f"Model directory ensured: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to create model directory: {e}")

    def _get_cascade_path(self, cascade_name):
        cascade_path = os.path.join(self.model_dir, cascade_name)
        if os.path.exists(cascade_path) and os.path.getsize(cascade_path) > 0:
            return cascade_path
        model_key = cascade_name.replace('.xml', '')
        downloaded_path = self._download_model(cascade_name, model_key)
        if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 0:
            return downloaded_path
        return None

    def _load_face_cascades(self):
        cascade_files = [
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt_tree.xml'
        ]
        
        for cascade_file in cascade_files:
            cascade_path = None
            try:
                cascade_path = self._get_cascade_path(cascade_file)
                if cascade_path:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.face_cascades.append(cascade)
                        logger.info(f"Loaded face cascade: {cascade_file}")
                    else:
                        logger.warning(f"Empty cascade file: {cascade_file}")
                else:
                    logger.warning(f"Cascade file not found: {cascade_file}")
            except Exception as e:
                logger.error(f"Failed to load {cascade_file}: {e}")

        if len(self.face_cascades) == 0:
            logger.warning("No face cascades loaded")

    def _load_eye_cascade(self):
        cascade_file = 'haarcascade_eye.xml'
        try:
            cascade_path = self._get_cascade_path(cascade_file)
            if cascade_path:
                self.eye_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.eye_cascade.empty():
                    logger.info("Loaded eye cascade")
                else:
                    logger.warning("Empty eye cascade file")
            else:
                logger.warning("Eye cascade file not found")
        except Exception as e:
            logger.error(f"Failed to load eye cascade: {e}")

    def _download_with_retry(self, url, filepath, max_retries=3, timeout=30):
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {os.path.basename(filepath)} from {url} (attempt {attempt + 1}/{max_retries})")
                socket.setdefaulttimeout(timeout)
                urllib.request.urlretrieve(url, filepath)
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    logger.info(f"Successfully downloaded {os.path.basename(filepath)}")
                    return True
                else:
                    logger.warning(f"Downloaded file is empty, retrying...")
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return False

    def _download_model(self, filename, model_type):
        filepath = os.path.join(self.model_dir, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Model file already exists: {filename}")
            return filepath
        
        urls = MODEL_URLS.get(model_type, [])
        for url in urls:
            if self._download_with_retry(url, filepath):
                return filepath
        
        logger.error(f"Failed to download {filename} from all sources")
        return filepath

    def _preload_object_model(self):
        try:
            logger.info("Preloading object detection model...")
            start_time = time.time()
            self._load_object_detection_model()
            elapsed = time.time() - start_time
            logger.info(f"Object detection model preloaded in {elapsed:.2f} seconds")
        except Exception as e:
            logger.warning(f"Preload failed, will load on first use: {e}")

    def _load_object_detection_model(self):
        if self.object_net is not None:
            return True

        local_prototxt = self._download_model("MobileNetSSD_deploy.prototxt", "prototxt")
        local_caffemodel = self._download_model("MobileNetSSD_deploy.caffemodel", "caffemodel")

        if os.path.exists(local_prototxt) and os.path.exists(local_caffemodel):
            try:
                self.object_net = cv2.dnn.readNetFromCaffe(local_prototxt, local_caffemodel)
                self.class_names = [
                    "background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant",
                    "sheep", "sofa", "train", "tvmonitor"
                ]
                logger.info("Object detection model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load object detection model: {e}")
        else:
            logger.error("Object detection model files not found")

        return False

    def is_object_model_loaded(self):
        return self.object_net is not None

    def load_image(self, file_bytes):
        if not file_bytes:
            logger.warning("Empty file bytes received")
            return None
        np_arr = np.frombuffer(file_bytes, np.uint8)
        if np_arr.size == 0:
            logger.warning("Empty numpy array from file bytes")
            return None
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to decode image")
        return image

    def canny_edge_detection(self, image, threshold1, threshold2):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return result
        except Exception as e:
            logger.error(f"Canny edge detection failed: {e}")
            return image.copy()

    def _has_eyes(self, face_roi_gray, min_eye_size=10):
        if self.eye_cascade is None:
            return True
        
        try:
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(min_eye_size, min_eye_size)
            )
            return len(eyes) >= 1
        except Exception as e:
            logger.error(f"Eye detection failed: {e}")
            return True

    def _is_skin_color(self, face_roi):
        try:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = face_roi.shape[0] * face_roi.shape[1]
            
            skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
            return skin_ratio > 0.1
        except Exception as e:
            logger.error(f"Skin color detection failed: {e}")
            return True

    def face_detection(self, image, min_neighbors=3, min_size=(30, 30), scale_factor=1.05,
                       require_eyes=False, require_skin=False, confidence_threshold=0.3):
        result = image.copy()
        if len(self.face_cascades) == 0:
            logger.warning("No face cascades available for detection")
            return result

        if image is None or image.size == 0:
            logger.warning("Empty image for face detection")
            return result

        if isinstance(min_size, int):
            min_size = (min_size, min_size)

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except Exception as e:
            logger.error(f"Error converting image to grayscale: {e}")
            return result

        img_height, img_width = image.shape[:2]
        total_area = img_height * img_width
        min_face_area = max(min_size[0] * min_size[1], total_area * 0.0005)
        max_face_area = total_area * 0.5

        all_faces = []

        for cascade in self.face_cascades:
            try:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if faces is not None and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        all_faces.append((x, y, w, h))
            except Exception as e:
                logger.error(f"Error in detectMultiScale with primary params: {e}")
                continue

        if len(all_faces) == 0:
            logger.info("No faces found with initial params, trying aggressive detection...")
            for cascade in self.face_cascades:
                try:
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.03,
                        minNeighbors=2,
                        minSize=(20, 20),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if faces is not None and len(faces) > 0:
                        for (x, y, w, h) in faces:
                            all_faces.append((x, y, w, h))
                except Exception as e:
                    logger.error(f"Error in detectMultiScale with fallback params: {e}")
                    continue

        def iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            return inter_area / union_area if union_area > 0 else 0

        deduped = []
        for face in all_faces:
            x, y, w, h = face
            face_area = w * h
            aspect_ratio = w / h if h > 0 else 1
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            if face_area < min_face_area or face_area > max_face_area:
                continue
            is_duplicate = False
            for existing in deduped:
                if iou(face, existing) > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(face)

        verified = []
        for face in deduped:
            x, y, w, h = face
            face_roi = image[y:y+h, x:x+w]
            face_roi_gray = gray[y:y+h, x:x+w]

            passed_filters = True
            if require_eyes and w >= 40:
                if not self._has_eyes(face_roi_gray, min_eye_size=max(3, w // 8)):
                    passed_filters = False
            if passed_filters and require_skin:
                if not self._is_skin_color(face_roi):
                    passed_filters = False
            if not passed_filters:
                continue

            cascade_count = 0
            for cascade in self.face_cascades:
                try:
                    sub_faces = cascade.detectMultiScale(
                        face_roi_gray,
                        scaleFactor=1.05,
                        minNeighbors=2,
                        minSize=(max(10, min_size[0] // 3), max(10, min_size[1] // 3))
                    )
                    if sub_faces is not None and len(sub_faces) > 0:
                        cascade_count += 1
                except Exception as e:
                    logger.error(f"Error in sub-detection: {e}")
                    continue

            if cascade_count >= 1:
                verified.append(face)

        for (x, y, w, h) in verified:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        logger.info(f"Face detection completed: {len(verified)} faces detected")
        return result

    def object_detection(self, image, confidence_threshold=0.3):
        result = image.copy()
        
        if not self.is_object_model_loaded():
            logger.info("Loading object detection model on demand")
            self._load_object_detection_model()

        if self.object_net is None or self.class_names is None:
            logger.error("Object detection model not available")
            return result

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        self.object_net.setInput(blob)
        detections = self.object_net.forward()

        min_box_area = (w * h) * 0.001
        detected_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                
                if idx == 0:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                box_width = endX - startX
                box_height = endY - startY
                box_area = box_width * box_height

                if box_area < min_box_area:
                    continue

                if box_width < 10 or box_height < 10:
                    continue

                label = f"{self.class_names[idx]}: {confidence:.2f}"
                cv2.rectangle(result, (startX, startY), (endX, endY), (0, 0, 255), 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    result,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                detected_count += 1

        logger.info(f"Object detection completed: {detected_count} objects detected")
        return result

    def resize_image(self, image, max_dimension=1024):
        try:
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return image
        except Exception as e:
            logger.error(f"Image resize failed: {e}")
            return image.copy()

    def image_to_bytes(self, image):
        if image is None or image.size == 0:
            logger.warning("Empty image for conversion to bytes")
            return None
        success, buffer = cv2.imencode('.png', image)
        if success:
            return buffer.tobytes()
        logger.error("Failed to encode image to bytes")
        return None

    def bgr_to_rgb(self, image):
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"BGR to RGB conversion failed: {e}")
            return image.copy()

    def get_supported_classes(self):
        if self.class_names is None:
            self._load_object_detection_model()
        return self.class_names[1:] if self.class_names else []

    def color_space_conversion(self, image, target_space):
        try:
            if target_space == "gray":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif target_space == "hsv":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif target_space == "rgb":
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image.copy()
        except Exception as e:
            logger.error(f"Color space conversion failed: {e}")
            return image.copy()

    def geometric_transform(self, image, transform_type, **kwargs):
        try:
            result = image.copy()
            h, w = image.shape[:2]

            if transform_type == "resize":
                scale = kwargs.get("scale", 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif transform_type == "rotate":
                angle = kwargs.get("angle", 0)
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos = abs(M[0, 0])
                sin = abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                result = cv2.warpAffine(image, M, (new_w, new_h))
            elif transform_type == "translate":
                tx = kwargs.get("tx", 0)
                ty = kwargs.get("ty", 0)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                result = cv2.warpAffine(image, M, (w, h))
            elif transform_type == "affine":
                pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
                pts2 = np.float32([[w * 0.1, h * 0.1], [w * 0.9, 0], [0, h * 0.9]])
                M = cv2.getAffineTransform(pts1, pts2)
                result = cv2.warpAffine(image, M, (w, h))

            return result
        except Exception as e:
            logger.error(f"Geometric transform failed: {e}")
            return image.copy()

    def image_threshold(self, image, threshold_type, **kwargs):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if threshold_type == "binary":
                thresh_val = kwargs.get("thresh", 127)
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            elif threshold_type == "binary_inv":
                thresh_val = kwargs.get("thresh", 127)
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            elif threshold_type == "truncate":
                thresh_val = kwargs.get("thresh", 127)
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_TRUNC)
            elif threshold_type == "to_zero":
                thresh_val = kwargs.get("thresh", 127)
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_TOZERO)
            elif threshold_type == "adaptive":
                block_size = kwargs.get("block_size", 11)
                C = kwargs.get("C", 2)
                if block_size % 2 == 0:
                    block_size += 1
                result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, block_size, C)
            elif threshold_type == "otsu":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                result = gray

            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Image threshold failed: {e}")
            return image.copy()

    def image_smoothing(self, image, filter_type, **kwargs):
        try:
            if filter_type == "gaussian":
                ksize = kwargs.get("ksize", 5)
                if ksize % 2 == 0:
                    ksize += 1
                return cv2.GaussianBlur(image, (ksize, ksize), 0)
            elif filter_type == "median":
                ksize = kwargs.get("ksize", 5)
                if ksize % 2 == 0:
                    ksize += 1
                return cv2.medianBlur(image, ksize)
            elif filter_type == "blur":
                ksize = kwargs.get("ksize", 5)
                return cv2.blur(image, (ksize, ksize))
            elif filter_type == "bilateral":
                d = kwargs.get("d", 9)
                sigma_color = kwargs.get("sigma_color", 75)
                sigma_space = kwargs.get("sigma_space", 75)
                return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return image.copy()
        except Exception as e:
            logger.error(f"Image smoothing failed: {e}")
            return image.copy()

    def morphological_transform(self, image, morph_type, **kwargs):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            kernel_size = kwargs.get("kernel_size", 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            iterations = kwargs.get("iterations", 1)

            if morph_type == "erode":
                result = cv2.erode(binary, kernel, iterations=iterations)
            elif morph_type == "dilate":
                result = cv2.dilate(binary, kernel, iterations=iterations)
            elif morph_type == "opening":
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif morph_type == "closing":
                result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            else:
                result = binary

            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Morphological transform failed: {e}")
            return image.copy()

    def histogram_equalization(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Histogram equalization failed: {e}")
            return image.copy()

    def process_image(self, image, function_type, **kwargs):
        try:
            processed = image.copy()

            if function_type == "canny":
                threshold1 = kwargs.get("threshold1", 100)
                threshold2 = kwargs.get("threshold2", 200)
                processed = self.canny_edge_detection(image, threshold1, threshold2)
            elif function_type == "face":
                min_neighbors = kwargs.get("min_neighbors", 3)
                min_size_val = kwargs.get("min_size", 30)
                min_size = (min_size_val, min_size_val) if isinstance(min_size_val, int) else min_size_val
                scale_factor = kwargs.get("scale_factor", 1.05)
                require_eyes = kwargs.get("require_eyes", False)
                require_skin = kwargs.get("require_skin", False)
                confidence_threshold = kwargs.get("confidence_threshold", 0.3)
                processed = self.face_detection(image, min_neighbors, min_size, scale_factor,
                                               require_eyes, require_skin, confidence_threshold)
            elif function_type == "object":
                confidence = kwargs.get("confidence", 0.3)
                processed = self.object_detection(image, confidence)
            elif function_type == "original":
                pass
            elif function_type == "color_space":
                target_space = kwargs.get("target_space", "gray")
                processed = self.color_space_conversion(image, target_space)
            elif function_type == "geometric":
                transform_type = kwargs.get("transform_type", "resize")
                scale = kwargs.get("scale", 1.0)
                angle = kwargs.get("angle", 0)
                tx = kwargs.get("tx", 0)
                ty = kwargs.get("ty", 0)
                processed = self.geometric_transform(image, transform_type, scale=scale, angle=angle, tx=tx, ty=ty)
            elif function_type == "threshold":
                threshold_type = kwargs.get("threshold_type", "binary")
                thresh = kwargs.get("thresh", 127)
                block_size = kwargs.get("block_size", 11)
                C = kwargs.get("C", 2)
                processed = self.image_threshold(image, threshold_type, thresh=thresh, block_size=block_size, C=C)
            elif function_type == "smoothing":
                filter_type = kwargs.get("filter_type", "gaussian")
                ksize = kwargs.get("ksize", 5)
                d = kwargs.get("d", 9)
                sigma_color = kwargs.get("sigma_color", 75)
                sigma_space = kwargs.get("sigma_space", 75)
                processed = self.image_smoothing(image, filter_type, ksize=ksize, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
            elif function_type == "morphology":
                morph_type = kwargs.get("morph_type", "erode")
                kernel_size = kwargs.get("kernel_size", 5)
                iterations = kwargs.get("iterations", 1)
                processed = self.morphological_transform(image, morph_type, kernel_size=kernel_size, iterations=iterations)
            elif function_type == "histogram":
                processed = self.histogram_equalization(image)

            return processed
        except Exception as e:
            logger.error(f"Image processing failed for {function_type}: {e}")
            return image.copy()

    def is_oss_enabled(self):
        return self.oss_storage is not None and self.oss_storage.is_enabled()

    def save_to_oss(self, image, filename=None, prefix='images/'):
        if not self.is_oss_enabled():
            logger.warning("OSS storage is not enabled")
            return None
        
        try:
            img_bytes = self.image_to_bytes(image)
            if img_bytes:
                return self.oss_storage.upload_image(img_bytes, filename, prefix)
            return None
        except Exception as e:
            logger.error(f"Failed to save image to OSS: {e}")
            return None

    def load_from_oss(self, oss_key):
        if not self.is_oss_enabled():
            logger.warning("OSS storage is not enabled")
            return None
        
        try:
            image_bytes = self.oss_storage.download_image(oss_key)
            if image_bytes:
                return self.load_image(image_bytes)
            return None
        except Exception as e:
            logger.error(f"Failed to load image from OSS: {e}")
            return None

    def list_oss_images(self, prefix='images/'):
        if not self.is_oss_enabled():
            logger.warning("OSS storage is not enabled")
            return []
        
        try:
            return self.oss_storage.list_images(prefix)
        except Exception as e:
            logger.error(f"Failed to list images from OSS: {e}")
            return []