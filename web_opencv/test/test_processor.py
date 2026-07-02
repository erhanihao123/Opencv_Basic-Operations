import pytest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.image_processor import ImageProcessor


class TestImageProcessor:

    @classmethod
    def setup_class(cls):
        cls.processor = ImageProcessor()
        cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_load_image(self):
        success, buffer = cv2.imencode('.png', self.test_image)
        file_bytes = buffer.tobytes()

        result = self.processor.load_image(file_bytes)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 3

    def test_load_image_empty_bytes(self):
        result = self.processor.load_image(b'')
        assert result is None or result.size == 0

    def test_canny_edge_detection(self):
        result = self.processor.canny_edge_detection(self.test_image, 100, 200)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_canny_threshold_bounds(self):
        result = self.processor.canny_edge_detection(self.test_image, 0, 255)
        assert result is not None

        result = self.processor.canny_edge_detection(self.test_image, 255, 0)
        assert result is not None

    def test_face_detection(self):
        result = self.processor.face_detection(self.test_image)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_object_detection(self):
        result = self.processor.object_detection(self.test_image, 0.5)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_object_confidence_bounds(self):
        result = self.processor.object_detection(self.test_image, 0.1)
        assert result is not None

        result = self.processor.object_detection(self.test_image, 0.9)
        assert result is not None

    def test_color_space_conversion_gray(self):
        result = self.processor.color_space_conversion(self.test_image, "gray")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_color_space_conversion_hsv(self):
        result = self.processor.color_space_conversion(self.test_image, "hsv")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_color_space_conversion_rgb(self):
        result = self.processor.color_space_conversion(self.test_image, "rgb")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_geometric_transform_resize(self):
        result = self.processor.geometric_transform(self.test_image, "resize", scale=0.5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == self.test_image.shape[0] // 2
        assert result.shape[1] == self.test_image.shape[1] // 2

    def test_geometric_transform_rotate(self):
        result = self.processor.geometric_transform(self.test_image, "rotate", angle=45)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 3

    def test_geometric_transform_translate(self):
        result = self.processor.geometric_transform(self.test_image, "translate", tx=50, ty=30)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_geometric_transform_affine(self):
        result = self.processor.geometric_transform(self.test_image, "affine")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_binary(self):
        result = self.processor.image_threshold(self.test_image, "binary", thresh=127)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_binary_inv(self):
        result = self.processor.image_threshold(self.test_image, "binary_inv", thresh=127)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_truncate(self):
        result = self.processor.image_threshold(self.test_image, "truncate", thresh=127)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_to_zero(self):
        result = self.processor.image_threshold(self.test_image, "to_zero", thresh=127)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_adaptive(self):
        result = self.processor.image_threshold(self.test_image, "adaptive", block_size=11, C=2)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_threshold_otsu(self):
        result = self.processor.image_threshold(self.test_image, "otsu")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_smoothing_gaussian(self):
        result = self.processor.image_smoothing(self.test_image, "gaussian", ksize=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_smoothing_median(self):
        result = self.processor.image_smoothing(self.test_image, "median", ksize=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_smoothing_blur(self):
        result = self.processor.image_smoothing(self.test_image, "blur", ksize=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_image_smoothing_bilateral(self):
        result = self.processor.image_smoothing(self.test_image, "bilateral", d=9, sigma_color=75, sigma_space=75)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_morphological_transform_erode(self):
        result = self.processor.morphological_transform(self.test_image, "erode", kernel_size=5, iterations=1)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_morphological_transform_dilate(self):
        result = self.processor.morphological_transform(self.test_image, "dilate", kernel_size=5, iterations=1)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_morphological_transform_opening(self):
        result = self.processor.morphological_transform(self.test_image, "opening", kernel_size=5, iterations=1)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_morphological_transform_closing(self):
        result = self.processor.morphological_transform(self.test_image, "closing", kernel_size=5, iterations=1)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_histogram_equalization(self):
        result = self.processor.histogram_equalization(self.test_image)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_resize_image_default(self):
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)

        result = self.processor.resize_image(large_image)

        assert result is not None
        assert max(result.shape[:2]) <= 1024

    def test_resize_image_custom_dimension(self):
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)

        result = self.processor.resize_image(large_image, max_dimension=512)

        assert result is not None
        assert max(result.shape[:2]) <= 512

    def test_resize_image_no_change(self):
        small_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)

        result = self.processor.resize_image(small_image)

        assert result is not None
        assert result.shape == small_image.shape

    def test_image_to_bytes(self):
        result = self.processor.image_to_bytes(self.test_image)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_image_to_bytes_invalid_input(self):
        invalid_image = np.array([])
        result = self.processor.image_to_bytes(invalid_image)
        assert result is None

    def test_bgr_to_rgb(self):
        result = self.processor.bgr_to_rgb(self.test_image)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_original(self):
        result = self.processor.process_image(self.test_image, "original")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_canny(self):
        result = self.processor.process_image(
            self.test_image, "canny", threshold1=100, threshold2=200
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_face(self):
        result = self.processor.process_image(self.test_image, "face")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_object(self):
        result = self.processor.process_image(
            self.test_image, "object", confidence=0.5
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_color_space(self):
        result = self.processor.process_image(self.test_image, "color_space", target_space="gray")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_geometric(self):
        result = self.processor.process_image(self.test_image, "geometric", transform_type="resize", scale=0.5)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_process_image_threshold(self):
        result = self.processor.process_image(self.test_image, "threshold", threshold_type="binary", thresh=127)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_smoothing(self):
        result = self.processor.process_image(self.test_image, "smoothing", filter_type="gaussian", ksize=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_morphology(self):
        result = self.processor.process_image(self.test_image, "morphology", morph_type="erode", kernel_size=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape

    def test_process_image_histogram(self):
        result = self.processor.process_image(self.test_image, "histogram")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape