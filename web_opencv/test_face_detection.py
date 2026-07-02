import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from backend.image_processor import ImageProcessor


def create_synthetic_group_photo():
    """Create a synthetic group photo with multiple face-like regions"""
    img = np.ones((600, 900, 3), dtype=np.uint8) * 240  # light background

    # Skin-like color
    skin_color = (160, 180, 220)  # BGR

    # Draw 12 face-like ovals at various positions
    face_positions = [
        (80, 80), (230, 80), (380, 80), (530, 80), (680, 80), (800, 80),
        (80, 280), (230, 280), (380, 280), (530, 280), (680, 280), (800, 280),
    ]
    for (cx, cy) in face_positions:
        cv2.ellipse(img, (cx, cy), (35, 45), 0, 0, 360, skin_color, -1)
        # Add two eye-like dark spots
        cv2.circle(img, (cx - 12, cy - 10), 6, (40, 40, 40), -1)
        cv2.circle(img, (cx + 12, cy - 10), 6, (40, 40, 40), -1)
        # Add a mouth-like line
        cv2.line(img, (cx - 10, cy + 15), (cx + 10, cy + 15), (80, 60, 60), 2)

    return img


def test_old_vs_new():
    """Compare old and new face detection behavior"""
    processor = ImageProcessor()
    image = create_synthetic_group_photo()

    img_height, img_width = image.shape[:2]
    total_area = img_height * img_width

    print(f"Image: {img_width}x{img_height}, total_area={total_area}")
    print(f"12 synthetic faces, each ~70x90 = 6300px")
    print(f"Old min_face_area (1%): {total_area * 0.01:.0f}  -> faces TOO SMALL, all filtered")
    print(f"New min_face_area (0.0005%): {total_area * 0.0005:.0f}  -> faces pass filter")
    print()

    # Simulate old behavior
    print("=== OLD behavior (min_face_area = 1%) ===")
    old_min = total_area * 0.01
    old_max = total_area * 0.25
    print(f"  Area bounds: [{old_min:.0f}, {old_max:.0f}]")
    print(f"  A 70x90 face (6300px) would be: {'PASS' if old_min <= 6300 <= old_max else 'REJECTED'}")

    # Simulate new behavior
    print()
    print("=== NEW behavior (min_face_area = 0.0005%) ===")
    new_min = total_area * 0.0005
    new_max = total_area * 0.5
    print(f"  Area bounds: [{new_min:.0f}, {new_max:.0f}]")
    print(f"  A 70x90 face (6300px) would be: {'PASS' if new_min <= 6300 <= new_max else 'REJECTED'}")

    # Now test actual detection
    print()
    print("=== Actual face detection test ===")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    for i, cascade in enumerate(processor.face_cascades):
        faces = cascade.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(f"  Cascade {i}: {len(faces)} raw detections")

    # Test with relaxed params (simulating new defaults)
    result = processor.face_detection(
        image,
        min_neighbors=3,
        min_size=(30, 30),
        scale_factor=1.05,
        require_eyes=False,
        require_skin=False,
        confidence_threshold=0.0,
    )

    # Count green rectangles (detected faces)
    green_pixels = np.sum((result[:, :, 0] < 100) & (result[:, :, 1] > 200) & (result[:, :, 2] < 100))
    estimated_faces = green_pixels // (2 * 2 * (70 + 90))  # rough estimate
    print(f"  Green rectangle pixels: {green_pixels}")
    print(f"  Estimated detected faces: ~{max(0, estimated_faces)}")

    # Save result for visual inspection
    out_path = os.path.join(os.path.dirname(__file__), "test_face_result.png")
    cv2.imwrite(out_path, result)
    print(f"  Result saved to: {out_path}")


if __name__ == "__main__":
    test_old_vs_new()
