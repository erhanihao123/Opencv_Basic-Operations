"""End-to-end test simulating the exact UI flow"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from backend.image_processor import ImageProcessor


def create_test_group_photo():
    """Create a realistic group photo with 14 people"""
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 240

    # Skin color
    skin = (160, 180, 220)

    # 14 faces in 2 rows
    positions = [
        (100, 150), (250, 150), (400, 150), (550, 150), (700, 150), (850, 150), (1000, 150),
        (100, 450), (250, 450), (400, 450), (550, 450), (700, 450), (850, 450), (1000, 450),
    ]

    for (cx, cy) in positions:
        # Face oval
        cv2.ellipse(img, (cx, cy), (40, 50), 0, 0, 360, skin, -1)
        # Eyes
        cv2.circle(img, (cx - 15, cy - 15), 7, (40, 40, 40), -1)
        cv2.circle(img, (cx + 15, cy - 15), 7, (40, 40, 40), -1)
        # Mouth
        cv2.ellipse(img, (cx, cy + 15), (12, 6), 0, 0, 180, (60, 60, 60), -1)

    return img


def test_ui_flow():
    """Simulate the exact flow from UI to detection"""
    processor = ImageProcessor()
    image = create_test_group_photo()

    print(f"Image shape: {image.shape}")
    print(f"Total area: {image.shape[0] * image.shape[1]}")
    print()

    # Simulate UI parameters (new defaults)
    ui_params = {
        "scale_factor": 1.05,
        "min_neighbors": 3,
        "min_size": 30,  # integer from slider
        "require_eyes": False,
        "require_skin": False,
        "confidence_threshold": 0.3,
    }

    print("UI parameters:")
    for k, v in ui_params.items():
        print(f"  {k}: {v}")
    print()

    # Simulate process_image conversion
    min_size_val = ui_params["min_size"]
    min_size = (min_size_val, min_size_val) if isinstance(min_size_val, int) else min_size_val
    print(f"After conversion: min_size = {min_size}")
    print()

    # Call face_detection with these parameters
    result = processor.face_detection(
        image,
        min_neighbors=ui_params["min_neighbors"],
        min_size=min_size,
        scale_factor=ui_params["scale_factor"],
        require_eyes=ui_params["require_eyes"],
        require_skin=ui_params["require_skin"],
        confidence_threshold=ui_params["confidence_threshold"],
    )

    # Count detected faces (green rectangles)
    green_mask = (result[:, :, 0] < 100) & (result[:, :, 1] > 200) & (result[:, :, 2] < 100)
    green_pixels = np.sum(green_mask)

    # Each rectangle is 2px thick, perimeter ~ 2*(80+100) = 360px
    estimated_faces = green_pixels // 360 if green_pixels > 0 else 0

    print(f"Green pixels: {green_pixels}")
    print(f"Estimated detected faces: {estimated_faces}")
    print(f"Expected: 14")
    print()

    if estimated_faces >= 10:
        print("✓ PASS: Face detection working correctly")
    else:
        print(" FAIL: Face detection not working as expected")

    # Save result
    out_path = os.path.join(os.path.dirname(__file__), "test_e2e_result.png")
    cv2.imwrite(out_path, result)
    print(f"Result saved to: {out_path}")

    return estimated_faces >= 10


if __name__ == "__main__":
    success = test_ui_flow()
    sys.exit(0 if success else 1)
