"""Detailed diagnostic test for face detection"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from backend.image_processor import ImageProcessor


def find_test_images():
    """Find any JPG/PNG images in the project"""
    images = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                # Skip model files and test results
                if 'model' not in path.lower() and 'result' not in path.lower():
                    images.append(path)
    return images


def trace_detection(processor, image, label=""):
    """Trace through the entire detection pipeline"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Image shape: {image.shape}")
    print(f"Total area: {image.shape[0] * image.shape[1]}")

    img_height, img_width = image.shape[:2]
    total_area = img_height * img_width

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    print(f"\n1. Grayscale conversion: OK")

    # Step 2: Raw detection with each cascade
    print(f"\n2. Raw cascade detection:")
    all_faces = []
    for i, cascade in enumerate(processor.face_cascades):
        faces = cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(f"   Cascade {i}: {len(faces)} faces detected")
        if len(faces) > 0:
            for j, (x, y, w, h) in enumerate(faces[:3]):
                print(f"     Face {j}: pos=({x},{y}), size=({w}x{h}), area={w*h}")
        all_faces.extend([(x, y, w, h) for (x, y, w, h) in faces])

    print(f"   Total raw faces: {len(all_faces)}")

    if len(all_faces) == 0:
        print(f"\n   No faces detected by cascades. Stopping.")
        return 0

    # Step 3: Area and aspect ratio filtering
    print(f"\n3. Area and aspect ratio filtering:")
    min_face_area = max(30*30, total_area * 0.0005)
    max_face_area = total_area * 0.5
    print(f"   Area bounds: [{min_face_area:.0f}, {max_face_area:.0f}]")

    deduped = []
    filtered_by_area = 0
    filtered_by_aspect = 0

    for face in all_faces:
        x, y, w, h = face
        face_area = w * h
        aspect_ratio = w / h if h > 0 else 1

        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            filtered_by_aspect += 1
            continue

        if face_area < min_face_area or face_area > max_face_area:
            filtered_by_area += 1
            continue

        deduped.append(face)

    print(f"   Filtered by aspect ratio: {filtered_by_aspect}")
    print(f"   Filtered by area: {filtered_by_area}")
    print(f"   Remaining after filtering: {len(deduped)}")

    # Step 4: IoU deduplication
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

    final_deduped = []
    duplicates = 0
    for face in deduped:
        is_duplicate = False
        for existing in final_deduped:
            if iou(face, existing) > 0.3:
                is_duplicate = True
                duplicates += 1
                break
        if not is_duplicate:
            final_deduped.append(face)

    print(f"   Duplicates removed: {duplicates}")
    print(f"   After deduplication: {len(final_deduped)}")

    # Step 5: Eye and skin filters (disabled in new defaults)
    print(f"\n5. Eye and skin filters (disabled in new defaults): SKIPPED")

    # Step 6: Cascade verification
    print(f"\n6. Cascade verification:")
    verified = []
    for face in final_deduped:
        x, y, w, h = face
        face_roi_gray = gray_eq[y:y+h, x:x+w]

        cascade_count = 0
        for cascade in processor.face_cascades:
            sub_faces = cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(10, 10)
            )
            if len(sub_faces) > 0:
                cascade_count += 1

        ratio = cascade_count / len(processor.face_cascades) if len(processor.face_cascades) > 0 else 0
        passes = ratio >= 0.3

        if passes:
            verified.append(face)

    print(f"   Verified faces: {len(verified)}")

    # Final result
    result = image.copy()
    for (x, y, w, h) in verified:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out_path = os.path.join(os.path.dirname(__file__), f"test_{label.replace(' ', '_')}.png")
    cv2.imwrite(out_path, result)
    print(f"\n   Result saved to: {out_path}")
    print(f"   Final detected faces: {len(verified)}")

    return len(verified)


def main():
    processor = ImageProcessor()

    print(f"Face cascades loaded: {len(processor.face_cascades)}")
    print(f"Eye cascade loaded: {processor.eye_cascade is not None}")

    # Find test images
    test_images = find_test_images()
    print(f"\nFound {len(test_images)} test images")

    if len(test_images) == 0:
        print("\nNo test images found. Please add a JPG/PNG image to the project.")
        return

    # Test each image
    results = []
    for img_path in test_images[:3]:  # Test up to 3 images
        image = cv2.imread(img_path)
        if image is None:
            print(f"\nFailed to load: {img_path}")
            continue

        filename = os.path.basename(img_path)
        count = trace_detection(processor, image, filename)
        results.append((filename, count))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for filename, count in results:
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {filename}: {count} faces detected")


if __name__ == "__main__":
    main()
