import cv2
import numpy as np
from backend.image_processor import ImageProcessor

def test_face_detection():
    print("=" * 50)
    print("测试人脸检测")
    print("=" * 50)
    
    processor = ImageProcessor()
    
    # 创建一个测试图像（模拟萨摩耶图片的背景）
    # 创建绿色背景（模拟树叶）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]  # 绿色
    
    # 添加一些纹理模拟树叶
    for _ in range(100):
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        size = np.random.randint(5, 20)
        cv2.circle(test_image, (x, y), size, (0, 100, 0), -1)
    
    # 测试人脸检测
    result = processor.face_detection(
        test_image,
        min_neighbors=6,
        min_size=(50, 50),
        scale_factor=1.1,
        require_eyes=True,
        require_skin=True,
        confidence_threshold=0.5
    )
    
    # 检查是否有绿色矩形框（人脸标记）
    has_green_rect = False
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if result[y, x, 0] == 0 and result[y, x, 1] == 255 and result[y, x, 2] == 0:
                has_green_rect = True
                break
        if has_green_rect:
            break
    
    if has_green_rect:
        print("❌ 测试失败：绿色背景（树叶）被误检为人脸")
        return False
    else:
        print("✅ 测试通过：绿色背景（树叶）未被误检为人脸")
        return True

def test_object_detection():
    print("\n" + "=" * 50)
    print("测试目标检测")
    print("=" * 50)
    
    processor = ImageProcessor()
    
    # 创建一个测试图像（模拟狗的形状）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [200, 200, 200]  # 灰色背景
    
    # 创建一个白色区域模拟狗
    cv2.rectangle(test_image, (200, 100), (400, 350), (255, 255, 255), -1)
    cv2.circle(test_image, (250, 150), 20, (0, 0, 0), -1)  # 左眼
    cv2.circle(test_image, (350, 150), 20, (0, 0, 0), -1)  # 右眼
    cv2.ellipse(test_image, (300, 200), (30, 20), 0, 0, 180, (0, 0, 0), -1)  # 嘴巴
    
    # 测试目标检测
    result = processor.object_detection(test_image, confidence_threshold=0.3)
    
    # 检查是否有红色矩形框（目标标记）
    has_red_rect = False
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if result[y, x, 0] == 0 and result[y, x, 1] == 0 and result[y, x, 2] == 255:
                has_red_rect = True
                break
        if has_red_rect:
            break
    
    if has_red_rect:
        print("✅ 测试通过：检测到目标并标注红色矩形框")
        return True
    else:
        print("❌ 测试失败：未检测到目标")
        return False

if __name__ == "__main__":
    face_result = test_face_detection()
    object_result = test_object_detection()
    
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"人脸检测：{'通过' if face_result else '失败'}")
    print(f"目标检测：{'通过' if object_result else '失败'}")
    
    if face_result and object_result:
        print("\n✅ 所有测试通过，可以启动应用")
    else:
        print("\n❌ 测试失败，需要修复问题")
