# utils.py
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import cv2
import os

def load_models():
    """加载本地censor检测模型"""
    try:
        # 尝试加载.pt文件
        pt_model_path = "models/model.pt"
        if os.path.exists(pt_model_path):
            censor_model = YOLO(pt_model_path)
            print(f"成功加载模型: {pt_model_path}")
            return None, censor_model
        else:
            print(f"模型文件不存在: {pt_model_path}")
            return None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def detect_censors(image_path, detection_model):
    """使用YOLO模型检测图像中的马赛克区域"""
    if detection_model is None:
        return []
    try:
        # 使用YOLO进行检测
        results = detection_model(image_path, conf=0.25, iou=0.7, verbose=False)
        
        detected_objects = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1,y1,x2,y2)
                conf = result.boxes.conf.cpu().numpy()   # 置信度
                cls = result.boxes.cls.cpu().numpy()     # 类别
                
                # 获取类别名称
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = conf[i]
                    class_id = int(cls[i])
                    class_name = names.get(class_id, f"class_{class_id}")
                    
                    # 返回格式: (边界框, 标签, 置信度)
                    detected_objects.append(((x1, y1, x2, y2), class_name, confidence))
        
        return detected_objects
    except Exception as e:
        print(f"Error detecting censors: {e}")
        return []

def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 1:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image

def apply_blur_mosaic(image_cv, box, kernel_size=(31, 31)):
    """应用常规模糊马赛克到指定边界框区域"""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    output_image = image_cv.copy()
    
    # 提取区域
    roi = output_image[y1:y2, x1:x2].copy()
    
    # 对区域进行模糊
    if roi.size > 0:
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
        output_image[y1:y2, x1:x2] = blurred_roi
    
    return output_image

def apply_black_lines_mosaic(image_cv, box, line_thickness=5, spacing=10, direction='horizontal'):
    """应用动漫风格黑色线条马赛克"""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    output_image = image_cv.copy()
    pil_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    if direction == 'horizontal':
        for y in range(y1, y2, spacing):
            draw.line([(x1, y), (x2, y)], fill="black", width=line_thickness)
    elif direction == 'vertical':
        for x in range(x1, x2, spacing):
            draw.line([(x, y1), (x, y2)], fill="black", width=line_thickness)
    elif direction == 'diagonal':
        # 斜线马赛克
        for i in range(-max(x2-x1, y2-y1), max(x2-x1, y2-y1), spacing):
            x_start = max(x1, x1 + i)
            y_start = max(y1, y1 - i)
            x_end = min(x2, x2 + i)
            y_end = min(y2, y2 - i)
            if x_start < x_end and y_start < y_end:
                draw.line([(x_start, y_start), (x_end, y_end)], fill="black", width=line_thickness)
    
    output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return output_image

def apply_white_mist_mosaic(image_cv, box, strength=0.8):
    """应用白色雾气马赛克"""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    output_image = image_cv.copy()
    
    # 确保区域在图像范围内
    h, w = output_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return output_image
    
    # 创建白色覆盖层
    roi = output_image[y1:y2, x1:x2].copy()
    white_layer = np.full_like(roi, 255, dtype=np.uint8)
    
    # 混合
    blended = cv2.addWeighted(roi, 1.0 - strength, white_layer, strength, 0)
    output_image[y1:y2, x1:x2] = blended
    
    return output_image

def apply_custom_image_mosaic(image_cv_rgb, box, custom_image_rgba_np):
    """应用自定义图像马赛克"""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    output_image_rgb = image_cv_rgb.copy()
    
    # 确保区域在图像范围内
    h, w = output_image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return output_image_rgb
    
    # 计算目标区域大小
    target_width = x2 - x1
    target_height = y2 - y1
    
    if target_width <= 0 or target_height <= 0:
        return output_image_rgb
    
    # 将自定义图像缩放到目标大小
    resized_custom_image = cv2.resize(custom_image_rgba_np, (target_width, target_height))
    
    # 分离RGB和Alpha通道
    custom_rgb = resized_custom_image[:, :, :3]
    if resized_custom_image.shape[2] == 4:
        custom_alpha = resized_custom_image[:, :, 3:4].astype(np.float32) / 255.0
    else:
        custom_alpha = np.ones((target_height, target_width, 1), dtype=np.float32)
    
    # 应用Alpha混合
    roi = output_image_rgb[y1:y2, x1:x2].astype(np.float32)
    custom_rgb = custom_rgb.astype(np.float32)
    
    blended = custom_alpha * custom_rgb + (1 - custom_alpha) * roi
    output_image_rgb[y1:y2, x1:x2] = blended.astype(np.uint8)
    
    return output_image_rgb