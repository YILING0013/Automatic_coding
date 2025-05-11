# utils.py
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import cv2
import os
import colorsys

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

def detect_censors(image_path, detection_model, conf_threshold=0.25, iou_threshold=0.7):
    """使用YOLO模型检测图像中的马赛克区域
    
    Args:
        image_path: 图像路径
        detection_model: YOLO模型
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
    """
    if detection_model is None:
        return []
    try:
        # 使用YOLO进行检测，使用自定义阈值
        results = detection_model(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
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

def adjust_box_by_scale(box, scale, img_shape):
    """根据比例调整边界框
    
    Args:
        box: 边界框(x1, y1, x2, y2)
        scale: 比例因子，1为原始大小，>1扩大，<1缩小
        img_shape: 图像形状(高度, 宽度)
    
    Returns:
        调整后的边界框
    """
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    
    new_width = width * scale
    new_height = height * scale
    
    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(img_shape[1], center_x + new_width / 2)
    new_y2 = min(img_shape[0], center_y + new_height / 2)
    
    return (new_x1, new_y1, new_x2, new_y2)

def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        # 检查是否为BGR格式并转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 1:  # Gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image

def apply_blur_mosaic(image_cv, box, kernel_size=(31, 31), scale=1.0, alpha=1.0):
    """应用常规模糊马赛克到指定边界框区域
    
    Args:
        image_cv: 图像
        box: 边界框(x1, y1, x2, y2)
        kernel_size: 模糊内核大小
        scale: 区域缩放比例
        alpha: 不透明度
        
    Returns:
        处理后的图像
    """
    h, w = image_cv.shape[:2]
    box = adjust_box_by_scale(box, scale, (h, w))
    x1, y1, x2, y2 = [int(v) for v in box]
    
    output_image = image_cv.copy()
    
    # 提取区域
    roi = output_image[y1:y2, x1:x2].copy()
    
    # 对区域进行模糊
    if roi.size > 0:
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
        
        # 应用透明度
        if alpha < 1.0:
            output_image[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, blurred_roi, alpha, 0)
        else:
            output_image[y1:y2, x1:x2] = blurred_roi
    
    return output_image

def apply_black_lines_mosaic(image_cv, box, line_thickness=5, spacing=10, scale=1.0, direction='horizontal', alpha=1.0):
    """应用动漫风格黑色线条马赛克
    
    Args:
        image_cv: 图像
        box: 边界框
        line_thickness: 线条粗细
        spacing: 线条间距
        scale: 区域缩放比例
        direction: 线条方向('horizontal', 'vertical', 'diagonal')
        alpha: 不透明度
        
    Returns:
        处理后的图像
    """
    h, w = image_cv.shape[:2]
    box = adjust_box_by_scale(box, scale, (h, w))
    x1, y1, x2, y2 = [int(v) for v in box]
    
    output_image = image_cv.copy()
    
    # 创建带线条的图层
    lines_layer = output_image.copy()
    pil_image = Image.fromarray(cv2.cvtColor(lines_layer, cv2.COLOR_BGR2RGB))
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
    
    lines_layer = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 应用透明度
    if 0.0 < alpha < 1.0:
        # 只在边界框区域内应用透明度混合
        roi = output_image[y1:y2, x1:x2].copy()
        lines_roi = lines_layer[y1:y2, x1:x2].copy()
        output_image[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, lines_roi, alpha, 0)
    else:
        output_image = lines_layer
    
    return output_image

def apply_white_mist_mosaic(image_cv, box, strength=0.8, scale=1.0, color=(255, 255, 255)):
    """应用雾气马赛克
    
    Args:
        image_cv: 图像
        box: 边界框
        strength: 雾气强度(0-1)
        scale: 区域缩放比例
        color: 雾气颜色(B,G,R)
        
    Returns:
        处理后的图像
    """
    h, w = image_cv.shape[:2]
    box = adjust_box_by_scale(box, scale, (h, w))
    x1, y1, x2, y2 = [int(v) for v in box]
    
    output_image = image_cv.copy()
    
    # 确保区域在图像范围内
    h, w = output_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return output_image
    
    # 创建覆盖层
    roi = output_image[y1:y2, x1:x2].copy()
    color_layer = np.full_like(roi, color, dtype=np.uint8)
    
    # 混合
    blended = cv2.addWeighted(roi, 1.0 - strength, color_layer, strength, 0)
    output_image[y1:y2, x1:x2] = blended
    
    return output_image

def apply_custom_image_mosaic(image_cv_rgb, box, custom_image_rgba_np, scale=1.0, alpha=None):
    """应用自定义图像马赛克
    
    Args:
        image_cv_rgb: RGB图像
        box: 边界框
        custom_image_rgba_np: RGBA自定义图像
        scale: 区域缩放比例
        alpha: 覆盖强度，若为None则使用图像自身alpha通道
        
    Returns:
        处理后的RGB图像
    """
    h, w = image_cv_rgb.shape[:2]
    box = adjust_box_by_scale(box, scale, (h, w))
    x1, y1, x2, y2 = [int(v) for v in box]
    
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
        if alpha is None:
            custom_alpha = resized_custom_image[:, :, 3:4].astype(np.float32) / 255.0
        else:
            # 使用用户指定的alpha值
            custom_alpha = np.ones((target_height, target_width, 1), dtype=np.float32) * alpha
    else:
        if alpha is None:
            custom_alpha = np.ones((target_height, target_width, 1), dtype=np.float32)
        else:
            custom_alpha = np.ones((target_height, target_width, 1), dtype=np.float32) * alpha
    
    # 应用Alpha混合
    roi = output_image_rgb[y1:y2, x1:x2].astype(np.float32)
    custom_rgb = custom_rgb.astype(np.float32)
    
    blended = custom_alpha * custom_rgb + (1 - custom_alpha) * roi
    output_image_rgb[y1:y2, x1:x2] = blended.astype(np.uint8)
    
    return output_image_rgb

def apply_light_mosaic(image_cv, box, intensity=0.8, feather=30, color=(255, 255, 255), scale=1.0):
    """应用炫光马赛克效果
    
    Args:
        image_cv: 图像
        box: 边界框
        intensity: 光强度(0-1)
        feather: 羽化边缘像素数
        color: 光颜色(B,G,R)
        scale: 区域缩放比例
        
    Returns:
        处理后的图像
    """
    h, w = image_cv.shape[:2]
    box = adjust_box_by_scale(box, scale, (h, w))
    x1, y1, x2, y2 = [int(v) for v in box]
    
    output_image = image_cv.copy()
    
    # 确保区域在图像范围内
    h, w = output_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return output_image
    
    # 创建蒙版，中心亮，边缘暗
    mask = np.zeros((y2-y1, x2-x1), dtype=np.float32)
    center_x, center_y = (x2-x1)//2, (y2-y1)//2
    
    # 生成径向渐变
    y_indices, x_indices = np.ogrid[:y2-y1, :x2-x1]
    dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    mask = np.clip(1.0 - dist_from_center / max_dist, 0, 1)
    mask = np.power(mask, 1.0 / (feather / 100.0 + 0.1))
    
    roi = output_image[y1:y2, x1:x2].copy()
    
    light_layer = np.full_like(roi, color, dtype=np.uint8)
    
    for c in range(3):
        roi_channel = roi[:,:,c].astype(np.float32)
        light_channel = light_layer[:,:,c].astype(np.float32)
        
        blended_channel = roi_channel * (1.0 - mask * intensity) + light_channel * (mask * intensity)
        roi[:,:,c] = np.clip(blended_channel, 0, 255).astype(np.uint8)
    
    output_image[y1:y2, x1:x2] = roi
    
    return output_image

def get_available_labels():
    """获取可用的标签列表"""
    return ["nipple_f", "penis", "pussy"]