# image_processor.py
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from imagecodecs import imread, imwrite
from utils import (
    load_models, detect_censors, to_rgb, to_rgba,
    apply_blur_mosaic, apply_black_lines_mosaic, apply_white_mist_mosaic,
    apply_custom_image_mosaic, apply_light_mosaic, get_available_labels
)

classification_model, detection_model = load_models()

DEFAULT_HEAD_PATH = "assets/head.png"

def get_default_custom_image():
    try:
        return to_rgba(imread(DEFAULT_HEAD_PATH))
    except Exception as e:
        print(f"Error loading default custom image: {e}")
        return np.zeros((50, 50, 4), dtype=np.uint8)

def process_single_image(image_path, mosaic_type, selected_regions, custom_image_path=None, 
                         line_direction='horizontal', conf_threshold=0.25, iou_threshold=0.7,
                         scale=1.0, alpha=1.0, blur_kernel_size=(31, 31), 
                         line_thickness=5, line_spacing=10, 
                         mist_color=(255, 255, 255),
                         light_intensity=0.8, light_feather=30, light_color=(255, 255, 255),
                         cached_detection_results=None):
    """
    处理单张图片
    :param image_path: 图像文件路径
    :param mosaic_type: 马赛克类型
    :param selected_regions: 用户选择要打码的区域名称列表
    :param custom_image_path: 自定义马赛克图像的路径
    :param line_direction: 线条方向 ('horizontal', 'vertical', 'diagonal')
    :param conf_threshold: 模型置信度阈值
    :param iou_threshold: 模型IOU阈值
    :param scale: 马赛克区域缩放比例
    :param alpha: 马赛克透明度
    :param blur_kernel_size: 模糊内核大小
    :param line_thickness: 线条粗细
    :param line_spacing: 线条间距
    :param mist_color: 雾气颜色(R,G,B)
    :param light_intensity: 光效强度
    :param light_feather: 光效羽化边缘
    :param light_color: 光效颜色(R,G,B)
    :param cached_detection_results: 缓存的检测结果，如果提供则使用而不重新检测
    :return: (original_pil_image, processed_pil_image, error_message)
    """
    try:
        # 读取图像
        original_image = to_rgb(imread(image_path))
        
        if not detection_model:
            return Image.fromarray(original_image), None, "错误：检测模型未能成功加载。"
        
        # 使用缓存的结果或进行新的检测
        if cached_detection_results is not None:
            detection_results = cached_detection_results
        else:
            # 使用检测模型检测边界框，传入自定义阈值
            detection_results = detect_censors(image_path, detection_model, conf_threshold, iou_threshold)
        
        filtered_boxes = []
        if detection_results:
            for result in detection_results:
                bbox, label, confidence = result
                # 如果用户选择了特定区域，进行过滤
                if not selected_regions or label in selected_regions:
                    filtered_boxes.append(bbox)
        
        if not filtered_boxes:
            return Image.fromarray(original_image), Image.fromarray(original_image), "未检测到需要打码的区域。"
        
        # 应用选择的马赛克类型
        processed_image = original_image.copy()
        
        # 为每个边界框应用马赛克
        for box in filtered_boxes:
            if mosaic_type == "常规模糊":
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                processed_image_bgr = apply_blur_mosaic(processed_image_bgr, box, 
                                                       kernel_size=blur_kernel_size, 
                                                       scale=scale, alpha=alpha)
                processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
            elif mosaic_type == "黑色线条":
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                processed_image_bgr = apply_black_lines_mosaic(processed_image_bgr, box, 
                                                              line_thickness=line_thickness, 
                                                              spacing=line_spacing,
                                                              scale=scale, 
                                                              direction=line_direction,
                                                              alpha=alpha)
                processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
            elif mosaic_type == "白色雾气":
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                # 转换颜色格式从RGB到BGR
                bgr_color = (mist_color[2], mist_color[1], mist_color[0])
                processed_image_bgr = apply_white_mist_mosaic(processed_image_bgr, box, 
                                                             strength=alpha,
                                                             scale=scale,
                                                             color=bgr_color)
                processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
            elif mosaic_type == "光效马赛克":
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                # 转换颜色格式从RGB到BGR
                bgr_color = (light_color[2], light_color[1], light_color[0])
                processed_image_bgr = apply_light_mosaic(processed_image_bgr, box, 
                                                        intensity=light_intensity,
                                                        feather=light_feather,
                                                        color=bgr_color,
                                                        scale=scale)
                processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
            elif mosaic_type == "自定义图像":
                custom_img_np = None
                if custom_image_path and os.path.exists(custom_image_path):
                    custom_img_np = to_rgba(imread(custom_image_path))
                else:
                    custom_img_np = get_default_custom_image()
                
                processed_image = apply_custom_image_mosaic(processed_image, box, custom_img_np,
                                                           scale=scale, alpha=alpha)
        
        return Image.fromarray(original_image), Image.fromarray(processed_image), None

    except Exception as e:
        import traceback
        print(f"处理图像时发生错误 ({image_path}): {e}")
        traceback.print_exc()
        return None, None, f"处理图像时发生错误: {e}"

def get_image_object_names(image_path, conf_threshold=0.25, iou_threshold=0.7):
    """获取图像中可识别的目标类型列表，并返回检测结果"""
    if not detection_model:
        return [], None, "错误：检测模型未加载。"
    try:
        results = detect_censors(image_path, detection_model, conf_threshold, iou_threshold)
        detected_names = set()
        
        if results:
            for result in results:
                _, label, _ = result
                if label:
                    detected_names.add(label)
        
        return sorted(list(detected_names)), results, None
    except Exception as e:
        return [], None, f"分析图像时出错: {e}"

def batch_process_images(input_path, output_folder_path, mosaic_type, selected_regions, 
                         custom_image_path=None, line_direction='horizontal',
                         conf_threshold=0.25, iou_threshold=0.7,
                         scale=1.0, alpha=1.0, blur_kernel_size=(31, 31), 
                         line_thickness=5, line_spacing=10, 
                         mist_color=(255, 255, 255),
                         light_intensity=0.8, light_feather=30, light_color=(255, 255, 255),
                         progress_callback=None, status_callback=None, image_preview_callback=None):
    """批量处理图像"""
    input_path_obj = Path(input_path)
    output_folder_obj = Path(output_folder_path)
    output_folder_obj.mkdir(parents=True, exist_ok=True)

    if input_path_obj.is_file():
        files_to_process = [input_path_obj]
    elif input_path_obj.is_dir():
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        files_to_process = [p for p in input_path_obj.rglob('*') if p.suffix.lower() in supported_extensions and p.is_file()]
    else:
        if status_callback:
            status_callback(f"错误：输入路径无效: {input_path}")
        return

    total_files = len(files_to_process)
    if status_callback:
        status_callback(f"开始处理 {total_files} 个文件...")

    # 批量处理中的检测结果缓存字典
    detection_cache = {}

    for i, file_path in enumerate(files_to_process):
        if status_callback:
            status_callback(f"正在处理: {file_path.name} ({i+1}/{total_files})")
        
        # 如果提供了预览回调，发送当前图像到预览区
        if image_preview_callback:
            try:
                current_image = Image.open(file_path)
                image_preview_callback(current_image, None)
            except Exception as e:
                print(f"预览图像错误: {e}")

        # 检查是否已存在此文件的检测结果
        file_path_str = str(file_path)
        cached_results = detection_cache.get(file_path_str)

        original_pil, processed_pil_image, error = process_single_image(
            file_path_str, mosaic_type, selected_regions, custom_image_path, line_direction,
            conf_threshold, iou_threshold, scale, alpha, blur_kernel_size,
            line_thickness, line_spacing, mist_color, light_intensity, light_feather, light_color,
            cached_detection_results=cached_results
        )
        
        # 如果没有使用缓存，则缓存新的检测结果
        if cached_results is None:
            detection_cache[file_path_str] = detect_censors(file_path_str, detection_model, conf_threshold, iou_threshold)
        
        # 更新处理后的图像预览
        if image_preview_callback and processed_pil_image:
            image_preview_callback(original_pil, processed_pil_image)

        if processed_pil_image and not error:
            try:
                if input_path_obj.is_dir():
                    relative_path = file_path.relative_to(input_path_obj)
                    output_file_path = output_folder_obj / relative_path
                else:
                    output_file_path = output_folder_obj / file_path.name
                
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 修复颜色空间问题：直接从PIL图像保存，避免额外的颜色空间转换
                processed_pil_image.save(str(output_file_path))

            except Exception as e:
                if status_callback:
                    status_callback(f"保存失败 {file_path.name}: {e}")
        elif error:
            if status_callback:
                status_callback(f"处理失败 {file_path.name}: {error}")
        
        if progress_callback:
            progress_callback(i + 1, total_files)

    if status_callback:
        status_callback(f"批量处理完成！已处理 {total_files} 个文件。")