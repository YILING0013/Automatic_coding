# image_processor.py
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from imagecodecs import imread, imwrite

from utils import (
    load_models, detect_censors, to_rgb, to_rgba,
    apply_blur_mosaic, apply_black_lines_mosaic, apply_white_mist_mosaic,
    apply_custom_image_mosaic, apply_light_mosaic, get_available_labels
)

# IS_PACKAGED_APP 如果为 True，则表示作为可执行文件运行 (例如由 PyInstaller 打包)
# 否则表示作为普通 Python 脚本运行。
IS_PACKAGED_APP = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
if IS_PACKAGED_APP:
    print("INFO: Application is running in packaged mode.")
else:
    print("INFO: Application is running in script mode.")

classification_model, detection_model = load_models()

DEFAULT_HEAD_PATH = "assets/head.png"

def _load_image_data_rgb(image_path_str: str) -> np.ndarray:
    """
    加载图像文件并确保其为 RGB 格式的 NumPy 数组。
    根据程序运行模式（打包或脚本）选择不同的加载库。
    """
    if not os.path.exists(image_path_str):
        raise FileNotFoundError(f"Image file not found: {image_path_str}")

    if IS_PACKAGED_APP:
        # 打包模式：使用 OpenCV
        img_cv = cv2.imread(image_path_str)
        if img_cv is None:
            raise IOError(f"OpenCV (cv2.imread) failed to load image: {image_path_str}")
        # OpenCV 默认加载为 BGR，转换为 RGB
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    else:
        # 脚本模式：使用 imagecodecs
        img_ic = imread(image_path_str)
        if img_ic is None:
            raise IOError(f"imagecodecs.imread failed to load image: {image_path_str}")
        
        if img_ic.ndim == 2:  # Grayscale
            return cv2.cvtColor(img_ic, cv2.COLOR_GRAY2RGB)
        elif img_ic.ndim == 3 and img_ic.shape[2] == 4:  # RGBA
            return cv2.cvtColor(img_ic, cv2.COLOR_RGBA2RGB) # 转换为 RGB (去除 alpha)
        elif img_ic.ndim == 3 and img_ic.shape[2] == 3:  # RGB
            return img_ic
        else:
            raise ValueError(f"Unsupported image format from imagecodecs for RGB conversion: {image_path_str}, shape: {img_ic.shape}")

def _load_image_data_rgba(image_path_str: str) -> np.ndarray:
    """
    加载图像文件并确保其为 RGBA 格式的 NumPy 数组。
    用于需要 alpha 通道的图像（例如自定义贴图）。
    根据程序运行模式选择不同的加载库。
    """
    if not os.path.exists(image_path_str):
        raise FileNotFoundError(f"Image file not found: {image_path_str}")

    if IS_PACKAGED_APP:
        # 打包模式：使用 OpenCV，并尝试保留 alpha 通道
        img_cv = cv2.imread(image_path_str, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            raise IOError(f"OpenCV (cv2.imread) failed to load image: {image_path_str}")
        
        if img_cv.ndim == 2:  # Grayscale
            return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGBA)
        elif img_cv.ndim == 3 and img_cv.shape[2] == 3:  # BGR (无 alpha)
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGBA)
        elif img_cv.ndim == 3 and img_cv.shape[2] == 4:  # BGRA
            return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unsupported image format from OpenCV for RGBA conversion: {image_path_str}, shape: {img_cv.shape}")
    else:
        # 脚本模式：使用 imagecodecs
        img_ic = imread(image_path_str)
        if img_ic is None:
            raise IOError(f"imagecodecs.imread failed to load image: {image_path_str}")

        if img_ic.ndim == 2:
            return cv2.cvtColor(img_ic, cv2.COLOR_GRAY2RGBA)
        elif img_ic.ndim == 3 and img_ic.shape[2] == 3:  # RGB
            return cv2.cvtColor(img_ic, cv2.COLOR_RGB2RGBA)
        elif img_ic.ndim == 3 and img_ic.shape[2] == 4:  # RGBA
            return img_ic
        else:
            raise ValueError(f"Unsupported image format from imagecodecs for RGBA conversion: {image_path_str}, shape: {img_ic.shape}")

def get_default_custom_image():
    """获取默认的自定义贴图 (RGBA格式)"""
    try:
        if not os.path.exists(DEFAULT_HEAD_PATH):
            print(f"错误: 默认自定义贴图路径不存在: {DEFAULT_HEAD_PATH}")
            return np.zeros((50, 50, 4), dtype=np.uint8)
        
        return _load_image_data_rgba(DEFAULT_HEAD_PATH)
    except Exception as e:
        print(f"加载默认自定义图像时发生错误 ({DEFAULT_HEAD_PATH}): {e}")
        return np.zeros((50, 50, 4), dtype=np.uint8) # 错误时的占位符

def process_single_image(image_path, mosaic_type, selected_regions, custom_image_path=None, 
                         line_direction='horizontal', conf_threshold=0.25, iou_threshold=0.7,
                         scale=1.0, alpha=1.0, blur_kernel_size=(31, 31), 
                         line_thickness=5, line_spacing=10, 
                         mist_color=(255, 255, 255), # RGB
                         light_intensity=0.8, light_feather=30, light_color=(255, 255, 255), # RGB
                         cached_detection_results=None):
    """
    处理单张图片。
    现在使用条件逻辑加载主图像和自定义图像。
    """
    try:
        original_image = _load_image_data_rgb(image_path)
        
        if not detection_model:
            return Image.fromarray(original_image), None, "错误：检测模型未能成功加载。"
        
        if cached_detection_results is not None:
            detection_results = cached_detection_results
        else:
            detection_results = detect_censors(image_path, detection_model, conf_threshold, iou_threshold)
        
        filtered_boxes = []
        if detection_results:
            for result in detection_results:
                bbox, label, confidence = result
                if not selected_regions or label in selected_regions:
                    filtered_boxes.append(bbox)
        
        if not filtered_boxes:
            return Image.fromarray(original_image), Image.fromarray(original_image), "未检测到需要打码的区域。"
        
        processed_image_np = original_image.copy() # 对 NumPy 数组进行操作
        
        custom_img_to_apply_np = None
        if mosaic_type == "自定义图像":
            if custom_image_path and os.path.exists(custom_image_path):
                try:
                    custom_img_to_apply_np = _load_image_data_rgba(custom_image_path)
                except Exception as e_custom:
                    print(f"加载自定义图像 '{custom_image_path}' 失败: {e_custom}。将使用默认图像。")
                    custom_img_to_apply_np = get_default_custom_image()
            else:
                if custom_image_path: # 提供了路径但文件不存在
                    print(f"警告: 自定义图像路径 '{custom_image_path}' 不存在。将使用默认图像。")
                custom_img_to_apply_np = get_default_custom_image()

        for box in filtered_boxes:
            processed_image_bgr = cv2.cvtColor(processed_image_np, cv2.COLOR_RGB2BGR)

            if mosaic_type == "常规模糊":
                processed_image_bgr = apply_blur_mosaic(processed_image_bgr, box, 
                                                       kernel_size=blur_kernel_size, 
                                                       scale=scale, alpha=alpha)
            elif mosaic_type == "黑色线条":
                processed_image_bgr = apply_black_lines_mosaic(processed_image_bgr, box, 
                                                              line_thickness=line_thickness, 
                                                              spacing=line_spacing,
                                                              scale=scale, 
                                                              direction=line_direction,
                                                              alpha=alpha)
            elif mosaic_type == "白色雾气":
                bgr_mist_color = (mist_color[2], mist_color[1], mist_color[0]) # RGB to BGR
                processed_image_bgr = apply_white_mist_mosaic(processed_image_bgr, box, 
                                                             strength=alpha, # alpha作雾气强度
                                                             scale=scale,
                                                             color=bgr_mist_color)
            elif mosaic_type == "光效马赛克":
                bgr_light_color = (light_color[2], light_color[1], light_color[0]) # RGB to BGR
                processed_image_bgr = apply_light_mosaic(processed_image_bgr, box, 
                                                        intensity=light_intensity,
                                                        feather=light_feather,
                                                        color=bgr_light_color,
                                                        scale=scale)
            elif mosaic_type == "自定义图像":
                if custom_img_to_apply_np is not None:
                    processed_image_np = apply_custom_image_mosaic(processed_image_np, box, custom_img_to_apply_np,
                                                               scale=scale, alpha=alpha if alpha is not None else 1.0)
                    continue # 跳过最后的 BGR to RGB 转换，因为 processed_image_np 已被更新
                else:
                    print("警告：自定义图像未能加载，此区域未应用自定义图像。")


            if mosaic_type != "自定义图像":
                 processed_image_np = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(original_image), Image.fromarray(processed_image_np), None

    except FileNotFoundError as e_fnf: # 特定处理文件未找到错误
        print(f"处理图像时发生文件未找到错误 ({image_path}): {e_fnf}")
        return None, None, f"文件未找到: {e_fnf}"
    except IOError as e_io: # 特定处理图像读写错误
        print(f"处理图像时发生IO错误 ({image_path}): {e_io}")
        return None, None, f"图像读取错误: {e_io}"
    except Exception as e:
        import traceback
        print(f"处理图像时发生未知错误 ({image_path}): {e}")
        traceback.print_exc()
        try:
            if isinstance(original_image, np.ndarray):
                pil_original = Image.fromarray(original_image)
            else:
                pil_original = original_image if original_image is not None else Image.new("RGB", (100,100), "pink") # 最后的备用
            return pil_original, None, f"处理图像时发生错误: {e}"
        except: # 如果连原始图像都无法返回
             return None, None, f"处理图像时发生严重错误: {e}"


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
                         mist_color=(255, 255, 255), # RGB
                         light_intensity=0.8, light_feather=30, light_color=(255, 255, 255), # RGB
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

    detection_cache = {}

    for i, file_path in enumerate(files_to_process):
        if status_callback:
            status_callback(f"正在处理: {file_path.name} ({i+1}/{total_files})")
        
        current_original_pil = None
        try:
            current_original_pil = Image.fromarray(_load_image_data_rgb(str(file_path)))
            if image_preview_callback:
                image_preview_callback(current_original_pil, None)
        except Exception as e_load_preview:
            print(f"批量处理中预览图像加载失败 ({file_path.name}): {e_load_preview}")
            if image_preview_callback: # 发送一个占位符
                error_placeholder = Image.new("RGB", (200, 200), "pink")
                try:
                    draw = ImageDraw.Draw(error_placeholder)
                    draw.text((10, 10), f"无法加载:\n{file_path.name[:20]}...", fill="black")
                except ImportError:
                    pass
                image_preview_callback(error_placeholder, None)


        file_path_str = str(file_path)
        cached_results = detection_cache.get(file_path_str)

        original_pil, processed_pil_image, error = process_single_image(
            file_path_str, mosaic_type, selected_regions, custom_image_path, line_direction,
            conf_threshold, iou_threshold, scale, alpha, blur_kernel_size,
            line_thickness, line_spacing, mist_color, light_intensity, light_feather, light_color,
            cached_detection_results=cached_results
        )
        
        if cached_results is None and not error : # 仅当处理成功且未使用缓存时才缓存结果
            try:
                _, new_detection_results, _ = get_image_object_names(file_path_str, conf_threshold, iou_threshold)
                if new_detection_results is not None:
                     detection_cache[file_path_str] = new_detection_results
            except Exception as e_cache_detect:
                print(f"为缓存重新检测时出错 ({file_path_str}): {e_cache_detect}")

        if image_preview_callback:
            image_preview_callback(original_pil if original_pil else current_original_pil, processed_pil_image)


        if processed_pil_image and not error:
            try:
                if input_path_obj.is_dir():
                    relative_path = file_path.relative_to(input_path_obj)
                    output_file_path = output_folder_obj / relative_path
                else:
                    output_file_path = output_folder_obj / file_path.name
                
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # processed_pil_image 是 PIL Image 对象
                save_format = output_file_path.suffix.lower()[1:]
                if save_format in ['jpg', 'jpeg'] and processed_pil_image.mode == 'RGBA':
                    # 对于JPEG，转换为RGB（去除Alpha）
                    processed_pil_image.convert('RGB').save(str(output_file_path))
                elif processed_pil_image.mode == 'P' and save_format in ['jpg', 'jpeg']: #调色板模式转RGB
                    processed_pil_image.convert('RGB').save(str(output_file_path))
                else:
                    processed_pil_image.save(str(output_file_path))

            except Exception as e_save:
                if status_callback:
                    status_callback(f"保存失败 {file_path.name}: {e_save}")
        elif error:
            if status_callback:
                status_callback(f"处理失败 {file_path.name}: {error}")
        
        if progress_callback:
            progress_callback(i + 1, total_files)

    if status_callback:
        status_callback(f"批量处理完成！已处理 {total_files} 个文件。")