# main_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import os
from pathlib import Path
import threading

from image_processor import (
    process_single_image,
    batch_process_images,
    get_image_object_names,
    get_default_custom_image,
    get_available_labels,
    DEFAULT_HEAD_PATH
)
from utils import detect_censors, load_models

# 全局变量
current_image_path = None
original_pil_image = None
processed_pil_image = None
custom_mosaic_image_path = None
custom_mosaic_pil_image = None

# 预览区域的固定尺寸
PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 1250

# 加载模型
classification_model, detection_model = load_models()
if not detection_model:
    print("警告：检测模型加载失败，应用功能将受限。")

class ImageMosaicApp(tb.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("图像打码工具")
        self.geometry("1250x900")

        # 检测结果缓存相关变量
        self.cached_detection_results = None  # 缓存当前图像的检测结果
        self.last_detection_conf = None       # 上次检测使用的置信度阈值
        self.last_detection_iou = None        # 上次检测使用的IOU阈值

        # 数据
        self.input_path = tk.StringVar()
        self.output_folder = tk.StringVar(value=str(Path.home() / "图像打码输出"))
        self.mosaic_type_var = tk.StringVar(value="常规模糊")
        self.available_regions = []
        self.selected_regions_vars = {}
        self.custom_image_path_var = tk.StringVar(value="使用默认图案")
        self.line_direction_var = tk.StringVar(value="horizontal")
        
        # 模型参数
        self.conf_threshold_var = tk.DoubleVar(value=0.25)
        self.iou_threshold_var = tk.DoubleVar(value=0.7)
        
        # 马赛克参数
        self.scale_var = tk.DoubleVar(value=1.0)
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.blur_kernel_size_var = tk.IntVar(value=31)
        self.line_thickness_var = tk.IntVar(value=5)
        self.line_spacing_var = tk.IntVar(value=10)
        self.mist_color_var = tk.StringVar(value="#FFFFFF")
        self.light_intensity_var = tk.DoubleVar(value=0.8)
        self.light_feather_var = tk.IntVar(value=30)
        self.light_color_var = tk.StringVar(value="#FFFFFF")

        # 主框架
        main_frame = tb.Frame(self, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # 左侧：控制面板
        controls_frame = tb.Labelframe(main_frame, text="控制面板", padding=10)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # 1. 文件/文件夹选择
        file_frame = tb.Frame(controls_frame)
        file_frame.pack(fill=X, pady=5)
        tb.Button(file_frame, text="选择图片/文件夹", command=self.select_input_path, bootstyle=PRIMARY).pack(side=LEFT, padx=5)
        tb.Entry(file_frame, textvariable=self.input_path, state="readonly").pack(side=LEFT, fill=X, expand=YES)

        # 2. 模型参数设置
        model_params_frame = tb.Labelframe(controls_frame, text="模型参数", padding=5)
        model_params_frame.pack(fill=X, pady=5)
        
        # 置信度阈值
        tb.Label(model_params_frame, text="置信度阈值:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.conf_value_label = tb.Label(model_params_frame, text=f"{self.conf_threshold_var.get():.2f}", width=4)
        self.conf_value_label.grid(row=0, column=2, padx=5, pady=2)
        conf_scale = tb.Scale(model_params_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var, 
                            orient=HORIZONTAL, bootstyle="success", command=self.update_conf_label)
        conf_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        
        # IOU阈值
        tb.Label(model_params_frame, text="IOU阈值:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.iou_value_label = tb.Label(model_params_frame, text=f"{self.iou_threshold_var.get():.2f}", width=4)
        self.iou_value_label.grid(row=1, column=2, padx=5, pady=2)
        iou_scale = tb.Scale(model_params_frame, from_=0.1, to=1.0, variable=self.iou_threshold_var, 
                           orient=HORIZONTAL, bootstyle="success", command=self.update_iou_label)
        iou_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        
        model_params_frame.columnconfigure(1, weight=1)

        # 3. 打码方式选择
        mosaic_frame = tb.Labelframe(controls_frame, text="选择打码方式", padding=5)
        mosaic_frame.pack(fill=X, pady=10)
        mosaic_options = ["常规模糊", "黑色线条", "白色雾气", "光效马赛克", "自定义图像"]
        for option in mosaic_options:
            rb = tb.Radiobutton(mosaic_frame, text=option, variable=self.mosaic_type_var, value=option, command=self.on_mosaic_type_change)
            rb.pack(anchor=W)
        
        # 马赛克通用参数
        self.common_params_frame = tb.Labelframe(mosaic_frame, text="通用参数", padding=5)
        self.common_params_frame.pack(fill=X, pady=5)
        
        # 区域缩放
        tb.Label(self.common_params_frame, text="区域缩放:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.scale_value_label = tb.Label(self.common_params_frame, text=f"{self.scale_var.get():.2f}", width=4)
        self.scale_value_label.grid(row=0, column=2, padx=5, pady=2)
        scale_scale = tb.Scale(self.common_params_frame, from_=0.5, to=2.0, variable=self.scale_var, 
                     orient=HORIZONTAL, bootstyle="info", command=self.update_scale_label)
        scale_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        
        # 透明度
        tb.Label(self.common_params_frame, text="透明度:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.alpha_value_label = tb.Label(self.common_params_frame, text=f"{self.alpha_var.get():.2f}", width=4)
        self.alpha_value_label.grid(row=1, column=2, padx=5, pady=2)
        alpha_scale = tb.Scale(self.common_params_frame, from_=0.0, to=1.0, variable=self.alpha_var, 
                     orient=HORIZONTAL, bootstyle="info", command=self.update_alpha_label)
        alpha_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        
        self.common_params_frame.columnconfigure(1, weight=1)
        
        # 模糊参数
        self.blur_params_frame = tb.Labelframe(mosaic_frame, text="模糊参数", padding=5)
        
        # 模糊大小
        tb.Label(self.blur_params_frame, text="模糊大小:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.blur_value_label = tb.Label(self.blur_params_frame, text=f"{self.blur_kernel_size_var.get()}", width=4)
        self.blur_value_label.grid(row=0, column=2, padx=5, pady=2)
        blur_kernel_scale = tb.Scale(self.blur_params_frame, from_=3, to=101, variable=self.blur_kernel_size_var, 
                           orient=HORIZONTAL, bootstyle="warning", command=self.update_blur_label)
        blur_kernel_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        
        self.blur_params_frame.columnconfigure(1, weight=1)
        
        # 线条方向选择
        self.line_params_frame = tb.Labelframe(mosaic_frame, text="线条参数", padding=5)
        
        # 线条方向
        direction_frame = tb.Frame(self.line_params_frame)
        direction_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky=EW)
        tb.Label(direction_frame, text="方向:").pack(side=LEFT)
        line_directions = [("水平", "horizontal"), ("垂直", "vertical"), ("斜线", "diagonal")]
        for text, value in line_directions:
            rb = tb.Radiobutton(direction_frame, text=text, variable=self.line_direction_var, value=value, 
                              command=self.on_line_direction_change)
            rb.pack(side=LEFT, padx=5)
        
        # 线条粗细
        tb.Label(self.line_params_frame, text="线条粗细:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.line_thickness_value_label = tb.Label(self.line_params_frame, text=f"{self.line_thickness_var.get()}", width=4)
        self.line_thickness_value_label.grid(row=1, column=2, padx=5, pady=2)
        line_thickness_scale = tb.Scale(self.line_params_frame, from_=1, to=20, variable=self.line_thickness_var,
                              orient=HORIZONTAL, bootstyle="warning", command=self.update_line_thickness_label)
        line_thickness_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        
        # 线条间距
        tb.Label(self.line_params_frame, text="线条间距:").grid(row=2, column=0, padx=5, pady=2, sticky=W)
        self.line_spacing_value_label = tb.Label(self.line_params_frame, text=f"{self.line_spacing_var.get()}", width=4)
        self.line_spacing_value_label.grid(row=2, column=2, padx=5, pady=2)
        line_spacing_scale = tb.Scale(self.line_params_frame, from_=5, to=50, variable=self.line_spacing_var,
                            orient=HORIZONTAL, bootstyle="warning", command=self.update_line_spacing_label)
        line_spacing_scale.grid(row=2, column=1, padx=5, pady=2, sticky=EW)
        
        self.line_params_frame.columnconfigure(1, weight=1)
        
        # 雾气参数
        self.mist_params_frame = tb.Labelframe(mosaic_frame, text="雾气参数", padding=5)
        
        tb.Label(self.mist_params_frame, text="雾气颜色:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.mist_color_button = tb.Button(self.mist_params_frame, text="选择颜色", 
                                         command=lambda: self.select_color(self.mist_color_var),
                                         bootstyle="secondary")
        self.mist_color_button.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        self.mist_color_preview = tb.Label(self.mist_params_frame, background=self.mist_color_var.get(), width=4)
        self.mist_color_preview.grid(row=0, column=2, padx=5, pady=2)
        
        self.mist_params_frame.columnconfigure(1, weight=1)
        
        # 光效参数
        self.light_params_frame = tb.Labelframe(mosaic_frame, text="光效参数", padding=5)
        
        # 光强度
        tb.Label(self.light_params_frame, text="光强度:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.light_intensity_value_label = tb.Label(self.light_params_frame, text=f"{self.light_intensity_var.get():.2f}", width=4)
        self.light_intensity_value_label.grid(row=0, column=2, padx=5, pady=2)
        light_intensity_scale = tb.Scale(self.light_params_frame, from_=0.1, to=1.0, variable=self.light_intensity_var,
                               orient=HORIZONTAL, bootstyle="warning", command=self.update_light_intensity_label)
        light_intensity_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        
        # 羽化边缘
        tb.Label(self.light_params_frame, text="羽化边缘:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.light_feather_value_label = tb.Label(self.light_params_frame, text=f"{self.light_feather_var.get()}", width=4)
        self.light_feather_value_label.grid(row=1, column=2, padx=5, pady=2)
        light_feather_scale = tb.Scale(self.light_params_frame, from_=5, to=100, variable=self.light_feather_var,
                             orient=HORIZONTAL, bootstyle="warning", command=self.update_light_feather_label)
        light_feather_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        
        tb.Label(self.light_params_frame, text="光颜色:").grid(row=2, column=0, padx=5, pady=2, sticky=W)
        self.light_color_button = tb.Button(self.light_params_frame, text="选择颜色", 
                                          command=lambda: self.select_color(self.light_color_var),
                                          bootstyle="secondary")
        self.light_color_button.grid(row=2, column=1, padx=5, pady=2, sticky=EW)
        self.light_color_preview = tb.Label(self.light_params_frame, background=self.light_color_var.get(), width=4)
        self.light_color_preview.grid(row=2, column=2, padx=5, pady=2)
        
        self.light_params_frame.columnconfigure(1, weight=1)
        
        # 自定义图像选择按钮
        self.custom_image_button = tb.Button(mosaic_frame, text="选择自定义贴图", command=self.select_custom_image, bootstyle=INFO)
        self.custom_image_label = tb.Label(mosaic_frame, textvariable=self.custom_image_path_var, wraplength=180)
        
        self.on_mosaic_type_change()

        # 4. 区域选择
        self.regions_frame = tb.Labelframe(controls_frame, text="选择打码区域", padding=5)
        self.regions_frame.pack(fill=BOTH, pady=10, expand=YES)
        self.regions_canvas = tk.Canvas(self.regions_frame, borderwidth=0, background="#ffffff")
        self.regions_scrollbar = tb.Scrollbar(self.regions_frame, orient="vertical", command=self.regions_canvas.yview, bootstyle="round")
        self.scrollable_frame = tb.Frame(self.regions_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.regions_canvas.configure(
                scrollregion=self.regions_canvas.bbox("all")
            )
        )
        self.regions_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.regions_canvas.configure(yscrollcommand=self.regions_scrollbar.set)

        self.regions_canvas.pack(side="left", fill="both", expand=True)
        self.regions_scrollbar.pack(side="right", fill="y")
        self.update_region_selection_ui([])

        # 5. 输出文件夹选择
        output_frame = tb.Frame(controls_frame)
        output_frame.pack(fill=X, pady=5, side=BOTTOM)
        tb.Button(output_frame, text="选择输出文件夹", command=self.select_output_folder, bootstyle=SECONDARY).pack(side=LEFT, padx=5)
        tb.Entry(output_frame, textvariable=self.output_folder, state="readonly").pack(side=LEFT, fill=X, expand=YES)
        
        # 6. 处理按钮
        process_button_frame = tb.Frame(controls_frame)
        process_button_frame.pack(fill=X, pady=10, side=BOTTOM)
        self.process_single_button = tb.Button(process_button_frame, text="处理当前图片", command=self.process_current_image, bootstyle=SUCCESS)
        self.process_single_button.pack(side=LEFT, padx=5, expand=True, fill=X)
        self.process_single_button.config(state=DISABLED)

        self.batch_process_button = tb.Button(process_button_frame, text="批量处理", command=self.start_batch_process, bootstyle=WARNING)
        self.batch_process_button.pack(side=LEFT, padx=5, expand=True, fill=X)
        self.batch_process_button.config(state=DISABLED)

        # 右侧：图像预览
        preview_frame = tb.Labelframe(main_frame, text="图像预览", padding=10)
        preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # 创建固定大小的预览容器
        self.original_frame = tb.Frame(preview_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.original_frame.pack(side=LEFT, padx=10, pady=5)
        self.original_frame.pack_propagate(False)  # 防止框架大小被内容影响

        self.processed_frame = tb.Frame(preview_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.processed_frame.pack(side=LEFT, padx=10, pady=5)
        self.processed_frame.pack_propagate(False)  # 防止框架大小被内容影响

        # 在固定框架内添加图像标签
        self.original_image_label = tb.Label(self.original_frame, text="原图区域", relief="solid", anchor=CENTER)
        self.original_image_label.pack(fill=BOTH, expand=YES)
        self.processed_image_label = tb.Label(self.processed_frame, text="效果预览区域", relief="solid", anchor=CENTER)
        self.processed_image_label.pack(fill=BOTH, expand=YES)

        # 底部：状态栏和进度条
        status_bar_frame = tb.Frame(self, padding=(5,2))
        status_bar_frame.pack(side=BOTTOM, fill=X)
        self.status_label = tb.Label(status_bar_frame, text="准备就绪", anchor=W)
        self.status_label.pack(side=LEFT, padx=5)
        self.progress_bar = tb.Progressbar(status_bar_frame, mode='determinate', length=200, bootstyle=SUCCESS)
        self.progress_bar.pack(side=RIGHT, padx=5)
        
        self.after_idle(self.update_color_preview)

    def update_conf_label(self, value):
        self.conf_value_label.config(text=f"{float(value):.2f}")
        self.on_param_change()
        
    def update_iou_label(self, value):
        self.iou_value_label.config(text=f"{float(value):.2f}")
        self.on_param_change()
    
    def update_scale_label(self, value):
        self.scale_value_label.config(text=f"{float(value):.2f}")
        self.on_param_change()
    
    def update_alpha_label(self, value):
        self.alpha_value_label.config(text=f"{float(value):.2f}")
        self.on_param_change()
    
    def update_blur_label(self, value):
        self.blur_value_label.config(text=f"{int(float(value))}")
        self.on_param_change()
        
    def update_line_thickness_label(self, value):
        self.line_thickness_value_label.config(text=f"{int(float(value))}")
        self.on_param_change()
        
    def update_line_spacing_label(self, value):
        self.line_spacing_value_label.config(text=f"{int(float(value))}")
        self.on_param_change()
        
    def update_light_intensity_label(self, value):
        self.light_intensity_value_label.config(text=f"{float(value):.2f}")
        self.on_param_change()
        
    def update_light_feather_label(self, value):
        self.light_feather_value_label.config(text=f"{int(float(value))}")
        self.on_param_change()

    def select_color(self, color_var):
        """选择颜色并更新预览"""
        color = colorchooser.askcolor(color_var.get())
        if color[1]:
            color_var.set(color[1])
            self.update_color_preview()
            self.on_param_change()
    
    def update_color_preview(self):
        """更新颜色预览标签"""
        self.mist_color_preview.config(background=self.mist_color_var.get())
        self.light_color_preview.config(background=self.light_color_var.get())

    def on_param_change(self, *args):
        """当任何参数变化时更新预览"""
        if current_image_path and original_pil_image:
            self.update_preview()

    def on_mosaic_type_change(self, *args):
        mosaic_type = self.mosaic_type_var.get()
        
        # 隐藏所有特定参数框架
        self.blur_params_frame.pack_forget()
        self.line_params_frame.pack_forget()
        self.mist_params_frame.pack_forget()
        self.light_params_frame.pack_forget()
        self.custom_image_button.pack_forget()
        self.custom_image_label.pack_forget()
        
        # 始终显示通用参数
        self.common_params_frame.pack(fill=X, pady=(5,0))
        
        # 根据马赛克类型显示特定参数
        if mosaic_type == "常规模糊":
            self.blur_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "黑色线条":
            self.line_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "白色雾气":
            self.mist_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "光效马赛克":
            self.light_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "自定义图像":
            self.custom_image_button.pack(fill=X, pady=(5,0))
            self.custom_image_label.pack(fill=X, pady=(0,5))
        
        if current_image_path and original_pil_image:
            self.update_preview()

    def on_line_direction_change(self, *args):
        if current_image_path and original_pil_image:
            self.update_preview()

    def select_input_path(self):
        dialog = tb.dialogs.MessageDialog(
            parent=self,
            title="选择输入类型",
            message="请选择要处理单个图片文件还是整个文件夹？",
            buttons=["图片文件:primary", "文件夹:secondary", "取消:light"],
        )
        dialog.show()
        
        result = dialog.result
        path = ""
        if result == "图片文件":
            path = filedialog.askopenfilename(
                title="选择图片文件",
                filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*"))
            )
        elif result == "文件夹":
            path = filedialog.askdirectory(title="选择包含图片的文件夹")

        if path:
            self.input_path.set(path)
            self.status_label.config(text=f"已选择: {Path(path).name}")
            if Path(path).is_file():
                global current_image_path
                current_image_path = path
                self.load_and_display_original_image(path)
                self.process_single_button.config(state=NORMAL)
                self.batch_process_button.config(state=DISABLED)
                self.update_available_regions()
            elif Path(path).is_dir():
                self.clear_previews()
                current_image_path = None
                self.process_single_button.config(state=DISABLED)
                self.batch_process_button.config(state=NORMAL)
                
                # 为批量处理设置预定义标签
                available_labels = get_available_labels()
                self.update_region_selection_ui(available_labels)

    def load_and_display_original_image(self, image_path):
        global original_pil_image
        try:
            original_pil_image = Image.open(image_path)
            self.display_image_on_label(original_pil_image, self.original_image_label)
            self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
            self.status_label.config(text=f"已加载: {Path(image_path).name}")
            
            # 重置检测缓存，因为图像已更改
            self.cached_detection_results = None
            self.last_detection_conf = None
            self.last_detection_iou = None
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载图片: {e}")
            original_pil_image = None
            self.clear_previews()

    def update_available_regions(self):
        if current_image_path and Path(current_image_path).is_file():
            self.status_label.config(text="正在分析图像中的目标...")
            self.update()
            
            def _analyze():
                conf_threshold = self.conf_threshold_var.get()
                iou_threshold = self.iou_threshold_var.get()
                
                # 使用更新版本的get_image_object_names函数，可以同时获取检测结果
                names, results, err = get_image_object_names(current_image_path, conf_threshold, iou_threshold)
                
                # 保存检测参数和结果
                self.last_detection_conf = conf_threshold
                self.last_detection_iou = iou_threshold
                self.cached_detection_results = results
                
                if err:
                    self.status_label.config(text=err)
                    # 如果检测失败，使用预定义标签
                    available_labels = get_available_labels()
                    self.update_region_selection_ui(available_labels)
                else:
                    self.status_label.config(text="目标分析完成。请选择打码区域。")
                    if not names:
                        # 如果没有检测到，使用预定义标签
                        names = get_available_labels()
                    self.update_region_selection_ui(names)
            
            threading.Thread(target=_analyze, daemon=True).start()

    def update_region_selection_ui(self, region_names):
        self.available_regions = region_names
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.selected_regions_vars.clear()

        if not region_names:
            tb.Label(self.scrollable_frame, text="当前图片未识别到可选区域").pack(padx=5, pady=5)
            return

        # 添加全选/取消全选按钮
        select_all_var = tk.BooleanVar(value=True)
        def toggle_all():
            for var in self.selected_regions_vars.values():
                var.set(select_all_var.get())
            self.on_region_selection_change()
            
        select_all_cb = tb.Checkbutton(self.scrollable_frame, text="全选/取消全选", 
                                      variable=select_all_var, command=toggle_all,
                                      bootstyle="secondary")
        select_all_cb.pack(anchor=W, padx=5, pady=(0, 5))

        for name in region_names:
            var = tk.BooleanVar(value=True)
            self.selected_regions_vars[name] = var
            cb = tb.Checkbutton(self.scrollable_frame, text=name, variable=var, bootstyle="primary", command=self.on_region_selection_change)
            cb.pack(anchor=W, padx=5)
        self.on_region_selection_change()

    def on_region_selection_change(self):
        if original_pil_image and current_image_path:
            self.update_preview()

    def select_custom_image(self):
        global custom_mosaic_image_path, custom_mosaic_pil_image
        path = filedialog.askopenfilename(
            title="选择自定义贴图 (推荐PNG透明底)",
            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*"))
        )
        if path:
            custom_mosaic_image_path = path
            self.custom_image_path_var.set(Path(path).name)
            try:
                custom_mosaic_pil_image = Image.open(path)
            except Exception as e:
                messagebox.showerror("贴图加载失败", f"无法加载自定义贴图: {e}")
                custom_mosaic_image_path = None
                custom_mosaic_pil_image = None
                self.custom_image_path_var.set("使用默认图案")
            self.update_preview()
        else:
            custom_mosaic_image_path = None 
            custom_mosaic_pil_image = None
            self.custom_image_path_var.set("使用默认图案")
            self.update_preview()

    def select_output_folder(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_folder.set(path)

    def get_selected_regions(self):
        return [name for name, var in self.selected_regions_vars.items() if var.get()]

    def get_current_parameters(self):
        """获取当前设置的所有参数"""
        mist_color_hex = self.mist_color_var.get()
        light_color_hex = self.light_color_var.get()
        
        # 将十六进制颜色转换为RGB
        mist_color = tuple(int(mist_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        light_color = tuple(int(light_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # 模糊内核大小必须是奇数
        blur_kernel_size = self.blur_kernel_size_var.get()
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        
        return {
            "mosaic_type": self.mosaic_type_var.get(),
            "line_direction": self.line_direction_var.get(),
            "conf_threshold": self.conf_threshold_var.get(),
            "iou_threshold": self.iou_threshold_var.get(),
            "scale": self.scale_var.get(),
            "alpha": self.alpha_var.get(),
            "blur_kernel_size": (blur_kernel_size, blur_kernel_size),
            "line_thickness": self.line_thickness_var.get(),
            "line_spacing": self.line_spacing_var.get(),
            "mist_color": mist_color,
            "light_intensity": self.light_intensity_var.get(),
            "light_feather": self.light_feather_var.get(),
            "light_color": light_color
        }

    def process_current_image(self):
        if not current_image_path or not original_pil_image:
            messagebox.showwarning("操作无效", "请先选择一张图片。")
            return
        
        self.status_label.config(text="正在处理当前图片...")
        self.progress_bar.start()
        self.update()

        params = self.get_current_parameters()
        selected_regions = self.get_selected_regions()
        
        def _process():
            global processed_pil_image
            custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" else None
            
            # 检查是否需要重新检测（阈值变化）
            current_conf = params["conf_threshold"]
            current_iou = params["iou_threshold"]
            cached_results = self.cached_detection_results
            
            if self.last_detection_conf != current_conf or self.last_detection_iou != current_iou:
                # 阈值变化，需要重新检测
                cached_results = None
                self.status_label.config(text="检测参数已变更，正在重新检测...")
            
            # 处理图像并传入缓存结果
            _, proc_img, error = process_single_image(
                current_image_path, 
                params["mosaic_type"], 
                selected_regions, 
                custom_path, 
                params["line_direction"],
                current_conf,
                current_iou,
                params["scale"],
                params["alpha"],
                params["blur_kernel_size"],
                params["line_thickness"],
                params["line_spacing"],
                params["mist_color"],
                params["light_intensity"],
                params["light_feather"],
                params["light_color"],
                cached_detection_results=cached_results
            )
            
            # 如果没有使用缓存或缓存失效，更新缓存
            if cached_results is None:
                self.cached_detection_results = detect_censors(current_image_path, detection_model, 
                                                             current_conf, current_iou)
                self.last_detection_conf = current_conf
                self.last_detection_iou = current_iou
            
            self.progress_bar.stop()
            if error:
                messagebox.showerror("处理失败", error)
                self.status_label.config(text=f"处理失败: {error}")
                processed_pil_image = None
                self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
            else:
                processed_pil_image = proc_img
                self.display_image_on_label(processed_pil_image, self.processed_image_label)
                self.status_label.config(text="图片处理完成。预览已更新。")
                self.prompt_save_processed_image()

        threading.Thread(target=_process, daemon=True).start()

    def prompt_save_processed_image(self):
        if processed_pil_image:
            if messagebox.askyesno("保存图片", "处理完成，是否保存打码后的图片？"):
                self.save_processed_image()

    def save_processed_image(self):
        if not processed_pil_image:
            messagebox.showwarning("无法保存", "没有已处理的图片可供保存。")
            return
        if not current_image_path:
            messagebox.showwarning("无法保存", "原始图片路径未知。")
            return

        original_file_path = Path(current_image_path)
        output_dir = Path(self.output_folder.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        default_name = f"{original_file_path.stem}_processed{original_file_path.suffix}"
        save_path = filedialog.asksaveasfilename(
            initialdir=str(output_dir),
            initialfile=default_name,
            defaultextension=original_file_path.suffix,
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
        )
        if save_path:
            try:
                # 确保保存为 RGB 格式，解决颜色空间问题
                if processed_pil_image.mode == 'RGBA':
                    # 如果有透明通道，需要转换为RGB同时保留图像质量
                    rgb_img = Image.new('RGB', processed_pil_image.size, (255, 255, 255))
                    rgb_img.paste(processed_pil_image, mask=processed_pil_image.split()[3])
                    rgb_img.save(save_path)
                else:
                    # 确保是RGB模式
                    processed_pil_image.convert('RGB').save(save_path)
                    
                messagebox.showinfo("保存成功", f"图片已保存到: {save_path}")
                self.status_label.config(text=f"已保存: {Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片时出错: {e}")

    def update_preview(self):
        if not current_image_path or not original_pil_image:
            return

        selected_regions = self.get_selected_regions()
        params = self.get_current_parameters()
        
        self.status_label.config(text="正在更新预览...")
        self.update()

        custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" else None
        
        def _update_preview_thread():
            global processed_pil_image
            
            # 检查是否需要重新检测（阈值变化）
            current_conf = params["conf_threshold"]
            current_iou = params["iou_threshold"]
            cached_results = self.cached_detection_results
            
            # 如果阈值变化了，需要重新检测
            if self.last_detection_conf != current_conf or self.last_detection_iou != current_iou:
                cached_results = None
                self.status_label.config(text="检测参数已变更，正在重新检测...")
            
            # 处理图像并传入缓存结果
            _, temp_processed_pil, error = process_single_image(
                current_image_path, 
                params["mosaic_type"], 
                selected_regions, 
                custom_path, 
                params["line_direction"],
                current_conf,
                current_iou,
                params["scale"],
                params["alpha"],
                params["blur_kernel_size"],
                params["line_thickness"],
                params["line_spacing"],
                params["mist_color"],
                params["light_intensity"],
                params["light_feather"],
                params["light_color"],
                cached_detection_results=cached_results
            )
            
            # 如果没有使用缓存或缓存失效，更新缓存
            if cached_results is None:
                self.cached_detection_results = detect_censors(current_image_path, detection_model, 
                                                              current_conf, current_iou)
                self.last_detection_conf = current_conf
                self.last_detection_iou = current_iou
            
            if error:
                self.status_label.config(text=f"预览更新错误: {error[:100]}...")
                self.display_image_on_label(original_pil_image, self.processed_image_label, "预览生成错误")
                processed_pil_image = None
            else:
                processed_pil_image = temp_processed_pil
                self.display_image_on_label(processed_pil_image, self.processed_image_label)
                self.status_label.config(text="预览已更新。")
        
        threading.Thread(target=_update_preview_thread, daemon=True).start()

    def start_batch_process(self):
        input_val = self.input_path.get()
        if not input_val:
            messagebox.showwarning("操作无效", "请先选择一个文件夹或图片文件进行批量处理。")
            return
        
        if not Path(input_val).exists():
             messagebox.showerror("错误", f"输入路径不存在: {input_val}")
             return

        output_val = self.output_folder.get()
        if not output_val:
            messagebox.showwarning("操作无效", "请选择输出文件夹。")
            return

        params = self.get_current_parameters()
        selected_regions = self.get_selected_regions()
        custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" else None

        self.batch_process_button.config(state=DISABLED)
        self.process_single_button.config(state=DISABLED)
        self.progress_bar['value'] = 0

        def _batch_thread():
            def progress_cb(current, total):
                self.progress_bar['value'] = (current / total) * 100
                self.update_idletasks()

            def status_cb(message):
                self.status_label.config(text=message)
                self.update_idletasks()
            
            def image_preview_cb(original, processed):
                # 在UI线程中更新图像预览
                self.after(100, lambda: self.display_image_on_label(original, self.original_image_label, "原图预览"))
                if processed:
                    self.after(100, lambda: self.display_image_on_label(processed, self.processed_image_label))
                else:
                    self.after(100, lambda: self.display_image_on_label(None, self.processed_image_label, "效果预览区域"))

            batch_process_images(
                input_val, output_val, 
                params["mosaic_type"], 
                selected_regions, 
                custom_path, 
                params["line_direction"],
                params["conf_threshold"],
                params["iou_threshold"],
                params["scale"],
                params["alpha"],
                params["blur_kernel_size"],
                params["line_thickness"],
                params["line_spacing"],
                params["mist_color"],
                params["light_intensity"],
                params["light_feather"],
                params["light_color"],
                progress_callback=progress_cb,
                status_callback=status_cb,
                image_preview_callback=image_preview_cb
            )
            messagebox.showinfo("批量处理完成", f"所有文件已处理完毕。\n输出到: {output_val}")
            self.batch_process_button.config(state=NORMAL if Path(input_val).is_dir() else DISABLED)
            self.process_single_button.config(state=NORMAL if Path(input_val).is_file() else DISABLED)
            self.progress_bar['value'] = 0
            self.status_label.config(text="准备就绪")

        threading.Thread(target=_batch_thread, daemon=True).start()

    def display_image_on_label(self, pil_image, label_widget, placeholder_text="图像区域"):
        try:
            if pil_image:
                # 获取固定尺寸预览区域的大小
                frame_width = label_widget.master.winfo_width()
                frame_height = label_widget.master.winfo_height()

                if frame_width <= 1 or frame_height <= 1:
                    # 如果框架尚未完全加载，延迟处理
                    label_widget.after_idle(lambda: self.display_image_on_label(pil_image, label_widget, placeholder_text))
                    return

                img_copy = pil_image.copy()
                
                original_width, original_height = img_copy.size
                ratio_w = frame_width / original_width
                ratio_h = frame_height / original_height
                scale_ratio = min(ratio_w, ratio_h)

                new_width = int(original_width * scale_ratio)
                new_height = int(original_height * scale_ratio)

                if new_width > 0 and new_height > 0:
                    img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img_copy)
                    # 先保存引用再设置图像，防止垃圾回收
                    label_widget.image = photo
                    label_widget.config(image=photo, text="")
                else:
                    label_widget.image = None
                    label_widget.config(image="", text=placeholder_text + "\n(图像过小或尺寸错误)")
            else:
                # 先清除旧图像引用，再更新配置
                label_widget.image = None
                label_widget.config(image="", text=placeholder_text)
        except Exception as e:
            print(f"显示图像错误: {e}")
            # 错误恢复处理
            label_widget.image = None
            label_widget.config(image="", text=f"{placeholder_text}\n(显示错误: {str(e)[:50]})")

    def clear_previews(self):
        global original_pil_image, processed_pil_image, current_image_path
        original_pil_image = None
        processed_pil_image = None
        current_image_path = None
        self.display_image_on_label(None, self.original_image_label, "原图区域")
        self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
        self.process_single_button.config(state=DISABLED)

if __name__ == "__main__":
    Path("assets").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if not Path(DEFAULT_HEAD_PATH).exists():
        print(f"警告: 默认自定义贴图 {DEFAULT_HEAD_PATH} 未找到。")

    app = ImageMosaicApp()
    app.mainloop()