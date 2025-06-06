# main_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageGrab
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import os
from pathlib import Path
import threading
import io
import platform
import re # For URL detection

try:
    import tkinterdnd2
    from tkinterdnd2 import DND_FILES, DND_TEXT # DND_TEXT might be useful for URLs
    TKINTERDND2_AVAILABLE = True
    print("tkinterdnd2 Python module imported successfully.")
except ImportError:
    TKINTERDND2_AVAILABLE = False
    print("警告: TkinterDnD2 库未找到。拖放功能将不可用。")
    print("请通过 'pip install tkinterdnd2' 安装。")

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False
    print("警告: 'pyperclip' 库未找到。复制文件路径到剪贴板功能将不可用。")
    print("请通过 'pip install pyperclip' 安装。")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: 'requests' 库未找到。从URL拖放图片功能将受限。")
    print("请通过 'pip install requests' 安装。")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("警告: 'beautifulsoup4' 库未找到。解析浏览器拖放的复杂图片功能将受限。")
    print("请通过 'pip install beautifulsoup4' 安装。")


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

PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 1250

classification_model, detection_model = load_models()
if not detection_model:
    print("警告：检测模型加载失败，应用功能将受限。")

base_window_class = tkinterdnd2.Tk if TKINTERDND2_AVAILABLE else tb.Window

class ImageMosaicApp(base_window_class):
    def __init__(self):
        super().__init__()

        if TKINTERDND2_AVAILABLE and not isinstance(self, tb.Window):
            try:
                self.style = tb.Style(theme="superhero")
            except Exception as e:
                print(f"应用 ttkbootstrap 主题失败: {e}. 将使用默认Tk主题。")
                self.style = tb.Style() # Fallback to default style
        elif isinstance(self, tb.Window):
             self.style = self.style # ttkbootstrap.Window already has a style
        else:
            self.style = tb.Style()


        self.title("图像打码工具")
        self.geometry("1250x900")

        self.dnd_initialized = TKINTERDND2_AVAILABLE
        if not TKINTERDND2_AVAILABLE:
             messagebox.showwarning("缺少库", "TkinterDnD2 未安装，拖放功能不可用。\n请运行: pip install tkinterdnd2", parent=self)


        self.cached_detection_results = None
        self.last_detection_conf = None
        self.last_detection_iou = None
        self.input_path = tk.StringVar()
        self.output_folder = tk.StringVar(value=str(Path.home() / "图像打码输出"))
        self.mosaic_type_var = tk.StringVar(value="常规模糊")
        self.available_regions = []
        self.selected_regions_vars = {}
        self.custom_image_path_var = tk.StringVar(value="使用默认图案")
        self.line_direction_var = tk.StringVar(value="horizontal")
        self.conf_threshold_var = tk.DoubleVar(value=0.25)
        self.iou_threshold_var = tk.DoubleVar(value=0.7)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.blur_kernel_size_var = tk.IntVar(value=31)
        self.line_thickness_var = tk.IntVar(value=5)
        self.line_spacing_var = tk.IntVar(value=10)
        self.mist_color_var = tk.StringVar(value="#FFFFFF")
        self.light_intensity_var = tk.DoubleVar(value=0.8)
        self.light_feather_var = tk.IntVar(value=30)
        self.light_color_var = tk.StringVar(value="#FFFFFF")
        self.in_mini_mode = False
        self.mini_mode_frame = None
        self._is_window_pinned = tk.BooleanVar(value=False)
        self._mini_mosaic_type_var = tk.StringVar(value="常规模糊")
        self.drop_target_label_mini = None

        self.programmatic_resize_in_progress = False

        self.full_mode_main_frame = tb.Frame(self, padding=10)
        self._setup_full_ui(self.full_mode_main_frame)
        self.full_mode_main_frame.pack(fill=BOTH, expand=YES)

        try:
            self.is_currently_maximized_internal = self.attributes('-zoomed')
        except tk.TclError:
            print("注意: 无法获取 '-zoomed' 属性, 自动模式切换可能不准确。将使用几何判断。")
            self.is_currently_maximized_internal = False
            
        self.bind("<Configure>", self.on_window_resize_check_state, add="+")

        menubar = tk.Menu(self)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="切换到迷你模式（适用于快捷从网页拖取图像，打码并发送到群聊）", command=self.switch_to_mini_mode)
        mode_menu.add_command(label="切换到完整模式（适用于对生成的大量图像进行批量处理）", command=self.switch_to_full_mode)
        menubar.add_cascade(label="点击此处切换模式", menu=mode_menu)
        self.config(menu=menubar)

        if not hasattr(self, 'style'):
            self.style = tb.Style()

        self.after_idle(self.update_color_preview)
        self.after_idle(self._initial_check_maximized_state)

    def _get_current_actual_maximized_state(self):
        """Helper to get current maximized state, handling -zoomed issues."""
        if self.in_mini_mode and self.winfo_width() < 500 and self.winfo_height() < 400:
            return False

        try:
            maximized = self.attributes('-zoomed')
            if maximized and self.winfo_width() < 500 and self.winfo_height() < 400:
                maximized = False
            return maximized
        except tk.TclError:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            return (
                self.winfo_width() >= screen_w - 30 and
                self.winfo_height() >= screen_h - 70
            )

    def _initial_check_maximized_state(self):
        """Checks the initial maximized state of the window more reliably."""
        self.is_currently_maximized_internal = self._get_current_actual_maximized_state()


    def _show_timed_message(self, title, message, duration_ms=1500, success=True):
        parent_widget = self.mini_mode_frame if self.in_mini_mode and self.mini_mode_frame and self.mini_mode_frame.winfo_exists() else self

        dialog = tk.Toplevel(parent_widget)
        dialog.title(title)
        dialog.transient(parent_widget)
        dialog.attributes("-topmost", True)

        parent_widget.update_idletasks()
        anchor_x = parent_widget.winfo_rootx()
        anchor_y = parent_widget.winfo_rooty()
        anchor_w = parent_widget.winfo_width()
        anchor_h = parent_widget.winfo_height()

        dialog_w = 300
        dialog_h = 80

        dialog_x = anchor_x + (anchor_w - dialog_w) // 2
        dialog_y = anchor_y + (anchor_h - dialog_h) // 2

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        if dialog_x + dialog_w > screen_w: dialog_x = screen_w - dialog_w
        if dialog_y + dialog_h > screen_h: dialog_y = screen_h - dialog_h
        if dialog_x < 0: dialog_x = 0
        if dialog_y < 0: dialog_y = 0

        dialog.geometry(f"{dialog_w}x{dialog_h}+{dialog_x}+{dialog_y}")
        dialog.resizable(False, False)
        dialog.overrideredirect(True)

        try:
            frame_style = SUCCESS if success else WARNING
            outer_frame = tb.Frame(dialog, bootstyle=frame_style, padding=1)
            outer_frame.pack(expand=True, fill=tk.BOTH)
            text_color_style = f"{frame_style}-fg" if hasattr(self.style.colors, f"get_{frame_style}_fg") else "inverse"
            msg_label = tb.Label(outer_frame, text=message, font=("Helvetica", 10),
                                 bootstyle=text_color_style,
                                 anchor="center", padding=(10,5), wraplength=dialog_w - 20)
        except:
            fg_color = "green" if success else "red"
            outer_frame = tk.Frame(dialog, bg="lightgrey" if success else "lightpink", bd=1, relief="solid")
            outer_frame.pack(expand=True, fill=tk.BOTH)
            msg_label = tk.Label(outer_frame, text=message, font=("Helvetica", 10),
                                 fg=fg_color, padx=10, pady=10, wraplength=dialog_w - 20)

        msg_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        dialog.after(duration_ms, dialog.destroy)

    def on_window_resize_check_state(self, event=None):
        current_actual_maximized = self._get_current_actual_maximized_state()

        if self.programmatic_resize_in_progress:
            self.is_currently_maximized_internal = current_actual_maximized
            return

        previous_internal_maximized_state = self.is_currently_maximized_internal
        mode_switched_by_handler = False

        if self.in_mini_mode:
            if current_actual_maximized and not previous_internal_maximized_state:
                self.switch_to_full_mode()
                mode_switched_by_handler = True
            elif not current_actual_maximized and previous_internal_maximized_state:
                 self.is_currently_maximized_internal = False
        else: 
            if not current_actual_maximized and previous_internal_maximized_state:
                self.switch_to_mini_mode()
                mode_switched_by_handler = True
            elif current_actual_maximized and not previous_internal_maximized_state:
                self.is_currently_maximized_internal = True

        if not mode_switched_by_handler:
            self.is_currently_maximized_internal = current_actual_maximized


    def _setup_full_ui(self, main_frame):
        controls_frame = tb.Labelframe(main_frame, text="控制面板", padding=10)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        file_frame = tb.Frame(controls_frame)
        file_frame.pack(fill=X, pady=5)
        tb.Button(file_frame, text="选择图片/文件夹", command=self.select_input_path, bootstyle=PRIMARY).pack(side=LEFT, padx=5)
        tb.Entry(file_frame, textvariable=self.input_path, state="readonly").pack(side=LEFT, fill=X, expand=YES)
        model_params_frame = tb.Labelframe(controls_frame, text="模型参数", padding=5)
        model_params_frame.pack(fill=X, pady=5)
        tb.Label(model_params_frame, text="置信度阈值:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.conf_value_label = tb.Label(model_params_frame, text=f"{self.conf_threshold_var.get():.2f}", width=4)
        self.conf_value_label.grid(row=0, column=2, padx=5, pady=2)
        conf_scale = tb.Scale(model_params_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var,
                            orient=HORIZONTAL, bootstyle="success", command=self.update_conf_label)
        conf_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        tb.Label(model_params_frame, text="IOU阈值:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.iou_value_label = tb.Label(model_params_frame, text=f"{self.iou_threshold_var.get():.2f}", width=4)
        self.iou_value_label.grid(row=1, column=2, padx=5, pady=2)
        iou_scale = tb.Scale(model_params_frame, from_=0.1, to=1.0, variable=self.iou_threshold_var,
                           orient=HORIZONTAL, bootstyle="success", command=self.update_iou_label)
        iou_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        model_params_frame.columnconfigure(1, weight=1)
        mosaic_frame = tb.Labelframe(controls_frame, text="选择打码方式", padding=5)
        mosaic_frame.pack(fill=X, pady=10)
        mosaic_options = ["常规模糊", "黑色线条", "白色雾气", "光效马赛克", "自定义图像"]
        for option in mosaic_options:
            rb = tb.Radiobutton(mosaic_frame, text=option, variable=self.mosaic_type_var, value=option, command=self.on_mosaic_type_change)
            rb.pack(anchor=W)
        self.common_params_frame = tb.Labelframe(mosaic_frame, text="通用参数", padding=5)
        tb.Label(self.common_params_frame, text="区域缩放:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.scale_value_label = tb.Label(self.common_params_frame, text=f"{self.scale_var.get():.2f}", width=4)
        self.scale_value_label.grid(row=0, column=2, padx=5, pady=2)
        scale_scale = tb.Scale(self.common_params_frame, from_=0.5, to=2.0, variable=self.scale_var,
                     orient=HORIZONTAL, bootstyle="info", command=self.update_scale_label)
        scale_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        tb.Label(self.common_params_frame, text="透明度:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.alpha_value_label = tb.Label(self.common_params_frame, text=f"{self.alpha_var.get():.2f}", width=4)
        self.alpha_value_label.grid(row=1, column=2, padx=5, pady=2)
        alpha_scale = tb.Scale(self.common_params_frame, from_=0.0, to=1.0, variable=self.alpha_var,
                     orient=HORIZONTAL, bootstyle="info", command=self.update_alpha_label)
        alpha_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        self.common_params_frame.columnconfigure(1, weight=1)
        self.blur_params_frame = tb.Labelframe(mosaic_frame, text="模糊参数", padding=5)
        tb.Label(self.blur_params_frame, text="模糊大小:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.blur_value_label = tb.Label(self.blur_params_frame, text=f"{self.blur_kernel_size_var.get()}", width=4)
        self.blur_value_label.grid(row=0, column=2, padx=5, pady=2)
        blur_kernel_scale = tb.Scale(self.blur_params_frame, from_=3, to=101, variable=self.blur_kernel_size_var,
                           orient=HORIZONTAL, bootstyle="warning", command=self.update_blur_label)
        blur_kernel_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        self.blur_params_frame.columnconfigure(1, weight=1)
        self.line_params_frame = tb.Labelframe(mosaic_frame, text="线条参数", padding=5)
        direction_frame = tb.Frame(self.line_params_frame)
        direction_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky=EW)
        tb.Label(direction_frame, text="方向:").pack(side=LEFT)
        line_directions = [("水平", "horizontal"), ("垂直", "vertical"), ("斜线", "diagonal")]
        for text, value in line_directions:
            rb = tb.Radiobutton(direction_frame, text=text, variable=self.line_direction_var, value=value,
                              command=self.on_line_direction_change)
            rb.pack(side=LEFT, padx=5)
        tb.Label(self.line_params_frame, text="线条粗细:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.line_thickness_value_label = tb.Label(self.line_params_frame, text=f"{self.line_thickness_var.get()}", width=4)
        self.line_thickness_value_label.grid(row=1, column=2, padx=5, pady=2)
        line_thickness_scale = tb.Scale(self.line_params_frame, from_=1, to=20, variable=self.line_thickness_var,
                              orient=HORIZONTAL, bootstyle="warning", command=self.update_line_thickness_label)
        line_thickness_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        tb.Label(self.line_params_frame, text="线条间距:").grid(row=2, column=0, padx=5, pady=2, sticky=W)
        self.line_spacing_value_label = tb.Label(self.line_params_frame, text=f"{self.line_spacing_var.get()}", width=4)
        self.line_spacing_value_label.grid(row=2, column=2, padx=5, pady=2)
        line_spacing_scale = tb.Scale(self.line_params_frame, from_=5, to=50, variable=self.line_spacing_var,
                            orient=HORIZONTAL, bootstyle="warning", command=self.update_line_spacing_label)
        line_spacing_scale.grid(row=2, column=1, padx=5, pady=2, sticky=EW)
        self.line_params_frame.columnconfigure(1, weight=1)
        self.mist_params_frame = tb.Labelframe(mosaic_frame, text="雾气参数", padding=5)
        tb.Label(self.mist_params_frame, text="雾气颜色:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.mist_color_button = tb.Button(self.mist_params_frame, text="选择颜色",
                                         command=lambda: self.select_color(self.mist_color_var, self.mist_color_preview),
                                         bootstyle="secondary")
        self.mist_color_button.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        self.mist_color_preview = tb.Label(self.mist_params_frame, background=self.mist_color_var.get(), width=4, relief="solid", borderwidth=1)
        self.mist_color_preview.grid(row=0, column=2, padx=5, pady=2)
        self.mist_params_frame.columnconfigure(1, weight=1)
        self.light_params_frame = tb.Labelframe(mosaic_frame, text="光效参数", padding=5)
        tb.Label(self.light_params_frame, text="光强度:").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        self.light_intensity_value_label = tb.Label(self.light_params_frame, text=f"{self.light_intensity_var.get():.2f}", width=4)
        self.light_intensity_value_label.grid(row=0, column=2, padx=5, pady=2)
        light_intensity_scale = tb.Scale(self.light_params_frame, from_=0.1, to=1.0, variable=self.light_intensity_var,
                               orient=HORIZONTAL, bootstyle="warning", command=self.update_light_intensity_label)
        light_intensity_scale.grid(row=0, column=1, padx=5, pady=2, sticky=EW)
        tb.Label(self.light_params_frame, text="羽化边缘:").grid(row=1, column=0, padx=5, pady=2, sticky=W)
        self.light_feather_value_label = tb.Label(self.light_params_frame, text=f"{self.light_feather_var.get()}", width=4)
        self.light_feather_value_label.grid(row=1, column=2, padx=5, pady=2)
        light_feather_scale = tb.Scale(self.light_params_frame, from_=5, to=100, variable=self.light_feather_var,
                             orient=HORIZONTAL, bootstyle="warning", command=self.update_light_feather_label)
        light_feather_scale.grid(row=1, column=1, padx=5, pady=2, sticky=EW)
        tb.Label(self.light_params_frame, text="光颜色:").grid(row=2, column=0, padx=5, pady=2, sticky=W)
        self.light_color_button = tb.Button(self.light_params_frame, text="选择颜色",
                                          command=lambda: self.select_color(self.light_color_var, self.light_color_preview),
                                          bootstyle="secondary")
        self.light_color_button.grid(row=2, column=1, padx=5, pady=2, sticky=EW)
        self.light_color_preview = tb.Label(self.light_params_frame, background=self.light_color_var.get(), width=4, relief="solid", borderwidth=1)
        self.light_color_preview.grid(row=2, column=2, padx=5, pady=2)
        self.light_params_frame.columnconfigure(1, weight=1)
        self.custom_image_button = tb.Button(mosaic_frame, text="选择自定义贴图", command=self.select_custom_image, bootstyle=INFO)
        self.custom_image_label = tb.Label(mosaic_frame, textvariable=self.custom_image_path_var, wraplength=180)
        self.on_mosaic_type_change()
        self.regions_frame = tb.Labelframe(controls_frame, text="选择打码区域", padding=5)
        self.regions_frame.pack(fill=BOTH, pady=10, expand=YES)
        bg_color = self.style.colors.get('bg') if hasattr(self, 'style') and hasattr(self.style, 'colors') else "#ffffff"
        self.regions_canvas = tk.Canvas(self.regions_frame, borderwidth=0, background=bg_color)
        self.regions_scrollbar = tb.Scrollbar(self.regions_frame, orient="vertical", command=self.regions_canvas.yview, bootstyle="round")
        self.scrollable_frame = tb.Frame(self.regions_canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.regions_canvas.configure(scrollregion=self.regions_canvas.bbox("all")))
        self.regions_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.regions_canvas.configure(yscrollcommand=self.regions_scrollbar.set)
        self.regions_canvas.pack(side="left", fill="both", expand=True)
        self.regions_scrollbar.pack(side="right", fill="y")
        self.update_region_selection_ui([])
        output_frame = tb.Frame(controls_frame)
        output_frame.pack(fill=X, pady=5, side=BOTTOM)
        tb.Button(output_frame, text="选择输出文件夹", command=self.select_output_folder, bootstyle=SECONDARY).pack(side=LEFT, padx=5)
        tb.Entry(output_frame, textvariable=self.output_folder, state="readonly").pack(side=LEFT, fill=X, expand=YES)
        process_button_frame = tb.Frame(controls_frame)
        process_button_frame.pack(fill=X, pady=10, side=BOTTOM)
        self.process_single_button = tb.Button(process_button_frame, text="处理当前图片", command=self.process_current_image, bootstyle=SUCCESS)
        self.process_single_button.pack(side=LEFT, padx=5, expand=True, fill=X)
        self.process_single_button.config(state=DISABLED)
        self.batch_process_button = tb.Button(process_button_frame, text="批量处理", command=self.start_batch_process, bootstyle=WARNING)
        self.batch_process_button.pack(side=LEFT, padx=5, expand=True, fill=X)
        self.batch_process_button.config(state=DISABLED)
        preview_frame = tb.Labelframe(main_frame, text="图像预览", padding=10)
        preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self.original_frame = tb.Frame(preview_frame, width=PREVIEW_WIDTH // 2 - 20, height=PREVIEW_HEIGHT - 50)
        self.original_frame.pack(side=LEFT, padx=10, pady=5, fill=BOTH, expand=True)
        self.original_frame.pack_propagate(False)
        self.processed_frame = tb.Frame(preview_frame, width=PREVIEW_WIDTH // 2 - 20, height=PREVIEW_HEIGHT - 50)
        self.processed_frame.pack(side=LEFT, padx=10, pady=5, fill=BOTH, expand=True)
        self.processed_frame.pack_propagate(False)
        self.original_image_label = tb.Label(self.original_frame, text="原图区域", relief="solid", anchor=CENTER)
        self.original_image_label.pack(fill=BOTH, expand=YES)
        self.processed_image_label = tb.Label(self.processed_frame, text="效果预览区域", relief="solid", anchor=CENTER)
        self.processed_image_label.pack(fill=BOTH, expand=YES)
        status_bar_frame = tb.Frame(self, padding=(5,2))
        status_bar_frame.pack(side=BOTTOM, fill=X)
        self.status_label = tb.Label(status_bar_frame, text="准备就绪", anchor=W)
        self.status_label.pack(side=LEFT, padx=5)
        self.progress_bar = tb.Progressbar(status_bar_frame, mode='determinate', length=200, bootstyle=SUCCESS)
        self.progress_bar.pack(side=RIGHT, padx=5)

    def switch_to_mini_mode(self):
        if self.in_mini_mode:
            return
        self.programmatic_resize_in_progress = True

        self.in_mini_mode = True
        self.title("图像打码 (迷你模式)")
        self.full_mode_main_frame.pack_forget()

        if self.mini_mode_frame is None:
            self.mini_mode_frame = tb.Frame(self, padding=10)
            self._setup_mini_ui(self.mini_mode_frame)
        self.mini_mode_frame.pack(fill=tk.BOTH, expand=True)

        try:
            if self.attributes('-zoomed'):
                self.attributes('-zoomed', False) # Attempt to unzoom
                self.update_idletasks()
        except tk.TclError:
            pass # -zoomed attribute might not be available or settable

        try:
            current_state = self.state()
            if current_state == 'zoomed':
                self.state('normal') # Fallback to state('normal')
                self.update_idletasks()
        except tk.TclError:
            pass # state() or state('normal') might not be available

        self.resizable(False, False)
        self.update_idletasks() # Ensure resizable change is processed

        self.after(50, self._apply_mini_mode_geometry_and_finish_switch)

    def _apply_mini_mode_geometry_and_finish_switch(self):
        self.geometry("350x280")
        self.set_stay_on_top(self._is_window_pinned.get())
        self.is_currently_maximized_internal = False # Mini mode is definitively not maximized
        self.update_idletasks()
        self.programmatic_resize_in_progress = False


    def switch_to_full_mode(self):
        if not self.in_mini_mode:
            return
        self.programmatic_resize_in_progress = True

        self.in_mini_mode = False
        self.title("图像打码工具")
        if self.mini_mode_frame:
            self.mini_mode_frame.pack_forget()

        self.full_mode_main_frame.pack(fill=tk.BOTH, expand=True)

        self.resizable(True, True)
        self.set_stay_on_top(False)
        self.geometry("1250x900")
        
        self.update_idletasks() # Process geometry and resizable changes
        self.is_currently_maximized_internal = self._get_current_actual_maximized_state()
        self.programmatic_resize_in_progress = False


    def _setup_mini_ui(self, parent_frame):
        parent_frame.columnconfigure(0, weight=1)
        mosaic_label = tb.Label(parent_frame, text="打码方式:")
        mosaic_label.grid(row=0, column=0, padx=5, pady=(10,0), sticky="w")
        mosaic_options = ["常规模糊", "黑色线条", "白色雾气", "光效马赛克", "自定义图像"]
        self._mini_mosaic_type_var.set(mosaic_options[0])
        mosaic_combo = tb.Combobox(parent_frame, textvariable=self._mini_mosaic_type_var, values=mosaic_options, state="readonly", bootstyle="primary")
        mosaic_combo.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        pin_checkbox = tb.Checkbutton(parent_frame, text="窗口置顶", variable=self._is_window_pinned,
                                      command=lambda: self.set_stay_on_top(self._is_window_pinned.get()),
                                      bootstyle="primary-round-toggle")
        pin_checkbox.grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.drop_target_label_mini = tb.Label(parent_frame, text="拖放图片到此处", relief="solid",
                                     borderwidth=2, anchor="center", padding=(10, 40), bootstyle="secondary")
        self.drop_target_label_mini.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
        parent_frame.rowconfigure(3, weight=1)

        if self.dnd_initialized:
            try:
                self.drop_target_label_mini.drop_target_register(DND_FILES, DND_TEXT)
                self.drop_target_label_mini.dnd_bind('<<Drop>>', self.handle_drop_mini_mode)
                print("Mini mode drop target registered for DND_FILES and DND_TEXT.")
            except tk.TclError as e_tcl:
                error_message = (f"拖放注册失败 (TclError): {e_tcl}\n"
                                 "TkinterDnD2的Tcl组件未正确加载。\n"
                                 "请确认库已正确安装且Tcl环境无误。")
                self.drop_target_label_mini.config(text=error_message.split('\n')[0])
                print(f"注册拖放目标时发生 TclError: {e_tcl}")
                messagebox.showerror("拖放功能错误", error_message, parent=self)
            except Exception as e_other:
                error_message = f"拖放注册时发生未知错误: {e_other}"
                self.drop_target_label_mini.config(text="拖放注册未知错误")
                print(error_message)
                messagebox.showerror("拖放功能错误", error_message, parent=self)
        else:
            self.drop_target_label_mini.config(text="拖放功能不可用\n(TkinterDnD2 Python模块未加载)")


    def set_stay_on_top(self, pin_value):
        self.attributes("-topmost", pin_value)

    def _is_url(self, text):
        if not isinstance(text, str): return False
        url_pattern = re.compile(
            r'^(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(url_pattern, text) is not None


    def handle_drop_mini_mode(self, event):
        if not self.in_mini_mode or not self.dnd_initialized: return

        self.drop_target_label_mini.config(text="正在解析...", bootstyle="info")
        self.update_idletasks()

        dropped_data = event.data.strip()
        print(f"Dropped data type: {type(dropped_data)}, content: '{dropped_data[:200]}...'")

        try:
            potential_files_raw = self.tk.splitlist(dropped_data)
            potential_files = [f.strip('{}').strip() for f in potential_files_raw if f.strip('{}').strip()]
        except Exception as e_split:
            print(f"Error splitting dropped data: {e_split}")
            potential_files = []

        if potential_files:
            first_item_as_path_str = potential_files[0]
            if not first_item_as_path_str.startswith('data:image'):
                try:
                    first_item_as_path = Path(first_item_as_path_str)
                    if first_item_as_path.exists() and first_item_as_path.is_file():
                        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
                        if first_item_as_path.suffix.lower() in valid_extensions:
                            self.drop_target_label_mini.config(text=f"处理文件: {first_item_as_path.name}", bootstyle="info")
                            self.update_idletasks()
                            threading.Thread(target=self._process_dropped_image_for_mini_mode, args=(str(first_item_as_path),), daemon=True).start()
                            return
                        else:
                            messagebox.showerror("文件错误", f"拖入的文件不是支持的图片格式: {first_item_as_path.name}", parent=self)
                            self.drop_target_label_mini.config(text="非图片文件!", bootstyle="danger")
                            self.after(3000, lambda: self.drop_target_label_mini.config(text="拖放图片到此处", bootstyle="secondary") if self.in_mini_mode and self.drop_target_label_mini.winfo_exists() else None)
                            return
                except Exception as e_path:
                    print(f"Error converting '{first_item_as_path_str}' to Path: {e_path}")

        if dropped_data.startswith('data:image'):
            self.drop_target_label_mini.config(text="处理内嵌图片...", bootstyle="info")
            self.update_idletasks()
            threading.Thread(target=self._process_dropped_base64_image, args=(dropped_data,), daemon=True).start()
            return

        if BS4_AVAILABLE and dropped_data.strip().lower().startswith("<img") and "src=" in dropped_data.lower():
            try:
                soup = BeautifulSoup(dropped_data, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and img_tag.get('src'):
                    src = img_tag.get('src')
                    if src.startswith('data:image'):
                        self.drop_target_label_mini.config(text="处理HTML内嵌图片...", bootstyle="info")
                        self.update_idletasks()
                        threading.Thread(target=self._process_dropped_base64_image, args=(src,), daemon=True).start()
                        return
                    elif self._is_url(src):
                        if REQUESTS_AVAILABLE:
                            self.drop_target_label_mini.config(text=f"下载HTML图片链接...", bootstyle="info")
                            self.update_idletasks()
                            threading.Thread(target=self._process_dropped_url_image, args=(src,), daemon=True).start()
                            return
                        else:
                            messagebox.showwarning("缺少库", "处理网络图片链接需要 'requests' 库。\n请运行: pip install requests", parent=self)
                            self.drop_target_label_mini.config(text="无法下载 (缺requests)", bootstyle="warning")
                            self.after(3000, lambda: self.drop_target_label_mini.config(text="拖放图片到此处", bootstyle="secondary") if self.in_mini_mode and self.drop_target_label_mini.winfo_exists() else None)
                            return
            except Exception as e_html:
                print(f"解析HTML拖放数据时出错: {e_html}")

        if self._is_url(dropped_data):
            if REQUESTS_AVAILABLE:
                self.drop_target_label_mini.config(text=f"下载图片链接...", bootstyle="info")
                self.update_idletasks()
                threading.Thread(target=self._process_dropped_url_image, args=(dropped_data,), daemon=True).start()
                return
            else:
                messagebox.showwarning("缺少库", "处理网络图片链接需要 'requests' 库。\n请运行: pip install requests", parent=self)
                self.drop_target_label_mini.config(text="无法下载 (缺requests)", bootstyle="warning")
                self.after(3000, lambda: self.drop_target_label_mini.config(text="拖放图片到此处", bootstyle="secondary") if self.in_mini_mode and self.drop_target_label_mini.winfo_exists() else None)
                return

        self.drop_target_label_mini.config(text="无法识别拖放内容!", bootstyle="warning")
        messagebox.showwarning("无法识别", "无法识别拖放的内容。\n请拖放图片文件、浏览器中的图片或图片链接。", parent=self)
        self.after(3000, lambda: self.drop_target_label_mini.config(text="拖放图片到此处", bootstyle="secondary") if self.in_mini_mode and self.drop_target_label_mini.winfo_exists() else None)


    def _process_dropped_base64_image(self, base64_src):
        try:
            import base64
            match = re.match(r'data:(image/(?P<format>\w+));base64,(?P<data>.+)', base64_src, re.DOTALL)
            if not match:
                raise ValueError("无效的Base64数据URI格式")

            img_format = match.group('format').lower()
            encoded_data = match.group('data')

            if img_format not in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']:
                print(f"不支持的Base64图片格式: {img_format}, 将尝试保存为png")
                img_format = 'png'

            image_data = base64.b64decode(encoded_data)
            image_pil = Image.open(io.BytesIO(image_data))

            temp_dir = Path(self.output_folder.get()) / "temp_dropped"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_image_path = temp_dir / f"dropped_base64_image.{img_format}"

            if image_pil.mode == 'P' and img_format in ['jpeg', 'jpg']:
                image_pil = image_pil.convert('RGB')
            elif image_pil.mode == 'RGBA' and img_format in ['jpeg', 'jpg', 'bmp']:
                 image_pil = image_pil.convert('RGB')

            image_pil.save(temp_image_path)
            self.after(0, lambda: self.drop_target_label_mini.config(text=f"处理内嵌图片...", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            self._process_dropped_image_for_mini_mode(str(temp_image_path))
        except Exception as e:
            print(f"处理Base64图片错误: {e}")
            messagebox.showerror("处理错误", f"处理内嵌图片时发生错误: {e}", parent=self)
            self.after(0, lambda: self.drop_target_label_mini.config(text="内嵌图片处理失败!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)

    def _process_dropped_url_image(self, image_url):
        if not REQUESTS_AVAILABLE:
            messagebox.showerror("功能缺失", "下载网络图片需要 'requests' 库。", parent=self)
            self.after(0, lambda: self.drop_target_label_mini.config(text="无法下载 (缺requests)", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            return

        try:
            self.after(0, lambda: self.drop_target_label_mini.config(text=f"下载中...", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(image_url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get('content-type')
            if not content_type or not content_type.startswith('image/'):
                messagebox.showerror("下载错误", f"链接内容似乎不是图片: {content_type}", parent=self)
                self.after(0, lambda: self.drop_target_label_mini.config(text="非图片链接!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
                return

            image_data = response.content
            image_pil = Image.open(io.BytesIO(image_data))

            img_format = content_type.split('/')[-1].lower().split(';')[0]
            if img_format == 'jpeg': img_format = 'jpg'
            valid_formats = ['png', 'jpg', 'gif', 'bmp', 'webp']
            if img_format not in valid_formats:
                url_path = Path(image_url)
                extracted_format = url_path.suffix[1:].lower() if url_path.suffix else ''
                if extracted_format in valid_formats:
                    img_format = extracted_format
                else:
                    img_format = 'png'

            temp_dir = Path(self.output_folder.get()) / "temp_downloaded"
            temp_dir.mkdir(parents=True, exist_ok=True)

            base_filename = Path(image_url).name.split('?')[0]
            if not base_filename or len(base_filename) > 100 or '.' not in base_filename :
                base_filename = f"downloaded_image"

            temp_image_filename = f"{Path(base_filename).stem}.{img_format}"
            temp_image_filename = "".join([c for c in temp_image_filename if c.isalnum() or c in ('.', '_')]).rstrip()
            if not Path(temp_image_filename).stem:
                 temp_image_filename = f"downloaded_image.{img_format}"


            temp_image_path = temp_dir / temp_image_filename

            if image_pil.mode == 'P' and img_format in ['jpeg', 'jpg']:
                image_pil = image_pil.convert('RGB')
            elif image_pil.mode == 'RGBA' and img_format in ['jpeg', 'jpg', 'bmp']:
                 image_pil = image_pil.convert('RGB')

            image_pil.save(temp_image_path)

            self.after(0, lambda: self.drop_target_label_mini.config(text=f"处理下载图片...", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            self._process_dropped_image_for_mini_mode(str(temp_image_path))

        except requests.exceptions.RequestException as e:
            print(f"下载图片错误 ({image_url}): {e}")
            messagebox.showerror("下载失败", f"无法下载图片链接: {e}", parent=self)
            self.after(0, lambda: self.drop_target_label_mini.config(text="下载失败!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
        except Exception as e:
            print(f"处理下载的图片时发生错误: {e}")
            messagebox.showerror("处理错误", f"处理下载的图片时出错: {e}", parent=self)
            self.after(0, lambda: self.drop_target_label_mini.config(text="处理下载图片失败!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)


    def _copy_image_to_clipboard(self, pil_image):
        if pil_image is None:
            return False
        try:
            if platform.system() == "Windows":
                output = io.BytesIO()
                pil_image.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]
                output.close()
                import win32clipboard
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                return True
        except ImportError:
            print("Windows剪贴板操作需要 pywin32。请运行: pip install pywin32")
        except Exception as e:
            print(f"复制图片到剪贴板时发生错误: {e}")
        return False

    def _process_dropped_image_for_mini_mode(self, image_path_str):
        image_path = Path(image_path_str)
        mosaic_type = self._mini_mosaic_type_var.get()
        selected_regions = get_available_labels()
        default_conf = self.conf_threshold_var.get()
        default_iou = self.iou_threshold_var.get()
        custom_img_actual_path = DEFAULT_HEAD_PATH
        if mosaic_type == "自定义图像":
            if custom_mosaic_image_path and os.path.exists(custom_mosaic_image_path) :
                 custom_img_actual_path = custom_mosaic_image_path

        try:
            self.after(0, lambda: self.drop_target_label_mini.config(text="正在打码...", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            _, processed_img_pil, error_msg = process_single_image(
                str(image_path), mosaic_type, selected_regions,
                custom_image_path=custom_img_actual_path,
                conf_threshold=default_conf, iou_threshold=default_iou,
            )

            if error_msg:
                messagebox.showerror("处理失败", error_msg, parent=self)
                self.after(0, lambda: self.drop_target_label_mini.config(text="处理失败!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
                return

            if processed_img_pil:
                copied_to_clipboard = self._copy_image_to_clipboard(processed_img_pil)
                if copied_to_clipboard:
                    self._show_timed_message("成功", "已打码并复制到剪贴板!", success=True)
                    self.after(0, lambda: self.drop_target_label_mini.config(text="已复制!", bootstyle="success") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
                else:
                    temp_save_dir = Path(self.output_folder.get()) / "temp_processed"
                    temp_save_dir.mkdir(parents=True, exist_ok=True)
                    save_suffix = image_path.suffix if image_path.suffix else '.png'
                    temp_save_path = temp_save_dir / f"{image_path.stem}_processed{save_suffix}"
                    save_image = processed_img_pil.copy()
                    if save_image.mode == 'P' and save_suffix.lower() in ['.jpeg', '.jpg']:
                        save_image = save_image.convert('RGB')
                    elif save_image.mode == 'RGBA' and save_suffix.lower() in ['.jpeg', '.jpg', '.bmp']:
                         save_image = save_image.convert('RGB')
                    save_image.save(temp_save_path)

                    if PYPERCLIP_AVAILABLE:
                        pyperclip.copy(str(temp_save_path))
                        self._show_timed_message("部分成功", f"图片已保存,路径已复制:\n{temp_save_path}", success=True, duration_ms=2500)
                        self.after(0, lambda: self.drop_target_label_mini.config(text="路径已复制", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
                    else:
                        messagebox.showinfo("处理完成", f"图片已处理并保存到:\n{temp_save_path}\n(未能自动复制)", parent=self)
                        self.after(0, lambda: self.drop_target_label_mini.config(text="已保存,请手动复制", bootstyle="info") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
            else:
                messagebox.showwarning("处理结果", "处理后未生成有效图像。", parent=self)
                self.after(0, lambda: self.drop_target_label_mini.config(text="处理后无图像!", bootstyle="warning") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("严重处理错误", f"处理图片时发生意外错误: {e}", parent=self)
            self.after(0, lambda: self.drop_target_label_mini.config(text="严重处理错误!", bootstyle="danger") if self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)
        finally:
             self.after(3000, lambda: self.drop_target_label_mini.config(text="拖放图片到此处", bootstyle="secondary") if self.in_mini_mode and self.drop_target_label_mini and self.drop_target_label_mini.winfo_exists() else None)

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
        val = int(float(value))
        if val % 2 == 0: val +=1
        self.blur_kernel_size_var.set(val)
        self.blur_value_label.config(text=f"{val}")
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
    def select_color(self, color_var, preview_label):
        color_code = colorchooser.askcolor(color=color_var.get(), parent=self)
        if color_code and color_code[1]:
            color_var.set(color_code[1])
            preview_label.config(background=color_code[1])
            self.on_param_change()
    def update_color_preview(self):
        if hasattr(self, 'mist_color_preview') and self.mist_color_preview.winfo_exists():
             self.mist_color_preview.config(background=self.mist_color_var.get())
        if hasattr(self, 'light_color_preview') and self.light_color_preview.winfo_exists():
             self.light_color_preview.config(background=self.light_color_var.get())
    def on_param_change(self, *args):
        if self.in_mini_mode: return
        if current_image_path and original_pil_image:
            self.update_preview()
    def on_mosaic_type_change(self, *args):
        if self.in_mini_mode: return
        mosaic_type = self.mosaic_type_var.get()
        self.blur_params_frame.pack_forget()
        self.line_params_frame.pack_forget()
        self.mist_params_frame.pack_forget()
        self.light_params_frame.pack_forget()
        self.custom_image_button.pack_forget()
        self.custom_image_label.pack_forget()
        self.common_params_frame.pack(fill=X, pady=(5,0))
        if mosaic_type == "常规模糊": self.blur_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "黑色线条": self.line_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "白色雾气": self.mist_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "光效马赛克": self.light_params_frame.pack(fill=X, pady=(5,0))
        elif mosaic_type == "自定义图像":
            self.custom_image_button.pack(fill=X, pady=(5,0))
            self.custom_image_label.pack(fill=X, pady=(0,5))
        if current_image_path and original_pil_image: self.update_preview()
    def on_line_direction_change(self, *args):
        if self.in_mini_mode: return
        if current_image_path and original_pil_image: self.update_preview()
    def select_input_path(self):
        dialog = tb.dialogs.MessageDialog(
            parent=self, title="选择输入类型",
            message="请选择要处理单个图片文件还是整个文件夹？",
            buttons=["图片文件:primary", "文件夹:secondary", "取消:light"])
        dialog.show()
        result = dialog.result
        path = ""
        if result == "图片文件":
            path = filedialog.askopenfilename(
                parent=self, title="选择图片文件",
                filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")))
        elif result == "文件夹":
            path = filedialog.askdirectory(parent=self, title="选择包含图片的文件夹")
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
                current_image_path = path
                self.input_path.set(path)
                self.process_single_button.config(state=DISABLED)
                self.batch_process_button.config(state=NORMAL)
                available_labels = get_available_labels()
                self.update_region_selection_ui(available_labels)
    def load_and_display_original_image(self, image_path):
        global original_pil_image
        try:
            original_pil_image = Image.open(image_path)
            self.display_image_on_label(original_pil_image, self.original_image_label)
            self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
            self.status_label.config(text=f"已加载: {Path(image_path).name}")
            self.cached_detection_results = None
            self.last_detection_conf = None
            self.last_detection_iou = None
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载图片: {e}", parent=self)
            original_pil_image = None
            self.clear_previews()
    def update_available_regions(self):
        if current_image_path and Path(current_image_path).is_file():
            self.status_label.config(text="正在分析图像中的目标...")
            self.update_idletasks()
            def _analyze():
                conf = self.conf_threshold_var.get()
                iou = self.iou_threshold_var.get()
                img_path_str = str(current_image_path)
                names, results, err = get_image_object_names(img_path_str, conf, iou)
                self.last_detection_conf = conf
                self.last_detection_iou = iou
                self.cached_detection_results = results
                final_names = names
                status_text = "目标分析完成。请选择打码区域。"
                if err:
                    status_text = err
                    final_names = get_available_labels()
                elif not names:
                    final_names = get_available_labels()
                self.after(0, lambda: self.status_label.config(text=status_text))
                self.after(0, lambda: self.update_region_selection_ui(final_names))
            threading.Thread(target=_analyze, daemon=True).start()
    def update_region_selection_ui(self, region_names):
        self.available_regions = region_names
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.selected_regions_vars.clear()
        if not region_names:
            tb.Label(self.scrollable_frame, text="当前图片未识别到可选区域").pack(padx=5, pady=5)
            return
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
        if self.in_mini_mode: return
        if original_pil_image and current_image_path:
            self.update_preview()
    def select_custom_image(self):
        global custom_mosaic_image_path
        path = filedialog.askopenfilename(
            parent=self, title="选择自定义贴图 (推荐PNG透明底)",
            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
        if path:
            custom_mosaic_image_path = path
            self.custom_image_path_var.set(Path(path).name)
            self.update_preview()
    def select_output_folder(self):
        path = filedialog.askdirectory(parent=self, title="选择输出文件夹")
        if path:
            self.output_folder.set(path)
    def get_selected_regions(self):
        return [name for name, var in self.selected_regions_vars.items() if var.get()]
    def get_current_parameters(self):
        mist_color_hex = self.mist_color_var.get()
        light_color_hex = self.light_color_var.get()
        mist_color = tuple(int(mist_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        light_color = tuple(int(light_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        blur_kernel_size = self.blur_kernel_size_var.get()
        if blur_kernel_size % 2 == 0: blur_kernel_size += 1
        return {"mosaic_type": self.mosaic_type_var.get(), "line_direction": self.line_direction_var.get(),
            "conf_threshold": self.conf_threshold_var.get(), "iou_threshold": self.iou_threshold_var.get(),
            "scale": self.scale_var.get(), "alpha": self.alpha_var.get(),
            "blur_kernel_size": (blur_kernel_size, blur_kernel_size),
            "line_thickness": self.line_thickness_var.get(), "line_spacing": self.line_spacing_var.get(),
            "mist_color": mist_color, "light_intensity": self.light_intensity_var.get(),
            "light_feather": self.light_feather_var.get(), "light_color": light_color}
    def process_current_image(self):
        if not current_image_path or not original_pil_image:
            messagebox.showwarning("操作无效", "请先选择一张图片。", parent=self)
            return
        self.status_label.config(text="正在处理当前图片...")
        self.progress_bar.start()
        self.update_idletasks()
        params = self.get_current_parameters()
        selected_regions = self.get_selected_regions()
        def _process():
            global processed_pil_image
            img_path_str = str(current_image_path)
            custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" and custom_mosaic_image_path and os.path.exists(custom_mosaic_image_path) else DEFAULT_HEAD_PATH
            current_conf = params["conf_threshold"]
            current_iou = params["iou_threshold"]
            cached_results = self.cached_detection_results
            if self.last_detection_conf != current_conf or self.last_detection_iou != current_iou:
                cached_results = None
                self.after(0, lambda: self.status_label.config(text="检测参数已变更，正在重新检测..."))
            _, proc_img, error = process_single_image(
                img_path_str, params["mosaic_type"], selected_regions, custom_path,
                params["line_direction"], current_conf, current_iou, params["scale"],
                params["alpha"], params["blur_kernel_size"], params["line_thickness"],
                params["line_spacing"], params["mist_color"], params["light_intensity"],
                params["light_feather"], params["light_color"], cached_detection_results=cached_results)
            if cached_results is None:
                if detection_model:
                    self.cached_detection_results = detect_censors(img_path_str, detection_model, current_conf, current_iou)
                else:
                    self.cached_detection_results = []
                self.last_detection_conf = current_conf
                self.last_detection_iou = current_iou

            self.after(0, lambda: self.progress_bar.stop())
            if error:
                messagebox.showerror("处理失败", error, parent=self)
                self.after(0, lambda: self.status_label.config(text=f"处理失败: {error}"))
                processed_pil_image = None
                self.after(0, lambda: self.display_image_on_label(None, self.processed_image_label, "效果预览区域"))
            else:
                processed_pil_image = proc_img
                self.after(0, lambda: self.display_image_on_label(processed_pil_image, self.processed_image_label))
                self.after(0, lambda: self.status_label.config(text="图片处理完成。预览已更新。"))
                self.after(0, self.prompt_save_processed_image)
        threading.Thread(target=_process, daemon=True).start()
    def prompt_save_processed_image(self):
        if processed_pil_image:
            if messagebox.askyesno("保存图片", "处理完成，是否保存打码后的图片？", parent=self):
                self.save_processed_image()
    def save_processed_image(self):
        if not processed_pil_image:
            messagebox.showwarning("无法保存", "没有已处理的图片可供保存。", parent=self)
            return
        if not current_image_path:
            messagebox.showwarning("无法保存", "原始图片路径未知。", parent=self)
            return
        original_file_path = Path(current_image_path)
        output_dir = Path(self.output_folder.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{original_file_path.stem}_processed{original_file_path.suffix}"
        save_path = filedialog.asksaveasfilename(
            parent=self, initialdir=str(output_dir), initialfile=default_name,
            defaultextension=original_file_path.suffix,
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")))
        if save_path:
            try:
                save_image = processed_pil_image.copy()
                save_path_obj = Path(save_path)
                suffix_lower = save_path_obj.suffix.lower()
                if save_image.mode == 'P' and suffix_lower in ['.jpeg', '.jpg']:
                    save_image = save_image.convert('RGB')
                elif save_image.mode == 'RGBA' and suffix_lower in ['.jpeg', '.jpg', '.bmp']:
                     save_image = save_image.convert('RGB')
                save_image.save(save_path)
                messagebox.showinfo("保存成功", f"图片已保存到: {save_path}", parent=self)
                self.status_label.config(text=f"已保存: {Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片时出错: {e}", parent=self)
    def update_preview(self):
        if self.in_mini_mode: return
        if not current_image_path or not original_pil_image: return
        selected_regions = self.get_selected_regions()
        params = self.get_current_parameters()
        self.status_label.config(text="正在更新预览...")
        self.update_idletasks()
        img_path_str = str(current_image_path)
        custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" and custom_mosaic_image_path and os.path.exists(custom_mosaic_image_path) else DEFAULT_HEAD_PATH
        def _update_preview_thread():
            global processed_pil_image
            current_conf = params["conf_threshold"]
            current_iou = params["iou_threshold"]
            cached_results = self.cached_detection_results
            if self.last_detection_conf != current_conf or self.last_detection_iou != current_iou:
                cached_results = None
                self.after(0, lambda: self.status_label.config(text="检测参数已变更，正在重新检测..."))
            _, temp_processed_pil, error = process_single_image(
                img_path_str, params["mosaic_type"], selected_regions, custom_path,
                params["line_direction"], current_conf, current_iou, params["scale"],
                params["alpha"], params["blur_kernel_size"], params["line_thickness"],
                params["line_spacing"], params["mist_color"], params["light_intensity"],
                params["light_feather"], params["light_color"], cached_detection_results=cached_results)
            if cached_results is None:
                if detection_model:
                    self.cached_detection_results = detect_censors(img_path_str, detection_model, current_conf, current_iou)
                else:
                    self.cached_detection_results = []
                self.last_detection_conf = current_conf
                self.last_detection_iou = current_iou
            if error:
                self.after(0, lambda: self.status_label.config(text=f"预览更新错误: {error[:100]}..."))
                self.after(0, lambda: self.display_image_on_label(original_pil_image, self.processed_image_label, "预览生成错误"))
                processed_pil_image = None
            else:
                processed_pil_image = temp_processed_pil
                self.after(0, lambda: self.display_image_on_label(processed_pil_image, self.processed_image_label))
                self.after(0, lambda: self.status_label.config(text="预览已更新。"))
        threading.Thread(target=_update_preview_thread, daemon=True).start()
    def start_batch_process(self):
        input_val_str = self.input_path.get()
        if not input_val_str:
            messagebox.showwarning("操作无效", "请先选择一个文件夹或图片文件进行批量处理。", parent=self)
            return
        input_path_obj = Path(input_val_str)
        if not input_path_obj.exists():
             messagebox.showerror("错误", f"输入路径不存在: {input_val_str}", parent=self)
             return
        output_val = self.output_folder.get()
        if not output_val:
            messagebox.showwarning("操作无效", "请选择输出文件夹。", parent=self)
            return
        params = self.get_current_parameters()
        selected_regions = self.get_selected_regions()
        custom_path = custom_mosaic_image_path if params["mosaic_type"] == "自定义图像" and custom_mosaic_image_path and os.path.exists(custom_mosaic_image_path) else DEFAULT_HEAD_PATH
        self.batch_process_button.config(state=DISABLED)
        self.process_single_button.config(state=DISABLED)
        self.progress_bar['value'] = 0
        def _batch_thread():
            def progress_cb(current, total):
                self.after(0, lambda: self.progress_bar.config(value=(current / total) * 100))
            def status_cb(message):
                self.after(0, lambda: self.status_label.config(text=message))
            def image_preview_cb(original, processed):
                self.after(0, lambda: self.display_image_on_label(original, self.original_image_label, "原图预览"))
                if processed:
                    self.after(0, lambda: self.display_image_on_label(processed, self.processed_image_label))
                else:
                    self.after(0, lambda: self.display_image_on_label(original, self.processed_image_label, "处理失败"))
            batch_process_images(
                str(input_path_obj), output_val, params["mosaic_type"], selected_regions, custom_path,
                params["line_direction"], params["conf_threshold"], params["iou_threshold"],
                params["scale"], params["alpha"], params["blur_kernel_size"],
                params["line_thickness"], params["line_spacing"], params["mist_color"],
                params["light_intensity"], params["light_feather"], params["light_color"],
                progress_callback=progress_cb, status_callback=status_cb, image_preview_callback=image_preview_cb)
            self.after(0, lambda: messagebox.showinfo("批量处理完成", f"所有文件已处理完毕。\n输出到: {output_val}", parent=self))
            self.after(0, lambda: self.batch_process_button.config(state=NORMAL if input_path_obj.is_dir() else DISABLED))
            self.after(0, lambda: self.process_single_button.config(state=NORMAL if input_path_obj.is_file() else DISABLED))
            self.after(0, lambda: self.progress_bar.config(value=0))
            self.after(0, lambda: self.status_label.config(text="准备就绪"))
        threading.Thread(target=_batch_thread, daemon=True).start()
    def display_image_on_label(self, pil_image, label_widget, placeholder_text="图像区域"):
        if not label_widget or not label_widget.winfo_exists() or not label_widget.master.winfo_exists():
            return
        try:
            if pil_image:
                label_widget.master.update_idletasks()
                frame_width = label_widget.master.winfo_width()
                frame_height = label_widget.master.winfo_height()

                if frame_width <= 1 or frame_height <= 1:
                    label_widget.after(100, lambda: self.display_image_on_label(pil_image, label_widget, placeholder_text))
                    return

                img_copy = pil_image.copy()
                original_width, original_height = img_copy.size
                if original_width == 0 or original_height == 0:
                    label_widget.image = None
                    label_widget.config(image="", text=placeholder_text + "\n(图像尺寸为0)")
                    return

                ratio_w = frame_width / original_width
                ratio_h = frame_height / original_height
                scale_ratio = min(ratio_w, ratio_h, 1.0)

                new_width = int(original_width * scale_ratio)
                new_height = int(original_height * scale_ratio)

                if new_width > 0 and new_height > 0:
                    img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img_copy)
                    label_widget.image = photo
                    label_widget.config(image=photo, text="")
                else:
                    label_widget.image = None
                    label_widget.config(image="", text=placeholder_text + "\n(缩放后尺寸无效)")
            else:
                label_widget.image = None
                label_widget.config(image="", text=placeholder_text)
        except Exception as e:
            print(f"显示图像错误: {e}")
            if label_widget and label_widget.winfo_exists():
                label_widget.image = None
                label_widget.config(image="", text=f"{placeholder_text}\n(显示错误: {str(e)[:30]})")

    def clear_previews(self):
        global original_pil_image, processed_pil_image
        original_pil_image = None
        processed_pil_image = None
        if hasattr(self, 'original_image_label') and self.original_image_label.winfo_exists():
            self.display_image_on_label(None, self.original_image_label, "原图区域")
        if hasattr(self, 'processed_image_label') and self.processed_image_label.winfo_exists():
            self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
        if hasattr(self, 'process_single_button') and self.process_single_button.winfo_exists():
            self.process_single_button.config(state=DISABLED)

if __name__ == "__main__":
    Path("assets").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    if not Path(DEFAULT_HEAD_PATH).exists():
        print(f"警告: 默认自定义贴图 {DEFAULT_HEAD_PATH} 未找到。")
    app = ImageMosaicApp()
    app.mainloop()
