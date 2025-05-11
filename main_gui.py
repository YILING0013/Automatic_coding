# main_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    DEFAULT_HEAD_PATH
)
from utils import load_models

# 全局变量
current_image_path = None
original_pil_image = None
processed_pil_image = None
custom_mosaic_image_path = None
custom_mosaic_pil_image = None

# 加载模型
classification_model, detection_model = load_models()
if not detection_model:
    print("警告：检测模型加载失败，应用功能将受限。")

class ImageMosaicApp(tb.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("图像打码工具")
        self.geometry("1200x800")

        # 数据
        self.input_path = tk.StringVar()
        self.output_folder = tk.StringVar(value=str(Path.home() / "图像打码输出"))
        self.mosaic_type_var = tk.StringVar(value="常规模糊")
        self.available_regions = []
        self.selected_regions_vars = {}
        self.custom_image_path_var = tk.StringVar(value="使用默认图案")
        self.line_direction_var = tk.StringVar(value="horizontal")

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

        # 2. 打码方式选择
        mosaic_frame = tb.Labelframe(controls_frame, text="选择打码方式", padding=5)
        mosaic_frame.pack(fill=X, pady=10)
        mosaic_options = ["常规模糊", "黑色线条", "白色雾气", "自定义图像"]
        for option in mosaic_options:
            rb = tb.Radiobutton(mosaic_frame, text=option, variable=self.mosaic_type_var, value=option, command=self.on_mosaic_type_change)
            rb.pack(anchor=W)
        
        # 线条方向选择
        self.line_direction_frame = tb.Labelframe(mosaic_frame, text="线条方向", padding=5)
        line_directions = [("水平", "horizontal"), ("垂直", "vertical"), ("斜线", "diagonal")]
        for text, value in line_directions:
            rb = tb.Radiobutton(self.line_direction_frame, text=text, variable=self.line_direction_var, value=value, command=self.on_line_direction_change)
            rb.pack(anchor=W)
        
        # 自定义图像选择按钮
        self.custom_image_button = tb.Button(mosaic_frame, text="选择自定义贴图", command=self.select_custom_image, bootstyle=INFO)
        self.custom_image_label = tb.Label(mosaic_frame, textvariable=self.custom_image_path_var, wraplength=180)
        
        self.on_mosaic_type_change()

        # 3. 区域选择
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

        # 4. 输出文件夹选择
        output_frame = tb.Frame(controls_frame)
        output_frame.pack(fill=X, pady=5, side=BOTTOM)
        tb.Button(output_frame, text="选择输出文件夹", command=self.select_output_folder, bootstyle=SECONDARY).pack(side=LEFT, padx=5)
        tb.Entry(output_frame, textvariable=self.output_folder, state="readonly").pack(side=LEFT, fill=X, expand=YES)
        
        # 5. 处理按钮
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

        self.original_image_label = tb.Label(preview_frame, text="原图区域", relief="solid", anchor=CENTER)
        self.original_image_label.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)
        self.processed_image_label = tb.Label(preview_frame, text="效果预览区域", relief="solid", anchor=CENTER)
        self.processed_image_label.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)

        # 底部：状态栏和进度条
        status_bar_frame = tb.Frame(self, padding=(5,2))
        status_bar_frame.pack(side=BOTTOM, fill=X)
        self.status_label = tb.Label(status_bar_frame, text="准备就绪", anchor=W)
        self.status_label.pack(side=LEFT, padx=5)
        self.progress_bar = tb.Progressbar(status_bar_frame, mode='determinate', length=200, bootstyle=SUCCESS)
        self.progress_bar.pack(side=RIGHT, padx=5)
        
        # 绑定事件
        self.original_image_label.bind("<Configure>", self.resize_preview_images)
        self.processed_image_label.bind("<Configure>", self.resize_preview_images)

    def on_mosaic_type_change(self, *args):
        if self.mosaic_type_var.get() == "自定义图像":
            self.custom_image_button.pack(fill=X, pady=(5,0))
            self.custom_image_label.pack(fill=X, pady=(0,5))
            self.line_direction_frame.pack_forget()
        elif self.mosaic_type_var.get() == "黑色线条":
            self.custom_image_button.pack_forget()
            self.custom_image_label.pack_forget()
            self.line_direction_frame.pack(fill=X, pady=(5,0))
        else:
            self.custom_image_button.pack_forget()
            self.custom_image_label.pack_forget()
            self.line_direction_frame.pack_forget()
        
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
                self.update_region_selection_ui(["所有检测区域"])

    def load_and_display_original_image(self, image_path):
        global original_pil_image
        try:
            original_pil_image = Image.open(image_path)
            self.display_image_on_label(original_pil_image, self.original_image_label)
            self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
            self.status_label.config(text=f"已加载: {Path(image_path).name}")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载图片: {e}")
            original_pil_image = None
            self.clear_previews()

    def update_available_regions(self):
        if current_image_path and Path(current_image_path).is_file():
            self.status_label.config(text="正在分析图像中的目标...")
            self.update()
            
            def _analyze():
                names, err = get_image_object_names(current_image_path)
                if err:
                    self.status_label.config(text=err)
                    self.update_region_selection_ui(["所有检测区域"])
                else:
                    self.status_label.config(text="目标分析完成。请选择打码区域。")
                    self.update_region_selection_ui(names if names else ["所有检测区域"])
            
            threading.Thread(target=_analyze, daemon=True).start()

    def update_region_selection_ui(self, region_names):
        self.available_regions = region_names
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.selected_regions_vars.clear()

        if not region_names:
            tb.Label(self.scrollable_frame, text="当前图片未识别到可选区域").pack(padx=5, pady=5)
            return

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
        selected = [name for name, var in self.selected_regions_vars.items() if var.get()]
        # 如果选择了"所有检测区域"，返回空列表表示不过滤
        if "所有检测区域" in selected:
            return []
        return selected

    def process_current_image(self):
        if not current_image_path or not original_pil_image:
            messagebox.showwarning("操作无效", "请先选择一张图片。")
            return
        
        self.status_label.config(text="正在处理当前图片...")
        self.progress_bar.start()
        self.update()

        mosaic_type = self.mosaic_type_var.get()
        selected_regions = self.get_selected_regions()
        line_direction = self.line_direction_var.get()
        
        def _process():
            global processed_pil_image
            custom_path = custom_mosaic_image_path if mosaic_type == "自定义图像" else None
            
            _, proc_img, error = process_single_image(
                current_image_path, mosaic_type, selected_regions, custom_path, line_direction
            )
            
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
                processed_pil_image.save(save_path)
                messagebox.showinfo("保存成功", f"图片已保存到: {save_path}")
                self.status_label.config(text=f"已保存: {Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片时出错: {e}")

    def update_preview(self):
        if not current_image_path or not original_pil_image:
            return

        selected_regions = self.get_selected_regions()
        self.status_label.config(text="正在更新预览...")
        self.update()

        mosaic_type = self.mosaic_type_var.get()
        custom_path = custom_mosaic_image_path if mosaic_type == "自定义图像" else None
        line_direction = self.line_direction_var.get()
        
        def _update_preview_thread():
            global processed_pil_image
            _, temp_processed_pil, error = process_single_image(
                current_image_path, mosaic_type, selected_regions, custom_path, line_direction
            )
            
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

        mosaic_type = self.mosaic_type_var.get()
        selected_regions = self.get_selected_regions()
        custom_path = custom_mosaic_image_path if mosaic_type == "自定义图像" else None
        line_direction = self.line_direction_var.get()

        self.batch_process_button.config(state=DISABLED)
        self.process_single_button.config(state=DISABLED)
        self.progress_bar['value'] = 0  # 修复这里

        def _batch_thread():
            def progress_cb(current, total):
                self.progress_bar['value'] = (current / total) * 100
                self.update_idletasks()

            def status_cb(message):
                self.status_label.config(text=message)
                self.update_idletasks()

            batch_process_images(
                input_val, output_val, mosaic_type, selected_regions, custom_path, line_direction,
                progress_callback=progress_cb,
                status_callback=status_cb
            )
            messagebox.showinfo("批量处理完成", f"所有文件已处理完毕。\n输出到: {output_val}")
            self.batch_process_button.config(state=NORMAL if Path(input_val).is_dir() else DISABLED)
            self.process_single_button.config(state=NORMAL if Path(input_val).is_file() else DISABLED)
            self.progress_bar['value'] = 0  # 修复这里
            self.status_label.config(text="准备就绪")

        threading.Thread(target=_batch_thread, daemon=True).start()

    def display_image_on_label(self, pil_image, label_widget, placeholder_text="图像区域"):
        if pil_image:
            label_width = label_widget.winfo_width()
            label_height = label_widget.winfo_height()

            if label_width <= 1 or label_height <= 1:
                label_widget.after(50, lambda: self.display_image_on_label(pil_image, label_widget, placeholder_text))
                return

            img_copy = pil_image.copy()
            
            original_width, original_height = img_copy.size
            ratio_w = label_width / original_width
            ratio_h = label_height / original_height
            scale_ratio = min(ratio_w, ratio_h)

            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)

            if new_width > 0 and new_height > 0:
                img_copy.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_copy)
                label_widget.config(image=photo, text="")
                label_widget.image = photo
            else:
                label_widget.config(image=None, text=placeholder_text + "\n(图像过小或尺寸错误)")
                label_widget.image = None
        else:
            label_widget.config(image=None, text=placeholder_text)
            label_widget.image = None

    def resize_preview_images(self, event=None):
        if original_pil_image:
            self.display_image_on_label(original_pil_image, self.original_image_label)
        if processed_pil_image:
            self.display_image_on_label(processed_pil_image, self.processed_image_label)
        else:
            self.display_image_on_label(None, self.processed_image_label, "效果预览区域")

    def clear_previews(self):
        global original_pil_image, processed_pil_image, current_image_path
        original_pil_image = None
        processed_pil_image = None
        current_image_path = None
        self.display_image_on_label(None, self.original_image_label, "原图区域")
        self.display_image_on_label(None, self.processed_image_label, "效果预览区域")
        self.update_region_selection_ui([])
        self.process_single_button.config(state=DISABLED)

if __name__ == "__main__":
    Path("assets").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if not Path(DEFAULT_HEAD_PATH).exists():
        print(f"警告: 默认自定义贴图 {DEFAULT_HEAD_PATH} 未找到。")

    app = ImageMosaicApp()
    app.mainloop()