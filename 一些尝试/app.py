import customtkinter as ctk
import tkinter as tk
import webbrowser

# 设置全局外观
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 1. 窗口基础设置 ---
        self.title("SLM 控制系统")
        # === 修改点 1: 尺寸调整为 350x600 ===
        self.geometry("350x615")
        self.resizable(False, False)
        
        # --- 2. 配色方案 ---
        self.colors = {
            "bg_root": "#1e1e1e",
            "bg_frame": "#252526",
            "text_main": "#ffffff",
            "text_sub": "#858585",
            "primary": "#007acc",
            "primary_hover": "#0062a3",
            "success": "#2e8b57",
            "success_hover": "#256f46",
            "input_bg": "#3c3c3c",
            "border": "#3e3e42",
            "preview_bg": "#000000",
            "tab_unselected": "#2d2d2d"
        }
        
        self.configure(fg_color=self.colors["bg_root"])
        
        # --- 字体定义 ---
        self.font_ui = ("Microsoft YaHei UI", 11)
        self.font_ui_bold = ("Microsoft YaHei UI", 11, "bold")
        self.font_small = ("Microsoft YaHei UI", 10)
        self.font_mono = ("Consolas", 10)

        # --- 3. 构建主界面 ---
        self._build_layout()

    def _build_layout(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 1. 标题栏
        self._build_header()
        
        # 2. 预览区 (加大版)
        self._build_preview_window()
        
        # 3. 选项卡 (功能区)
        self._build_tabs()

    def _build_header(self):
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=24)
        header_frame.pack(fill="x", pady=(0, 4))
        
        ctk.CTkLabel(
            header_frame, text="SLM 光场调控终端", 
            font=("Microsoft YaHei UI", 13, "bold"), 
            text_color=self.colors["text_main"]
        ).pack(side="left")
        
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.pack(side="right")
        cv = tk.Canvas(status_frame, width=8, height=8, bg=self.colors["bg_root"], highlightthickness=0)
        cv.pack(side="left", padx=5)
        cv.create_oval(0, 0, 8, 8, fill=self.colors["success"], outline="")
        ctk.CTkLabel(status_frame, text="System Ready", font=("Arial", 9), text_color=self.colors["text_sub"]).pack(side="left")

    def _build_preview_window(self):
        """全局预览窗口 (位于 Tab 上方)"""
        # === 修改点 2: 高度增加到 180px，视野更大 ===
        self.preview_frame = ctk.CTkFrame(
            self.main_container, 
            height=180, 
            corner_radius=4,
            fg_color=self.colors["preview_bg"],
            border_width=1, border_color="#444"
        )
        self.preview_frame.pack(fill="x", pady=(0, 8))
        self.preview_frame.pack_propagate(False)
        
        # 绘制网格
        canvas = tk.Canvas(self.preview_frame, bg="#000000", highlightthickness=0, height=180)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._draw_grid(canvas, 350, 180)
        
        ctk.CTkLabel(self.preview_frame, text="NO SIGNAL", font=self.font_small, text_color="#444").place(relx=0.5, rely=0.5, anchor="center")

    def _build_tabs(self):
        """选项卡构建"""
        self.tabview = ctk.CTkTabview(
            self.main_container,
            fg_color=self.colors["bg_frame"],
            corner_radius=6,
            segmented_button_fg_color=self.colors["tab_unselected"],
            segmented_button_selected_color=self.colors["primary"],
            segmented_button_selected_hover_color=self.colors["primary_hover"],
            segmented_button_unselected_color=self.colors["tab_unselected"],
            segmented_button_unselected_hover_color="#333",
            text_color="#cecece",
        )
        self.tabview.pack(fill="both", expand=True) 
        self.tabview._segmented_button.configure(font=("Microsoft YaHei UI", 10, "bold"))

        self.tab_ctrl = self.tabview.add("设备控制")
        self.tab_cfg = self.tabview.add("配置脚本")
        self.tab_log = self.tabview.add("日志打印")
        self.tab_about = self.tabview.add("关于")
        
        self._build_tab_control()
        self._build_tab_config()
        self._build_tab_logs()
        self._build_tab_about()

    def _build_tab_control(self):
        """Tab 1: 设备控制"""
        frame = self.tab_ctrl
        
        # --- Step 1: 填写配置脚本 ---
        self._add_step_header(frame, "1. 配置脚本路径")
        
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 4))
        
        self.entry_script = ctk.CTkEntry(
            row1, placeholder_text="config/experiment.json",
            height=26, font=self.font_small,
            fg_color=self.colors["input_bg"], border_width=0
        )
        self.entry_script.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        btn_file = ctk.CTkButton(
            row1, text="...", width=30, height=26,
            fg_color="#333", hover_color="#444", text_color="#ccc"
        )
        btn_file.pack(side="right")

        # --- Step 2: 设备初始化 ---
        self._add_step_header(frame, "2. 硬件连接")
        
        btn_init = ctk.CTkButton(
            frame, text="初始化设备 (SLM & Camera)", 
            height=28, font=self.font_ui,
            fg_color=self.colors["primary"], 
            hover_color=self.colors["primary_hover"]
        )
        btn_init.pack(fill="x", pady=(0, 4))

        # --- Step 3: 相机预览控制 ---
        self._add_step_header(frame, "3. 相机预览设定")
        
        btn_preview = ctk.CTkButton(
            frame, text="▶ 开启实时预览", 
            height=28, font=self.font_ui,
            fg_color="#333", hover_color="#444", 
            border_width=1, border_color="#555"
        )
        btn_preview.pack(fill="x", pady=(0, 4))
        
        slider_row = ctk.CTkFrame(frame, fg_color="transparent")
        slider_row.pack(fill="x", pady=(0, 4))
        
        ctk.CTkLabel(slider_row, text="曝光:", font=("Arial", 10), text_color=self.colors["text_sub"]).pack(side="left")
        
        self.slider = ctk.CTkSlider(
            slider_row, from_=0, to=10000, number_of_steps=100,
            height=16, progress_color=self.colors["primary"],
            button_color=self.colors["primary"], 
            button_hover_color=self.colors["primary_hover"]
        )
        self.slider.set(2000)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.val_label = ctk.CTkLabel(slider_row, text="2000", font=("Arial", 10), width=35, text_color=self.colors["text_sub"])
        self.val_label.pack(side="right")
        self.slider.configure(command=lambda v: self.val_label.configure(text=str(int(v))))

        # --- Step 4: 任务执行 ---
        self._add_step_header(frame, "4. 实验任务")
        
        self.task_menu = ctk.CTkOptionMenu(
            frame, 
            values=["Task_A: GS算法迭代", "Task_B: 神经网络推理"],
            height=26, font=self.font_small,
            fg_color=self.colors["input_bg"], button_color=self.colors["input_bg"],
            dropdown_font=self.font_small
        )
        self.task_menu.pack(fill="x", pady=(0, 5))
        
        btn_run = ctk.CTkButton(
            frame, text="★ 开始实验", 
            height=34, font=self.font_ui_bold,
            fg_color=self.colors["success"], 
            hover_color=self.colors["success_hover"]
        )
        btn_run.pack(fill="x", pady=(2, 6))

    def _build_tab_config(self):
        """Tab 2: 配置脚本"""
        self.cfg_box = ctk.CTkTextbox(self.tab_cfg, fg_color="#1a1a1a", text_color="#dcdcdc", font=self.font_mono, wrap="none")
        self.cfg_box.pack(fill="both", expand=True, padx=2, pady=2)
        self.cfg_box.insert("1.0", "{\n    \"experiment\": \"demo\",\n    \"exposure\": 5000\n}")

    def _build_tab_logs(self):
        """Tab 3: 日志"""
        self.log_box = ctk.CTkTextbox(self.tab_log, fg_color="#1a1a1a", text_color="#2cc985", font=self.font_mono)
        self.log_box.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_box.insert("1.0", "[System] Ready...")

    def _build_tab_about(self):
        """Tab 4: 关于"""
        container = ctk.CTkFrame(self.tab_about, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(container, text="SLM", width=54, height=54, corner_radius=12, fg_color=self.colors["primary"], font=("Arial", 20, "bold"), text_color="white").pack(pady=(0, 15))
        ctk.CTkLabel(container, text="光场调控控制系统", font=self.font_ui_bold).pack()
        ctk.CTkLabel(container, text="v2.3.1", font=self.font_small, text_color=self.colors["text_sub"]).pack(pady=(2, 15))
        ctk.CTkButton(container, text="GitHub Repository", height=26, width=130, fg_color="#24292e", hover_color="#000000", font=self.font_small, command=lambda: webbrowser.open("https://github.com")).pack()

    def _add_step_header(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=(4, 2))
        ctk.CTkFrame(f, width=3, height=11, fg_color=self.colors["primary"]).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(f, text=text, font=("Microsoft YaHei UI", 10, "bold"), text_color=self.colors["text_sub"]).pack(side="left")

    def _draw_grid(self, canvas, w, h):
        canvas.delete("grid")
        # 适应新的宽高
        for i in range(0, w + 20, 20): canvas.create_line(i, 0, i, h, fill="#1a1a1a")
        for j in range(0, h + 20, 20): canvas.create_line(0, j, w, j, fill="#1a1a1a")

if __name__ == "__main__":
    app = App()
    app.mainloop()