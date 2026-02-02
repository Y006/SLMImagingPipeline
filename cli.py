import typer
import yaml
import json
import time
import datetime
import threading
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from loguru import logger

from rich.console import Console
from rich.table import Table
from rich.json import JSON
from rich.panel import Panel
from rich import box
# 初始化 Rich 控制台
console = Console()

# 引入硬件
from src.hardware.screen import display_image
from src.hardware.slm import SLM
from src.hardware.camera import HikCamera
from src.hardware.MvImport.MvCameraControl_class import MV_Image_Bmp
from src.utils.config_utils import task_codes_from_time

DEFAULT_CONFIG_PATH = "configs/single_shot_config.yaml"

app = typer.Typer(help="SLM 成像控制终端 (Decoupled Logic)")

# ==============================================================================
# 1. 公共工具函数 (仅负责脏活累活：加载配置、创建文件夹、保存图片)
#    不要在这里写任何硬件控制逻辑
# ==============================================================================

def load_context(config_path_str: str):
    """加载配置并准备输出目录"""
    # 1. 读 YAML
    path = Path(config_path_str)
    if not path.is_absolute(): path = Path.cwd() / path
    if not path.exists():
        logger.error(f"配置文件缺失: {path}")
        raise typer.Exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 准备目录
    proj = cfg.get("project", {})
    root = Path(proj.get("root_dir", "./output"))
    if not root.is_absolute(): root = Path.cwd() / root
    
    save_dir = root / proj.get("id", "default")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return cfg, save_dir

def save_meta_image(bmp_path: Path, metadata: dict):
    """保存带元数据的 PNG"""
    try:
        png_path = bmp_path.with_suffix(".png")
        img = Image.open(bmp_path)
        info = PngInfo()
        for k, v in metadata.items():
            val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, int, float)) else str(v)
            info.add_text(k, val)
        img.save(png_path, "PNG", pnginfo=info)
        bmp_path.unlink(missing_ok=True)
        logger.success(f"已保存: {png_path.name}")
    except Exception as e:
        logger.error(f"保存失败: {e}")

# ==============================================================================
# 2. 具体的业务逻辑函数 (各自独立，不互相调用，没有 True/False 开关)
# ==============================================================================

from rich.table import Table
from rich import box
from rich.text import Text

# ... 之前的 imports ...

def render_complex_data(data):
    """
    辅助函数：将字典/列表渲染为无边框表格，确保路径不被截断
    """
    if isinstance(data, dict):
        if not data:
            return "[dim italic](空)[/]"
        
        # 使用 Grid，不设置宽度限制
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cyan bold", justify="left", no_wrap=True) # Key 不换行
        grid.add_column(style="white", overflow="fold") # Value 自动折行而不是截断
        
        for k, v in data.items():
            grid.add_row(str(k), render_complex_data(v))
        return grid

    elif isinstance(data, list):
        if not data:
            return "[dim italic](空列表)[/]"
        grid = Table.grid(padding=(0, 1))
        for item in data:
            grid.add_row("•", render_complex_data(item))
        return grid
        
    else:
        val_str = str(data)
        if val_str == "":
            return "[dim](未填写)[/]"
        
        # [关键修改]: 简单的路径识别高亮
        # 如果看起来像绝对路径 (包含 :/) 或长路径，用下划线+黄色显示
        if ":/" in val_str or "/" in val_str or "\\" in val_str:
            return Text(val_str, style="yellow underline")
            
        return val_str

def inspect_png_meta(image_path: str):
    path = Path(image_path)
    
    if not path.exists():
        console.print(f"[bold red]错误: 文件不存在 -> {path}[/]")
        return

    try:
        with Image.open(path) as img:
            meta = img.info
            img_format = img.format
            img_size = img.size
            img_mode = img.mode

        # 顶部基础信息
        info_text = f"📐 [b]{img_size[0]}x{img_size[1]}[/b]   |   🎨 [b]{img_mode}[/b]   |   💾 [b]{img_format}[/b]"
        
        # [修改点1]: 限制 Panel 宽度，避免过宽；Table 会根据内容自适应
        console.print(Panel(info_text, style="blue", box=box.ROUNDED, width=80)) 

        if not meta:
            console.print(Panel(f"图片: [bold cyan]{path.name}[/]\n无嵌入元数据", style="yellow"))
            return

        # --- 主表格构建 ---
        table = Table(
            title=f"元数据透视: {path.name}",
            box=box.ROUNDED,       
            border_style="bright_blue", 
            show_header=True,
            header_style="bold magenta",
            # [修改点3]: 使用 collapse_padding=True 稍微紧凑一点，留空间给内容
            collapse_padding=True,
            show_lines=True,
            # 让表格根据内容扩展，但可以设置最小宽度
            min_width=80  # 可选：确保表格至少有 80 字符宽
        )
        
        # [修改点4]: 允许内容列自动换行 (overflow="fold")，绝对禁止省略 ("ellipsis")
        table.add_column("字段 (Field)", style="green", no_wrap=True) 
        table.add_column("详细内容 (Content)", style="white", overflow="fold")

        # 排序策略
        priority_keys = ["mode", "timestamp", "project_id", "note", "repeat_idx"]
        sorted_keys = sorted(
            meta.keys(), 
            key=lambda x: (0 if x in priority_keys else 1, x)
        )

        for key in sorted_keys:
            raw_value = meta[key]
            try:
                parsed_data = json.loads(raw_value)
            except (json.JSONDecodeError, TypeError):
                parsed_data = raw_value

            renderable = render_complex_data(parsed_data)
            table.add_row(key, renderable)

        # 输出表格
        console.print(table)

    except Exception as e:
        console.print(f"[bold red]读取失败: {e}[/]")

def run_psf_logic(config_path: str, repeat: int, note: str):
    """
    流程：加载配置 -> 过滤参数(Capture+Physical) -> 初始化SLM -> 初始化相机 -> 拍摄
    """
    # 1. 加载上下文
    config, save_dir = load_context(config_path)
    raw_cap_conf = config.get("capture_settings", {})

    # --- 路径检查 ---
    slm_path_str = raw_cap_conf.get("slm_image_path")
    if not slm_path_str:
        logger.error("配置错误: PSF 模式必须指定 slm_image_path")
        raise typer.Exit(code=1)
    
    slm_path = Path(slm_path_str)
    if not slm_path.is_absolute():
        slm_path = Path.cwd() / slm_path
    
    if not slm_path.exists():
        logger.error(f"SLM 图片不存在: {slm_path}")
        raise typer.Exit(code=1)

    slm = None
    cam = None

    try:
        # --- 2. 初始化 SLM ---
        slm = SLM(verbose=True)
        if not slm.init(): 
            logger.error("SLM 初始化失败")
            raise typer.Exit(code=2)
            
        slm.img_show(str(slm_path))
        time.sleep(1)

        # --- 3. 初始化相机 ---
        cam = HikCamera(dev_index=0)
        if not cam.open():
            logger.error("相机打开失败")
            raise typer.Exit(code=3)

        # =========================================================
        # [修改点 1]：清洗 Capture Settings (只保留曝光和SLM路径)
        # =========================================================
        # 定义 PSF 模式下，capture_settings 里哪些是有用的
        valid_setting_keys = ["exposure_us", "slm_image_path"]
        
        # 提取并清洗，自动忽略 monitor_idx 等无关参数
        clean_settings = {
            k: raw_cap_conf.get(k) 
            for k in valid_setting_keys 
            if k in raw_cap_conf
        }

        # =========================================================
        # [修改点 2]：清洗 Physical Setup (只保留系统固有参数)
        # =========================================================
        full_phy = config.get("physical_setup", {})
        valid_phy_keys = [
            "sensor_mask_distance", 
            "mask_object_distance", 
            "point_light_brightness", 
            "mask_LED-light_distance"
        ]
        clean_physical = {
            k: full_phy.get(k) 
            for k in valid_phy_keys 
            if k in full_phy
        }
        # =========================================================

        # --- 4. 拍摄循环 ---
        exp = raw_cap_conf.get("exposure_us", 20000)
        timeout = raw_cap_conf.get("camera_timeout_ms", 5000)
        logger.info(f"开始采集 PSF | 曝光: {exp}us | 数量: {repeat}")

        for i in range(repeat):
            code4, task_id = task_codes_from_time("psf", k=4)
            suffix = f"_{i+1:02d}" if repeat > 1 else ""
            tmp_bmp = save_dir / f"psf_{task_id}{suffix}.bmp"

            if cam.snap(tmp_bmp, exposure_us=exp, timeout_ms=timeout, img_type=MV_Image_Bmp):
                meta = {
                    "mode": "psf",
                    "note": note,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "repeat_idx": i + 1,
                    
                    # 使用清洗后的纯净数据
                    "settings": clean_settings, 
                    "physical": clean_physical,
                    
                    "project": config.get("project")
                }
                save_meta_image(tmp_bmp, meta)
                
                if repeat > 1:
                    time.sleep(0.2)
            else:
                logger.warning(f"拍摄失败: {i+1}/{repeat}")

    except Exception as e:
        logger.exception(f"PSF 运行时错误: {e}")
        raise typer.Exit(code=4)

    finally:
        if cam: cam.close()
        # if slm: slm.close()

def run_measurement_logic(config_path: str, repeat: int, note: str):
    """
    [Measurement 专用逻辑]
    流程：加载配置 -> 路径检查(SLM+Display) -> 启动显示 -> 初始化SLM -> 初始化相机 -> 拍摄
    """
    # 1. 加载上下文
    config, save_dir = load_context(config_path)
    raw_cap_conf = config.get("capture_settings", {})

    # ==========================================
    # 路径安全检查 (SLM + Display)
    # ==========================================
    
    # 1. SLM 路径检查
    slm_path_str = raw_cap_conf.get("slm_image_path")
    if not slm_path_str:
        logger.error("配置错误: Measurement 模式必须指定 slm_image_path")
        raise typer.Exit(code=1)
    slm_path = Path(slm_path_str)
    if not slm_path.is_absolute(): slm_path = Path.cwd() / slm_path
    if not slm_path.exists():
        logger.error(f"SLM 图片不存在: {slm_path}")
        raise typer.Exit(code=1)

    # 2. Display 路径检查 (新增)
    disp_path_str = raw_cap_conf.get("display_image_path")
    if not disp_path_str:
        logger.error("配置错误: Measurement 模式必须指定 display_image_path")
        raise typer.Exit(code=1)
    disp_path = Path(disp_path_str)
    if not disp_path.is_absolute(): disp_path = Path.cwd() / disp_path
    if not disp_path.exists():
        logger.error(f"显示图片不存在: {disp_path}")
        raise typer.Exit(code=1)

    # ==========================================
    # 资源初始化
    # ==========================================
    slm = None
    cam = None
    # display 运行在 daemon 线程，随主进程退出，无需显式 close 句柄

    try:
        # --- 2. 启动显示器 (Display) ---
        # 放到 try 块里，虽然线程启动一般不报错，但保持逻辑统一
        mon_idx = raw_cap_conf.get("monitor_idx", 1)
        scale_f = raw_cap_conf.get("scale_factor", 1.0)
        
        logger.info(f"正在显示图片: {disp_path.name} (显示器 {mon_idx})")
        t = threading.Thread(
            target=display_image,
            args=(str(disp_path), mon_idx, scale_f),
            daemon=True
        )
        t.start()
        # 给显示器一点点反应时间 (可选)
        time.sleep(0.5)

        # --- 3. 初始化 SLM ---
        slm = SLM(verbose=True)
        if not slm.init(): 
            logger.error("SLM 初始化失败")
            raise typer.Exit(code=2)
            
        slm.img_show(str(slm_path))
        wait_slm = raw_cap_conf.get("slm_settle_time_s", 0.1) 
        time.sleep(wait_slm)

        # --- 4. 初始化相机 ---
        cam = HikCamera(dev_index=0)
        if not cam.open():
            logger.error("相机打开失败")
            raise typer.Exit(code=3)

        # =========================================================
        # [清洗 1]：Capture Settings (保留显示器和SLM参数)
        # =========================================================
        valid_setting_keys = [
            "exposure_us", 
            "slm_image_path", 
            "display_image_path", 
            "monitor_idx", 
            "scale_factor"
        ]
        clean_settings = {
            k: raw_cap_conf.get(k) 
            for k in valid_setting_keys 
            if k in raw_cap_conf
        }

        # =========================================================
        # [清洗 2]：Physical Setup (保留物体和几何结构参数)
        # =========================================================
        full_phy = config.get("physical_setup", {})
        
        # 剔除了 point_light_brightness (PSF专用)
        # 保留了 object_name, brightness (屏幕亮度/物体亮度)
        valid_phy_keys = [
            "object_name",
            "brightness", 
            "sensor_mask_distance", 
            "mask_object_distance"
        ]
        clean_physical = {
            k: full_phy.get(k) 
            for k in valid_phy_keys 
            if k in full_phy
        }
        # =========================================================

        # --- 5. 拍摄循环 ---
        exp = raw_cap_conf.get("exposure_us", 20000)
        timeout = raw_cap_conf.get("camera_timeout_ms", 5000)
        
        logger.info(f"开始采集 Measurement | 曝光: {exp}us | 数量: {repeat}")

        for i in range(repeat):
            # 使用 "meas" 或 "m" 作为文件前缀
            code4, task_id = task_codes_from_time("m", k=4)
            suffix = f"_{i+1:02d}" if repeat > 1 else ""
            tmp_bmp = save_dir / f"meas_{task_id}{suffix}.bmp"

            if cam.snap(tmp_bmp, exposure_us=exp, timeout_ms=timeout, img_type=MV_Image_Bmp):
                meta = {
                    "mode": "measurement",
                    "note": note,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "repeat_idx": i + 1,
                    
                    # 使用清洗后的数据
                    "settings": clean_settings, 
                    "physical": clean_physical,
                    
                    "project": config.get("project")
                }
                save_meta_image(tmp_bmp, meta)
                
                if repeat > 1:
                    time.sleep(0.2)
            else:
                logger.warning(f"拍摄失败: {i+1}/{repeat}")

    except Exception as e:
        logger.exception(f"Measurement 运行时错误: {e}")
        raise typer.Exit(code=4)

    finally:
        if cam: cam.close()
        # if slm: slm.close()

def run_calibration_logic(config_path: str):
    """
    [标定专用逻辑]
    流程：加载配置 -> 启动显示器 -> 启动 SLM -> 挂起等待
    特点：完全没有相机代码
    """
    cfg, _ = load_context(config_path) # 不需要 save_dir，因为不存图
    cap = cfg.get("capture_settings", {})

    # --- 1. 显示器 ---
    disp_path = cap.get("display_image_path")
    if disp_path:
        logger.info(f"显示图片: {disp_path}")
        t = threading.Thread(
            target=display_image,
            args=(disp_path, cap.get("monitor_idx", 1), cap.get("scale_factor", 1.0)),
            daemon=True
        )
        t.start()
    
    # --- 2. SLM ---
    slm_path = cap.get("slm_image_path")
    if slm_path:
        logger.info(f"SLM 加载: {slm_path}")
        slm = SLM(verbose=True)
        slm.init()
        slm.img_show(slm_path)
    
    # --- 3. 挂起 ---
    logger.info(">>> 系统已就绪，正在显示图案... (按 Ctrl+C 退出) <<<")
    # 这里不需要相机，所以根本不 import/init 相机类
    time.sleep(999999)

# ==============================================================================
# 3. CLI 入口 (非常干净，只负责传参)
# ==============================================================================

@app.command()
def capture_psf(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """采集 PSF"""
    run_psf_logic(config, repeat, note)

@app.command()
def capture_measurement(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """采集测量数据"""
    run_measurement_logic(config, repeat, note)

@app.command()
def calibrate(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")
):
    """标定/预览模式"""
    run_calibration_logic(config)

@app.command()
def inspect(
    filepath: str = typer.Argument(..., help="要检查的 PNG 图片路径")
):
    """
    [工具] 查看 PNG 图片中嵌入的实验元数据
    """
    inspect_png_meta(filepath)

if __name__ == "__main__":
    app()