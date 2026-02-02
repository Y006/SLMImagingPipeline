"""
元数据检查工具

提供 PNG 图片元数据的可视化查看功能
"""

import json
from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def render_complex_data(data):
    """
    辅助函数：将字典/列表渲染为无边框表格，确保路径不被截断
    
    Args:
        data: 要渲染的数据（dict, list 或其他）
        
    Returns:
        Rich renderable object
    """
    if isinstance(data, dict):
        if not data:
            return "[dim italic](空)[/]"
        
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cyan bold", justify="left", no_wrap=True)
        grid.add_column(style="white", overflow="fold")
        
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
        
        # 路径识别高亮
        if ":/" in val_str or "/" in val_str or "\\" in val_str:
            return Text(val_str, style="yellow underline")
            
        return val_str


def inspect_png_meta(image_path: str):
    """
    查看 PNG 图片中嵌入的元数据
    
    Args:
        image_path: PNG 图片路径
    """
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
            collapse_padding=True,
            show_lines=True,
            min_width=80
        )
        
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
