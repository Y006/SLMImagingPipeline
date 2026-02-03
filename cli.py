"""
SLM 成像系统 - 命令行入口
"""

import typer
from loguru import logger
from src.cli import (
    run_psf_logic,
    run_measurement_logic,
    run_calibration_logic,
    inspect_png_meta,
    run_admm_logic,
    run_wiener_logic
)

DEFAULT_CONFIG_PATH = "configs/config.yaml"

# 头部帮助信息：保持简洁，去掉过多的颜色干扰
APP_HELP = """
[bold cyan]SLM 成像系统控制终端：这是一个集成了 [bold green]数据采集[/bold green]、[bold yellow]系统标定[/bold yellow] 和 [bold magenta]图像重建[/bold magenta] 的一站式命令行工具。[/bold cyan] 🚀

[bold]✨ 典型工作流:[/bold]
1. [green]采集 PSF[/green] : [italic]uv run cli.py psf[/italic]
2. [yellow]采集测量[/yellow]: [italic]uv run cli.py meas[/italic]
3. [magenta]Reconstruction[/magenta]: [italic]uv run cli.py ad path/to/image.png[/italic]
5. [cyan]查看捕获图片中记录的元数据[/cyan]: [italic]uv run cli.py ins path/to/image.png[/italic]

[dim]提示: 使用 --help 查看具体命令参数。[/dim]
"""

app = typer.Typer(
    help=APP_HELP, 
    add_completion=False, 
    rich_markup_mode="rich"  # 开启 Rich 渲染模式
)


# ==============================================================================
# 命令定义 (全部回归默认组，确保在一个框内)
# ==============================================================================

@app.command(name="psf", hidden=True)
@app.command(name="capture-psf")
def capture_psf(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """[bold yellow](缩写: psf)[/bold yellow]  [cyan][数据采集][/cyan] 采集 PSF""" 
    run_psf_logic(config, repeat, note)


@app.command(name="meas", hidden=True)
@app.command(name="capture-measurement")
def capture_measurement(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """[bold yellow](缩写: meas)[/bold yellow] [cyan][数据采集][/cyan] 采集测量数据"""
    run_measurement_logic(config, repeat, note)


@app.command(name="sys_cal", hidden=True)
@app.command(name="system_calibrate")
def calibrate(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")
):
    """[bold yellow](缩写: sys_cal)[/bold yellow]  [cyan][辅助工具][/cyan] 系统标定：加载 SLM 和显示器场景图片，便于用相机软件进行预览"""
    run_calibration_logic(config)


@app.command(name="ins", hidden=True)
@app.command(name="inspect")
def inspect(
    filepath: str = typer.Argument(..., help="要检查的 PNG 图片路径")
):
    """[bold yellow](缩写: ins)[/bold yellow]  [cyan][辅助工具][/cyan] 捕获的图片会记录捕获时的 YAML 文件里面的参数到 PNG 文件的元数据，可用该工具进行查看"""
    inspect_png_meta(filepath)


@app.command(name="ad", hidden=True)
@app.command(name="admm")
def admm(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径"),
    note: str = typer.Option("", "--note", "-n", help="本次重建备注")
):
    """[bold yellow](缩写: ad)[/bold yellow]   [cyan][算法重建][/cyan] ADMM 算法"""
    run_admm_logic(config, note)


@app.command(name="wn", hidden=True)
@app.command(name="wiener")
def wiener(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径"),
    note: str = typer.Option("", "--note", "-n", help="本次重建备注")
):
    """[bold yellow](缩写: wn)[/bold yellow]   [cyan][算法重建][/cyan] Wiener 滤波"""
    run_wiener_logic(config, note)

@app.command(name="screen_cal", hidden=True)
@app.command(name="screen_calibrate")
def screen_calibrate():
    """[bold yellow](缩写: screen_cal)[/bold yellow]   [cyan][辅助工具][/cyan] 屏幕标定：用于计算屏幕的物理尺寸（TODO）"""
    logger.info("屏幕标定功能尚在开发中.")


@app.command(name="mask_gen", hidden=True)
@app.command(name="mask_generate")
def mask_generate():
    """[bold yellow](缩写: mask_gen)[/bold yellow]   [cyan][数据生成][/cyan] 生成用于加载到 SLM 上的掩膜图案（TODO）"""
    logger.info("掩膜生成工具尚在开发中，目前没有集成在 CLI 中，你可以阅读 src/utils/gen_mask.py 和 src/utils/mask.py 两个文件并用 data/mask_patten_gen 里面的 Jupyter 文件进行掩膜生成.")

if __name__ == "__main__":
    app()