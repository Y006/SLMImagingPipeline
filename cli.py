"""
SLM 成像系统 - 命令行入口

所有业务逻辑已迁移到 src/cli/ 模块，本文件只负责命令声明和参数传递
"""

import typer
from src.cli import (
    run_psf_logic,
    run_measurement_logic,
    run_calibration_logic,
    inspect_png_meta,
    run_admm_logic,
    run_wiener_logic
)

DEFAULT_CONFIG_PATH = "configs/config.yaml"

app = typer.Typer(help="SLM 成像控制终端")


# ==============================================================================
# 数据采集命令
# ==============================================================================

@app.command()
def capture_psf(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """[数据采集] 采集 PSF"""
    run_psf_logic(config, repeat, note)


@app.command()
def capture_measurement(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    repeat: int = typer.Option(1, "--repeat", "-r"),
    note: str = typer.Option("", "--note", "-n")
):
    """[数据采集] 采集测量数据"""
    run_measurement_logic(config, repeat, note)


@app.command()
def calibrate(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")
):
    """[标定] 标定/预览模式"""
    run_calibration_logic(config)


# ==============================================================================
# 工具命令
# ==============================================================================

@app.command()
def inspect(
    filepath: str = typer.Argument(..., help="要检查的 PNG 图片路径")
):
    """[工具] 查看 PNG 图片中嵌入的实验元数据"""
    inspect_png_meta(filepath)


# ==============================================================================
# 算法命令
# ==============================================================================

@app.command()
def admm(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径"),
    note: str = typer.Option("", "--note", "-n", help="本次重建备注")
):
    """[算法] ADMM 重建 (完全基于 YAML 配置)"""
    run_admm_logic(config, note)


@app.command()
def wiener(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径"),
    note: str = typer.Option("", "--note", "-n", help="本次重建备注")
):
    """[算法] Wiener 滤波重建 (完全基于 YAML 配置)"""
    run_wiener_logic(config, note)


if __name__ == "__main__":
    app()
