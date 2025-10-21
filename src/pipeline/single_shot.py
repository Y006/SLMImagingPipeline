import os
import time
import yaml
import threading
from src.hardware.screen import display_image
from src.hardware.slm import SLM
from src.hardware.camera import HikCamera
from src.hardware.MvImport.MvCameraControl_class import *
import time, threading, datetime
from src.utils.config_utils import (
    prepare_run_environment, task_codes_from_time,
    append_log, write_run_yaml, to_json_str, generate_file_prefix
)

def main():
    
    CONFIG_PATH = r"D:\qjy\SLMImagingPipeline\configs\single_shot_config.yaml"  # 可替换为实际路径
    config, run_data, run_path, proj_dir, kind = prepare_run_environment(CONFIG_PATH)

    if config["task"]["mode"] == "capture_psf":
        use_display = False
        use_slm     = True
        use_camera  = True
    if config["task"]["mode"] == "capture_measurement":
        use_display = True
        use_slm     = True
        use_camera  = True
    if config["task"]["mode"] == "calibration":
        use_display = True
        use_slm     = True
        use_camera  = False

    try:
    # 初始化设备
        if use_slm:    
            slm = SLM(verbose=True)
            settle_ms = 30
            slm.init()

        if use_camera:
            cam = HikCamera(dev_index=0)
            timeout_ms = 3000
            if not cam.open():
                print("[ERR] 相机打开失败")
                exit(2)
    
    # 生成任务 ID
        code4, task_id = task_codes_from_time(kind, k=4)

    # 运行设备
        if use_display:
            print("[INFO] 启动显示器显示线程")
            display_thread = threading.Thread(
                target=display_image,
                args=(
                    config["capture_settings"]["display_image_path"],
                    config["capture_settings"]["monitor_idx"],
                    config["capture_settings"]["scale_factor"]
                    # 目前这里的函数设置没有添加 config["capture_settings"]["display_position"]
                )
            )
            display_thread.daemon = True
            display_thread.start()

        if use_slm:
            if not os.path.exists(config["capture_settings"]["slm_image_path"]):
                print("[ERR] SLM图片路径无效")
                exit(4)
            slm.img_show(config["capture_settings"]["slm_image_path"])
            time.sleep(settle_ms / 1000.0)

        if config["task"]["mode"] == "calibration":
            print("[INFO] 进入标定模式，保持显示器和SLM显示，按 Ctrl+C 退出或等待1000秒后自动退出")
            print("[INFO] calibration 模式：不写 run.yaml 日志")
            time.sleep(10000)   # 保持足够时间用于手动对焦等操作
            return

        captured = None
        if use_camera and kind in ("psf", "m"):
            num_width = 3   # 保存图片的编号位数
            file_prefix = generate_file_prefix(kind, run_data, num_width)
            out_path = proj_dir / f"{file_prefix}-{code4}.jpg"
            ok = cam.snap(
                out_path,
                exposure_us=config["capture_settings"]["exposure_us"],
                timeout_ms=timeout_ms,
                img_type=MV_Image_Jpeg
            )
            if ok:
                captured = str(out_path)
                if config["task"]["mode"] == "capture_psf":
                    print(f"[OK] PSF 拍摄成功!")
                if config["task"]["mode"] == "capture_measurement":
                    print(f"[OK] Measurement 拍摄成功!")
            else:
                print("[ERR] 拍摄失败！")
    
    # 写入 run.yaml
        entry_key = f"task_{code4}"
        entry_val = {
            "task_id": task_id,
            "task_time": datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
            "mode": config["task"]["mode"],
            "description": config["task"].get("description", ""),
            "capture_settings": to_json_str(config["capture_settings"]),
            "physical_setup": to_json_str(config["physical_setup"])
        }
        if captured:
            if kind == "psf": entry_val["psf_path"] = captured
            else: entry_val["measurement_path"] = captured
        append_log(run_data, {entry_key: entry_val})
        write_run_yaml(run_path, run_data)
        print(f"[OK] wrote {run_path}")

    finally:
        if use_camera:
            cam.close()

if __name__ == "__main__":
    main()
