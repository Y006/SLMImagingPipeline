# ====== config_utils.py ======
from pathlib import Path
import yaml, json, hashlib, datetime, os

BASE62_ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# -----------------------------
# 通用工具函数
# -----------------------------
def base62(n:int)->str:
    if n==0: return BASE62_ALPH[0]
    s=[]; b=len(BASE62_ALPH)
    while n: n,r=divmod(n,b); s.append(BASE62_ALPH[r])
    return "".join(reversed(s))

def short_b62_from_obj(obj,k=3)->str:
    h=hashlib.sha1(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).digest()
    return base62(int.from_bytes(h[:6],"big"))[:k]

def to_json_str(obj)->str:
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"))

def task_codes_from_time(kind:str, k=4):
    """生成时间相关任务 ID"""
    task_time_iso = datetime.datetime.now().astimezone().isoformat(timespec='microseconds')
    payload = {"kind": kind, "task_time": task_time_iso}
    code = short_b62_from_obj(payload, k=k)
    task_id = f"{kind}-{code}"
    return code, task_id

def generate_file_prefix(kind: str, run_data: dict, num_width: int = 5) -> str:
    """根据已有日志生成编号前缀，例如 psf00001、m00012"""
    existing_count = len(run_data.get("save_log", []))
    file_index = existing_count + 1
    return f"{file_index:0{num_width}d}-{kind}"

# -----------------------------
# 读写 YAML 文件
# -----------------------------
def load_config(path)->dict:
    p=Path(path)
    if not p.is_absolute():
        p=Path(__file__).resolve().parent/p
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_run_yaml(p:Path)->dict:
    if not p.exists(): return {}
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_run_yaml(p:Path,d:dict):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        yaml.safe_dump(d,f,allow_unicode=True,sort_keys=False)

def append_log(run_data:dict,entry:dict):
    run_data.setdefault("save_log",[])
    run_data["save_log"].append({"idx":len(run_data["save_log"])+1,**entry})

# -----------------------------
# 项目与运行配置加载
# -----------------------------
def prepare_run_environment(path):
    """加载 config.yaml 并准备项目文件夹、run.yaml"""
    cfg = load_config(path)
    mode = cfg["task"]["mode"]
    if mode=="capture_psf": kind="psf"
    elif mode=="capture_measurement": kind="m"
    elif mode=="calibration": kind="calibration"
    else: raise ValueError(f"Unknown mode {mode}")

    proj_id = cfg["project"]["id"]
    root = Path(cfg["project"]["root_dir"])
    proj_dir = root / proj_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    run_path = proj_dir / "run.yaml"

    run_data = read_run_yaml(run_path)
    if "project" not in run_data:
        run_data["project"] = {
            "id": proj_id,
            "root_dir": str(cfg["project"]["root_dir"]),
            "description": cfg["project"].get("description","")
        }

    return cfg, run_data, run_path, proj_dir, kind
