# ==========================
# min_hf_upload_check.py
import os, json, sys, uuid
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# 强制使用官方端点；禁用交互式 git 提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 创建仓库必须用官方站
os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")

# 读取 token
with open("tokens.json", "r") as f:
    token = json.load(f)["huggingface"]

# 目标命名空间（确保你对其有写权限）
ns = os.getenv("HUGGINGFACENAME", "SmartDazi")
repo_name = f"tmp_push_check_{uuid.uuid4().hex[:8]}"
repo_id = f"{ns}/{repo_name}"

api = HfApi(endpoint=os.environ["HF_ENDPOINT"])

# 1) 权限快速检查
me = api.whoami(token=token)
orgs = {o["name"] for o in me.get("orgs", [])}
if ns not in {me["name"], *orgs}:
    print(f"ERROR: 无 {ns} 命名空间写权限")
    sys.exit(3)

# 2) 确保 repo 存在（dataset 类型）
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

# 3) 构造最小数据并 push
ds = Dataset.from_list([{"problem": "1+1=?", "answer": "2", "score": 1.0}])
dd = DatasetDict({"train": ds})
print(repo_id)
dd.push_to_hub(repo_id, private=True, config_name="check", token=token, commit_message="minimal push check")

print("OK: pushed to", f"https://hf-mirror.com/{repo_id}")

