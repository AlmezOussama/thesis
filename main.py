import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText



from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    save_dir = Path("models/qwen_8b")
    save_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Thinking-FP8",
        local_dir=str(save_dir),
        local_dir_use_symlinks=False,
    )

if __name__ == "__main__":
    main()

