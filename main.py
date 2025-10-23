import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    save_dir = "models/qwen3_4b_thinking"

    # Save both
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

