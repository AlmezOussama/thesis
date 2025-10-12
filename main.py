import pandas as pd
import numpy as np
import json


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

save_directory = "models/qwen3-4b"

# Save both model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

'data/arc-agi_evaluation_challenges.json'
'data/arc-agi_evaluation_solutions.json'
'data/arc-agi_test_challenges.json'


 # current_file = Path(__file__).resolve()
    # project_root = current_file.parent.parent
    # model_dir = project_root / "models" / "qwen4b_awq"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

    # llm = LLM(
    #     model=str(model_dir),
    #     tensor_parallel_size=1,  # number of GPUs used
    #     gpu_memory_utilization=0.9
    # )

    # sampling_params = SamplingParams(
    #     n = 2,
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=512
    # )

    # prompt = "what do you know about the arc challenge?"

    # outputs = llm.generate(prompt, sampling_params)

    # return outputs