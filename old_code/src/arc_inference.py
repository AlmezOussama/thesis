from vllm import LLM, SamplingParams
from pathlib import Path
import torch
import os

def init_llm_dual_gpu(model_subdir="qwen3_4b_thinking"):
    # Locate model
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    print("Model directory found:", model_dir)

    # Check CUDA devices
    n_gpus = torch.cuda.device_count()
    print("Available CUDA devices:", n_gpus)
    for i in range(n_gpus):
        print(f"Device {i}:", torch.cuda.get_device_name(i))

    if n_gpus < 1:
        raise RuntimeError("No CUDA devices available.")

    # Let vLLM use all available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    try:
        llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=1,  
            gpu_memory_utilization=0.6,
            max_model_len=92000, 
        )
        print("LLM initialized successfully.")
    except Exception as e:
        print("Failed to initialize LLM:", e)
        raise

    sampling_params = SamplingParams(
        n=1,  # start small to test
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
    )

    return llm, sampling_params


if __name__ == "__main__":
    llm, sampling_params = init_llm_dual_gpu()
    
    # Simple test prompt
    prompt = "Hello, what is 2 + 2?"
    outputs = llm.generate([prompt], sampling_params)
    for sample in outputs[0].outputs:
        print("Model output:", sample.text.strip())
