# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

save_dir = "models/llama3.2_3B"

# Save model
model.save_pretrained(save_dir)

# Save tokenizer
tokenizer.save_pretrained(save_dir)