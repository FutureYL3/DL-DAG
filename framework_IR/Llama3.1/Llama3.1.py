import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from export_and_draw import export_and_draw_model


model_path = "/data/datasets/models-hf/Llama-3.1-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.config.use_cache = False
model = model
model.eval()

# 构造一个简单 prompt
prompt = "Hello, this is a Llama3.1 profiling test."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

export_and_draw_model(model, (input_ids,), "Llama3.1_dag")





