import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model_path = "/data/datasets/models-hf/Llama-3.1-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to(device) # type: ignore
model.eval()

# 构造一个简单 prompt
prompt = "Hello, this is a Llama3.1 profiling test."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# 关键修复：预热以触发 torch.compile 编译
# 如果不预热，Profiler 会记录整个编译过程，导致数据量爆炸引发 MemoryError
# print("Warming up and compiling... (This may take a while)")
# with torch.no_grad():
#     # 跑一次以触发编译
#     _ = model(input_ids, use_cache=True)

# 配置 profiler
use_cache = True

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,

) as prof:
    with torch.no_grad():
        # 1) prefill 阶段
        with record_function("prefill"):
            out = model(input_ids, use_cache=use_cache)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 2) decode 阶段，循环若干步
        num_new_tokens = 8
        for step in range(num_new_tokens):
            with record_function(f"decode_step_{step}"):
                out = model(next_token, use_cache=use_cache, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

torch.cuda.synchronize()

# # 打印算子级 summary
# print(prof.key_averages().table(
#     sort_by="self_cuda_time_total",
#     row_limit=30
# ))

# 导出 Chrome trace，后续你用脚本过滤出 op <-> kernel 映射信息
prof.export_chrome_trace("trace.json")
