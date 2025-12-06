import torch
import torchvision.models as models

from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18(weights=None).eval().cuda()
example_x = torch.randn(1, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,          # 记录 tensor shape
    with_stack=True,             # 记录 Python 调用栈（开了会贵些）
    profile_memory=True         # 记录内存使用（选用）
) as prof:
    with torch.no_grad():
        with record_function("forward_step"):
            out = model(example_x)

torch.cuda.synchronize()
prof.export_chrome_trace("trace.json")
