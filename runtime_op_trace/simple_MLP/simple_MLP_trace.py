import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # 逻辑算子 1：fc1
        with record_function("LAYER_fc1_linear"):
            x = self.fc1(x)

        # 逻辑算子 2：relu
        with record_function("LAYER_relu"):
            x = torch.relu(x)

        # 逻辑算子 3：fc2
        with record_function("LAYER_fc2_linear"):
            x = self.fc2(x)

        return x

model = MyModel().eval().cuda()

inputs = torch.randn(1, 784, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,          # 记录 tensor shape
    with_stack=True,             # 记录 Python 调用栈（开了会贵些）
    profile_memory=True         # 记录内存使用（选用）
) as prof:
    with torch.no_grad():
        with record_function("forward_step"):
            out = model(inputs)

torch.cuda.synchronize()

prof.export_chrome_trace("trace.json")
