import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from export_and_draw import export_and_draw_model

d_model = 64
nhead = 8
dim_feedforward = 256
num_layers = 2

encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    batch_first=True,
)
model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
model = model.eval()

# 假设序列长度 16，batch=1
example_x = torch.randn(1, 16, d_model)

export_and_draw_model(model, (example_x,), "simple_transformer")
