import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.models as models
from export_and_draw import export_and_draw_model

model = models.resnet18(weights=None)   # 或 weights=models.ResNet18_Weights.DEFAULT
example_x = torch.randn(1, 3, 224, 224)

export_and_draw_model(model, (example_x,), "resnet18_dag")
