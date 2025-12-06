import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from transformers import BertModel, BertConfig

from export_and_draw import export_and_draw_model


class BertTinyWrapper(nn.Module):
    """
    只包一层，把 HuggingFace BertModel 的输出简化成一个 Tensor:
    return last_hidden_state: [batch, seq_len, hidden]
    """
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids):
        # 这里只传 input_ids，attention_mask / token_type_ids 默认会在内部自动生成
        # 你也可以自己显式传进来，这里先保持简单
        outputs = self.bert(input_ids=input_ids)
        # outputs.last_hidden_state: [batch, seq_len, hidden]
        return outputs.last_hidden_state


def main():
    # # 1. 选择设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    # 2. 构造一个小 BERT 配置
    config = BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        # 其他保持默认即可，比如 vocab_size=30522
    )

    # 3. 建模 + eval 模式（关掉 dropout）
    model = BertTinyWrapper(config)
    # model = BertTinyWrapper(config).to(device)
    model.eval()

    # 4. 构造一个示例输入
    #
    # batch_size = 1, seq_len = 16
    # 用随机的 token id 即可，不需要真 tokenizer
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 16),
        # device=device,
        dtype=torch.long,
    )

    # 5. 调用你的导出 + 画图函数
    export_and_draw_model(
        model,
        (input_ids,),
        "bert_tiny_dag",
    )


if __name__ == "__main__":
    main()
