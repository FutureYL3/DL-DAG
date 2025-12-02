import json
import torch
from torch import nn
from torch.export import export  # PyTorch 2.x
from torch.fx.passes.shape_prop import ShapeProp
import torch.fx as fx

from opeator_dag import Node, Edge, DAG

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = MyModel().eval()

# 1. 准备一个“示例输入”
example_x = torch.randn(1, 784)

# 2. 导出计算图
ep = export(model, (example_x,))  # ExportedProgram

gm: fx.GraphModule = ep.graph_module
graph: fx.Graph = gm.graph

Graph = DAG()

for node in graph.nodes:
    if node.op == 'output' or node.op == 'placeholder':
        continue

    tensorMetaData = node.meta.get('tensor_meta')
    shape = tensorMetaData.shape if tensorMetaData else None
    dtype = tensorMetaData.dtype if tensorMetaData else None

    Graph.addNode(Node(
        id=node.name,
        type=node.op,
        args=node.args,
        kwargs=node.kwargs,
        op=f"torch.ops.{str(node.target)}",
        shape=str(shape) if shape else None,
        dtype=str(dtype) if dtype else None
    ))

for node in Graph.nodes:
    for arg in node.args:
        for n in Graph.nodes:
            if n.id == arg.name:
                edge = Edge(from_node=n, to_node=node)
                Graph.addEdge(edge)

# ===== 3. 导出为 JSON =====
graph_json = {
    "nodes": [
        {
            "id": n.id,
            "op": n.op,
            "type": n.type,
            "shape": n.shape,
            "dtype": n.dtype,
        }
        for n in Graph.nodes
    ],
    "edges": [
        {
            "from": e.from_node.op,
            "to": e.to_node.op,
        }
        for e in Graph.edges
    ],
}


with open("graph.json", "w", encoding="utf-8") as f:
    json.dump(graph_json, f, indent=2, ensure_ascii=False)

print("Saved DAG to graph.json")
