import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from torch import nn
from torch.export import export
import torch.fx as fx
from graphviz import Digraph

from opeator_dag import Node, Edge, DAG

def export_and_draw_model(model: nn.Module, example_input: tuple, output_name: str = "model_dag", only_print=False):
    """
    Exports a PyTorch model to a DAG, saves it as JSON, and renders it as a PNG image.
    
    Args:
        model: The PyTorch model (nn.Module).
        example_input: A sample input tensor or tuple of tensors for the model.
        output_name: The base name for the output files (without extension).
                     Will generate {output_name}.json and {output_name}.png.
    """
    
    # 1. Export the model
    ep = export(model, example_input)
    gm: fx.GraphModule = ep.graph_module
    graph: fx.Graph = gm.graph

    # optionally, you can print the graph for debugging
    if only_print:
        print(gm)
        print(graph)
        return

    # 2. Build the DAG
    dag = DAG()

    for node in graph.nodes:
        tensorMetaData = node.meta.get('tensor_meta')
        shape = tensorMetaData.shape if tensorMetaData else None
        dtype = tensorMetaData.dtype if tensorMetaData else None

        if node.op == 'output':
            continue

        # for resnet18 skip BN counter self add_
        if str(dtype) == 'torch.int64' and shape is not None and len(shape) == 0:
            continue

        # Filter out compiler metadata and side-effect nodes
        target_str = str(node.target)
        ignore_patterns = [
            "lazy_load_decompositions",
            "_vmap_increment_nesting",
            "_vmap_decrement_nesting",
            "torch__dynamo__trace_wrapped_higher_order_op",
            "function_const",
            "_remove_batch_dim",
            "aten._assert_tensor_metadata"
        ]
        if any(pattern in target_str for pattern in ignore_patterns):
            continue

        # GPT-2 specific filtering: remove redundant mask generation and position_id logic
        if "gpt2" in output_name:
            gpt2_ignore_ops = [
                "aten.arange.default",  # Redundant position_ids generation
                "aten.add_.Tensor",     # Redundant position_ids generation
                "aten.new_ones",        # Mask generation
                "aten.le",              # Mask generation
                "aten.eq",              # Mask generation
                "aten.expand",          # Mask generation
                "aten.__and__"          # Mask generation
            ]
            if any(op in target_str for op in gpt2_ignore_ops):
                continue
            
            # Filter 'to' operators used in mask generation (usually bool or converting mask)
            # Valid 'to' in GPT2 usually converts float32 embeddings or logits
            if "aten.to" in target_str and str(dtype) == "torch.bool":
                continue

        if node.op == 'placeholder':
            # Usually 'x' or parameters
            if node.name == 'x' or node.name == 'src' or 'input' in node.name: 
                # Heuristic: treat 'x' or 'src' or explicit inputs as input nodes
                dag.addNode(Node(
                    id=node.name,
                    args=[],
                    kwargs={},
                    op="input",
                    shape=str(shape) if shape else None,
                    dtype=str(dtype) if dtype else None
                ))
                continue
            else:
                continue
        
        # Normal nodes
        dag.addNode(Node(
            id=node.name,
            type=node.op,
            args=node.args,
            kwargs=node.kwargs,
            op=f"torch.ops.{str(node.target)}",
            shape=str(shape) if shape else None,
            dtype=str(dtype) if dtype else None
        ))

    # 3. Build Edges
    for node in dag.nodes:
        # Helper function to recursively find nodes in args/kwargs
        def find_nodes(obj):
            if isinstance(obj, fx.Node):
                yield obj
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    yield from find_nodes(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    yield from find_nodes(item)

        # Check args
        for arg in node.args:
            for arg_node in find_nodes(arg):
                arg_name = arg_node.name
                for n in dag.nodes:
                    if n.id == arg_name:
                        edge = Edge(from_node=n, to_node=node)
                        dag.addEdge(edge)
        
        # Check kwargs
        for key, arg in node.kwargs.items():
            for arg_node in find_nodes(arg):
                arg_name = arg_node.name
                for n in dag.nodes:
                    if n.id == arg_name:
                        edge = Edge(from_node=n, to_node=node)
                        dag.addEdge(edge)

    # 4. Connect disconnected nodes to input (Original script logic)
    # Find the input node
    input_nodes = [n for n in dag.nodes if n.op == "input"]
    if input_nodes:
        input_node = input_nodes[0]
        for node in dag.nodes:
            # Check if node has incoming edges
            has_incoming = False
            for edge in dag.edges:
                if edge.to_node.id == node.id:
                    has_incoming = True
                    break
            
            if not has_incoming and node.type != 'placeholder' and node.op != "input":
                dag.addEdge(Edge(
                    from_node=input_node,
                    to_node=node
                ))

    # 5. Export to JSON
    graph_json = {
        "nodes": [
            {
                "id": n.id,
                "op": n.op,
                "type": n.type,
                "shape": n.shape,
                "dtype": n.dtype,
            }
            for n in dag.nodes
        ],
        "edges": [
            {
                "from": e.from_node.id,
                "to": e.to_node.id,
            }
            for e in dag.edges
        ],
    }

    json_filename = f"{output_name}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)
    print(f"Saved DAG JSON to {json_filename}")

    # 6. Render with Graphviz
    # Use sfdp engine for large graphs (faster layout for many nodes)
    dot = Digraph("OperatorDAG", format="svg", engine="sfdp")
    dot.attr("graph", overlap="false", splines="true")
    dot.attr("node", shape="box", fontsize="10")

    # Add nodes
    for n in dag.nodes:
        label_lines = [n.id]
        if n.op: label_lines.append(n.op)
        if n.shape: label_lines.append(str(n.shape))
        if n.dtype: label_lines.append(str(n.dtype))
        
        label = "\n".join(label_lines)
        dot.node(n.id, label=label)

    # Add edges
    for e in dag.edges:
        dot.edge(e.from_node.id, e.to_node.id)

    # Render
    out_path = dot.render(output_name, cleanup=True)
    print(f"Saved DAG Image to {out_path}")
