class Node:
    def __init__(self, id, type=None, args=None, kwargs=None, op=None, shape=None, dtype=None):
        self.id = id
        self.type = type
        self.args = args
        self.kwargs = kwargs
        self.op = op
        self.shape = shape
        self.dtype = dtype

class Edge:
    def __init__(self, from_node: Node, to_node: Node):
        self.from_node = from_node
        self.to_node = to_node


class DAG:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def addNode(self, node: Node):
        self.nodes.append(node)

    def addEdge(self, edge: Edge):
        self.edges.append(edge)

