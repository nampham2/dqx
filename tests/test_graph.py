from dqx import graph
from rich.console import Console

def walker(node: graph.Node) -> None:
    print(node.inspect_str())

def test_graph() -> None:
    root = graph.RootNode("Some checks")
    root.add_child(check_1:= graph.CheckNode("check_1"))
    check_1.add_child(graph.AssertionNode("assertion_1"))


    print("\n")
    tree = root.inspect()
    Console().print(tree)
    