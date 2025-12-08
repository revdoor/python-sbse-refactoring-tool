import ast
from typing import Optional

NODE_TYPE_TO_AST_NAME = {
    "FunctionDef": "FunctionDef",
    "If": "If",
    "IfElse": "If",
    "While": "While",
    "For": "For",
}


class TargetNodeFinder(ast.NodeVisitor):
    """Find a target node by type and sequential number."""

    def __init__(self, target_node_type, target_node_no: int):
        super().__init__()
        self.target_node_type = target_node_type
        self.target_node_no = target_node_no
        self.type_order: dict[str, int] = {}
        self.found_node: Optional[ast.AST] = None

    def generic_visit(self, node):
        if self.found_node:
            return

        typ = type(node).__name__
        expected = NODE_TYPE_TO_AST_NAME.get(
            self.target_node_type.value,
            self.target_node_type.value
        )

        self.type_order[typ] = self.type_order.get(typ, 0) + 1

        if typ == expected and self.type_order[typ] == self.target_node_no:
            self.found_node = node
            return

        super().generic_visit(node)


def find_target_node(
        root: ast.Module,
        target_node_type,
        target_node_no: int
) -> Optional[ast.AST]:
    """Find target node using NodeVisitor (same traversal as node_order generation)."""
    finder = TargetNodeFinder(target_node_type, target_node_no)
    finder.visit(root)
    return finder.found_node
