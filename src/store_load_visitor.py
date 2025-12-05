import ast


class StoreLoadVisitor(ast.NodeVisitor):
    def __init__(self, ignore_nodes=None):
        self.store_ids = set()
        self.load_ids = set()
        self.ignore_nodes = ignore_nodes

    def visit_Name(self, node):
        if self.ignore_nodes and node in self.ignore_nodes:
            return

        if isinstance(node.ctx, ast.Load):
            self.load_ids.add(node.id)
        else:
            self.store_ids.add(node.id)

    def generic_visit(self, node):
        if self.ignore_nodes and node in self.ignore_nodes:
            return

        super().generic_visit(node)
