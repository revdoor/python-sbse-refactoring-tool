"""
This module defines the CandidateGenerator class,
which is used to find the possible RefactoringOperator instances
for the given source code.
"""

import ast
from refactoring_operator import InlineMethodOperator


class OrderVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.type_order = {}
        self.node_order = {}

    def generic_visit(self, node):
        typ = type(node).__name__

        if typ not in self.type_order:
            self.type_order[typ] = 1
        else:
            self.type_order[typ] += 1

        self.node_order[node] = self.type_order[typ]

        super().generic_visit(node)


class CandidateGenerator:
    def __init__(self):
        pass

    def generate_candidates(self, source_code):
        # TODO: Implement candidate generation logic
        # It should find various valid refactoring operators applicable to the source code
        pass

    def generate_inline_method_candidates(self, source_code):
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        order = order_visitor.node_order

        candidates = []

        for node in root.body:
            if isinstance(node, ast.FunctionDef):
                # consider functions with single statement only
                # to handle simple inline method refactoring
                if len(node.body) == 1:
                    no = order[node]
                    candidates.append(InlineMethodOperator(no))

        return candidates
