"""
This module defines the DependencyChecker class,
which is used to check the dependency for the nodes between the surroundings.
"""
import ast
import random
from pathlib import Path
from type_enums import RefactoringOperatorType


class DependencyVisitor(ast.NodeVisitor):
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


class DependencyChecker:
    @staticmethod
    def is_dependency_free(top_lvl_node, parent_node, attr_name, idx, length):
        if not hasattr(parent_node, attr_name):
            return False

        attr = getattr(parent_node, attr_name)
        if len(attr) < idx + length:
            return False

        target_nodes = attr[idx: idx + length]

        target_visitor = DependencyVisitor()
        for target_node in target_nodes:
            target_visitor.visit(target_node)

        target_store_ids = target_visitor.store_ids
        target_load_ids = target_visitor.load_ids

        outer_visitor = DependencyVisitor(target_nodes)
        outer_visitor.visit(top_lvl_node)

        outer_store_ids = outer_visitor.store_ids
        outer_load_ids = outer_visitor.load_ids

        print("Target store IDs:", target_store_ids)
        print("Target load IDs:", target_load_ids)
        print("Outer store IDs:", outer_store_ids)
        print("Outer load IDs:", outer_load_ids)

    @staticmethod
    def _dependency(node):
        visitor = DependencyVisitor()

        visitor.visit(node)

        return visitor.store_ids, visitor.load_ids


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    for operator in RefactoringOperatorType:
        print(f"Dependency check for {operator.value}...")

        with open(script_dir / f'dump_target_code/dump_target_code_{operator.name.lower()}.py', 'r') as f:
            source_code = f.read()

        root = ast.parse(source_code)

        for _node in ast.walk(root):
            if isinstance(_node, ast.FunctionDef):
                body = _node.body

                i, j = random.choices(range(len(body)), k=2)
                if i > j:
                    i, j = j, i

                print(f"Function {_node.name}, from statement {i} to {j}:")
                DependencyChecker.is_dependency_free(_node, _node, 'body', i, j - i + 1)
                print()
