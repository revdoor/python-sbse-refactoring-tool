"""
This module defines the DependencyChecker class,
which is used to check the dependency for the nodes between the surroundings.
"""
import ast
from pathlib import Path
from type_enums import RefactoringOperatorType


class DependencyChecker:
    @staticmethod
    def check_dependency(top_lvl_node, start_node, length):
        pass

    @staticmethod
    def dependency_name(node):
        store_ids = set()
        load_ids = set()

        for _node in ast.walk(node):
            if isinstance(_node, ast.Name):
                _id = _node.id

                if isinstance(_node.ctx, ast.Load):
                    load_ids.add(_id)
                else:
                    store_ids.add(_id)

        print(store_ids)
        print(load_ids)


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    for operator in RefactoringOperatorType:
        print(f"Dependency check for {operator.value}...")

        with open(script_dir / f'dump_target_code/dump_target_code_{operator.name.lower()}.py', 'r') as f:
            source_code = f.read()

        root = ast.parse(source_code)

        for _node in ast.walk(root):
            if isinstance(_node, ast.FunctionDef):
                print(f"Function {_node.name}")
                DependencyChecker.dependency_name(_node)
