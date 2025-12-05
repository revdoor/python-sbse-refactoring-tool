import ast
from pathlib import Path
from type_enums import RefactoringOperatorType


def dump_ast(source_code):
    root = ast.parse(source_code)
    return ast.dump(root, indent=4)


if __name__ == '__main__':
    script_dir = Path(__file__).parent.resolve()

    for operator in RefactoringOperatorType:
        print(f"Dump AST for {operator.value}...")

        file_path = script_dir / f'dump_target_code/dump_target_code_{operator.name.lower()}.py'

        if not file_path.exists():
            print(f"!!!Dump code for {operator.value} does not exist. Skipping!!!")
            continue

        with open(file_path, 'r') as f:
            source_code = f.read()

        ast_representation = dump_ast(source_code)

        with open(script_dir / f'ast_output/ast_output_{operator.name.lower()}.txt', 'w') as f:
            f.write(ast_representation)
