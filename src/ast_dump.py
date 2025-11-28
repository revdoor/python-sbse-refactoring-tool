import ast
from type_enums import RefactoringOperatorType


def dump_ast(source_code):
    root = ast.parse(source_code)
    return ast.dump(root, indent=4)


if __name__ == '__main__':
    for operator in RefactoringOperatorType:
        print(f"Dump AST for {operator.value}...")
        with open(f'dump_target_code/dump_target_code_{operator.name.lower()}.py', 'r') as f:
            source_code = f.read()

        ast_representation = dump_ast(source_code)

        with open(f'ast_output/ast_output_{operator.name.lower()}.txt', 'w') as f:
            f.write(ast_representation)
