import ast


def dump_ast(source_code):
    root = ast.parse(source_code)
    return ast.dump(root, indent=4)


if __name__ == '__main__':
    for operator in ['im', 'dc']:
        with open(f'dump_target_code/dump_target_code_{operator}.py', 'r') as f:
            source_code = f.read()

        ast_representation = dump_ast(source_code)

        with open(f'ast_output/ast_output_{operator}.txt', 'w') as f:
            f.write(ast_representation)
