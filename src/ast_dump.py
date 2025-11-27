import ast


def dump_ast(source_code):
    root = ast.parse(source_code)
    return ast.dump(root, indent=4)


if __name__ == '__main__':
    file_name = 'dump_target_code.py'

    with open(file_name, 'r') as f:
        source_code = f.read()

    ast_representation = dump_ast(source_code)

    with open('ast_output.txt', 'w') as f:
        f.write(ast_representation)
