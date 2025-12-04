import ast


def ast_equal(node1, node2):
    # recursively compare two AST nodes for equality
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, ast.AST):
        for field in node1._fields:
            if not ast_equal(getattr(node1, field, None),
                             getattr(node2, field, None)):
                return False
        return True

    elif isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        return all(ast_equal(n1, n2) for n1, n2 in zip(node1, node2))

    else:
        return node1 == node2


def find_same_level_ifs(if_node):
    # if-elif-elif-else structure -> if-orelse, in orelse if-orelse, ...
    # so, we need to traverse orelse till it is not an If node
    # collect the conditions and bodies too

    branches = []

    cur_node = if_node
    while isinstance(cur_node, ast.If):
        branches.append((cur_node.test, cur_node.body))

        if cur_node.orelse and len(cur_node.orelse) == 1 and isinstance(cur_node.orelse[0], ast.If):
            cur_node = cur_node.orelse[0]
        else:
            if cur_node.orelse:
                branches.append((None, cur_node.orelse))
            break

    return branches
