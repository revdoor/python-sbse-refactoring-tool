import ast


_MISSING = object()


def ast_equal(target1, target2):
    # recursively compare two AST nodes for equality
    if type(target1) is not type(target2):
        return False

    if isinstance(target1, ast.AST):
        for field in target1._fields:
            if not ast_equal(getattr(target1, field, _MISSING),
                             getattr(target2, field, _MISSING)):
                return False
        return True

    elif isinstance(target1, list):
        if len(target1) != len(target2):
            return False
        return all(ast_equal(n1, n2) for n1, n2 in zip(target1, target2))

    else:
        return target1 == target2


def ast_similar(target1, target2):
    """
    Compare two AST nodes for structural similarity,
    ignoring specific identifier names.
    The identifier names should have a one-to-one mapping.
    """
    map_id1_to_id2 = dict()
    map_id2_to_id1 = dict()

    is_def_node = isinstance(target1, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    if is_def_node and hasattr(target1, 'name') and hasattr(target2, 'name'):
        map_id1_to_id2[target1.name] = target2.name
        map_id2_to_id1[target2.name] = target1.name

    def _map_ids(id1, id2):
        existing_id2 = map_id1_to_id2.get(id1, _MISSING)
        existing_id1 = map_id2_to_id1.get(id2, _MISSING)

        if existing_id2 is not _MISSING or existing_id1 is not _MISSING:
            return existing_id2 == id2 and existing_id1 == id1

        map_id1_to_id2[id1] = id2
        map_id2_to_id1[id2] = id1
        return True

    def _ast_similar(node1, node2, parent1=None, parent2=None, field_name=None):
        if type(node1) is not type(node2):
            return False

        if isinstance(node1, ast.AST):
            is_call_func = (isinstance(parent1, ast.Call) and field_name == 'func')

            for field in node1._fields:
                if field == 'id' or field == 'arg':
                    id1 = getattr(node1, field, _MISSING)
                    id2 = getattr(node2, field, _MISSING)

                    if is_call_func and field == 'id':
                        # function call
                        if is_def_node:
                            if id1 == target1.name and id2 == target2.name:
                                continue
                            elif id1 == target1.name or id2 == target2.name:
                                return False
                        if id1 != id2:
                            return False
                    if not _map_ids(id1, id2):
                        return False
                elif is_def_node and field == 'name':
                    name1 = getattr(node1, field, _MISSING)
                    name2 = getattr(node2, field, _MISSING)

                    if name1 == target1.name and name2 == target2.name:
                        continue
                    elif name1 == target1.name or name2 == target2.name:
                        return False
                    if name1 != name2:
                        return False
                else:
                    if not _ast_similar(getattr(node1, field, _MISSING),
                                        getattr(node2, field, _MISSING),
                                        parent1=node1, parent2=node2, field_name=field):
                        return False
            return True

        elif isinstance(node1, list):
            if len(node1) != len(node2):
                return False
            return all(_ast_similar(n1, n2, parent1, parent2, field_name) for n1, n2 in zip(node1, node2))

        else:
            return node1 == node2

    result = _ast_similar(target1, target2)

    return result


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


def _is_recursive(func_node: ast.FunctionDef) -> bool:
    func_name = func_node.name
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == func_name:
                return True
    return False


def create_codes_from_stmts(stmts):
    return "\n".join(ast.unparse(stmts))


def create_return_nodes_from_assign_or_augassign(node):
    assert isinstance(node, ast.Assign) or isinstance(node, ast.AugAssign)
    return ast.Return(value=node.value)
