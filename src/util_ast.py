import ast

from util import get_random_name


_MISSING = object()


def ast_equal(node1, node2):
    # recursively compare two AST nodes for equality
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, ast.AST):
        for field in node1._fields:
            if not ast_equal(getattr(node1, field, _MISSING),
                             getattr(node2, field, _MISSING)):
                return False
        return True

    elif isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        return all(ast_equal(n1, n2) for n1, n2 in zip(node1, node2))

    else:
        return node1 == node2


def ast_similar(node1, node2):
    """
    Compare two AST nodes for structural similarity,
    ignoring specific identifier names.
    The identifier names should have a one-to-one mapping.
    """
    map_id1_to_id2 = dict()
    map_id2_to_id1 = dict()

    # if the nodes have names (e.g., function definitions),
    # temporarily rename both nodes to a common name
    # to ignore the name difference during comparison
    if hasattr(node1, 'name') and hasattr(node2, 'name'):
        orig_name1 = node1.name
        orig_name2 = node2.name

        temp_name = get_random_name()

        node1.name = temp_name
        node2.name = temp_name

        name_changed = True
    else:
        orig_name1 = None
        orig_name2 = None

        name_changed = False

    def _map_ids(id1, id2):
        existing_id2 = map_id1_to_id2.get(id1, _MISSING)
        existing_id1 = map_id2_to_id1.get(id2, _MISSING)

        if existing_id2 is not _MISSING or existing_id1 is not _MISSING:
            return existing_id2 == id2 and existing_id1 == id1

        map_id1_to_id2[id1] = id2
        map_id2_to_id1[id2] = id1
        return True

    def _ast_similar(node1, node2):
        if type(node1) is not type(node2):
            return False

        if isinstance(node1, ast.AST):
            for field in node1._fields:
                if field == 'id' or field == 'arg':
                    id1 = getattr(node1, field, _MISSING)
                    id2 = getattr(node2, field, _MISSING)

                    if not _map_ids(id1, id2):
                        return False
                else:
                    if not _ast_similar(getattr(node1, field, _MISSING),
                                        getattr(node2, field, _MISSING)):
                        return False
            return True

        elif isinstance(node1, list):
            if len(node1) != len(node2):
                return False
            return all(_ast_similar(n1, n2) for n1, n2 in zip(node1, node2))

        else:
            return node1 == node2

    result = _ast_similar(node1, node2)

    if name_changed:
        node1.name = orig_name1
        node2.name = orig_name2

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
