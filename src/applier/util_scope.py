import ast
from typing import Optional
from type_enums import NodeType


def get_attr_name_from_node_type(node_type: NodeType) -> str:
    """Get the attribute name ('body' or 'orelse') from node type."""
    return 'orelse' if node_type is NodeType.IfElse else 'body'


def find_enclosing_class(root: ast.Module, target_node: ast.AST) -> Optional[ast.ClassDef]:
    """Find the ClassDef that contains the target node, if any."""
    for node in ast.walk(root):
        if isinstance(node, ast.ClassDef):
            for descendant in ast.walk(node):
                if descendant is target_node:
                    return node
    return None


def find_enclosing_function(root: ast.Module, target_node: ast.AST) -> Optional[ast.FunctionDef]:
    """Find the FunctionDef that contains the target node."""
    if isinstance(target_node, ast.FunctionDef):
        return target_node

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            for item in ast.walk(node):
                if item is target_node:
                    return node
    return None


def find_direct_enclosing_method(
        enclosing_class: ast.ClassDef,
        target_node: ast.AST
) -> Optional[ast.FunctionDef]:
    """Find the method in enclosing_class that directly contains target_node."""
    for item in enclosing_class.body:
        if isinstance(item, ast.FunctionDef):
            for n in ast.walk(item):
                if n is target_node:
                    return item
    return None


def check_name_conflict(
        root: ast.Module,
        new_name: str,
        enclosing_class: Optional[ast.ClassDef],
        exclude_node: Optional[ast.AST] = None
) -> bool:
    """Check if a function/method name already exists in the given scope."""
    scope = enclosing_class.body if enclosing_class else root.body

    for node in scope:
        if node is exclude_node:
            continue
        if isinstance(node, ast.FunctionDef) and node.name == new_name:
            return True
    return False


def remove_function_from_scope(
        root: ast.Module,
        node: ast.FunctionDef,
        enclosing_class: Optional[ast.ClassDef]
) -> None:
    """Remove a function/method from its enclosing scope."""
    scope = enclosing_class.body if enclosing_class else root.body
    scope.remove(node)


def insert_function_to_scope(
        root: ast.Module,
        new_func: ast.FunctionDef,
        enclosing_function: Optional[ast.FunctionDef],
        enclosing_class: Optional[ast.ClassDef],
        target_node: ast.AST,
        uses_self: bool
) -> None:
    """
    Insert a new function/method into the appropriate scope.

    Args:
        root: The module AST
        new_func: The new function to insert
        target_node: The original function (insert position reference)
        enclosing_class: The class containing target_node, or None for module level
        uses_self: Whether the new function uses 'self'
    """
    if enclosing_class and uses_self:
        # Insert as a method in the class
        direct_method = find_direct_enclosing_method(enclosing_class, target_node)
        if direct_method is not None:
            idx = enclosing_class.body.index(direct_method)
            enclosing_class.body.insert(idx, new_func)
        else:
            enclosing_class.body.insert(0, new_func)
    else:
        # Insert at module level
        if enclosing_class is not None and enclosing_class in root.body:
            idx = root.body.index(enclosing_class)
        elif enclosing_function is not None and enclosing_function in root.body:
            idx = root.body.index(enclosing_function)
        else:
            idx = 0
        root.body.insert(idx, new_func)
