import ast
from typing import Optional
from helpers.store_load_visitor import StoreLoadVisitor


class RefactoringHelper:
    """Common utilities for refactoring operations."""

    # --- Self Detection ---

    @staticmethod
    def uses_self_in_stmts(nodes: list[ast.stmt]) -> bool:
        return any(RefactoringHelper.uses_self_in_node(node) for node in nodes)

    @staticmethod
    def uses_self_in_node(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == 'self':
                return True
            if (isinstance(n, ast.Attribute) and
                    isinstance(n.value, ast.Name) and
                    n.value.id == 'self'):
                return True
        return False

    # --- Variable Analysis ---

    @staticmethod
    def get_used_variables_in_stmts(nodes: list[ast.stmt]) -> tuple[set[str], set[str]]:
        """Get (loaded, stored) variable names in statements."""
        visitor = StoreLoadVisitor()
        for node in nodes:
            visitor.visit(node)
        return visitor.load_ids, visitor.store_ids

    @staticmethod
    def get_used_variables_in_expr(node: ast.expr) -> set[str]:
        """Get variable names loaded in an expression."""
        return {
            n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    @staticmethod
    def get_function_params(func: ast.FunctionDef) -> set[str]:
        """Get all parameter names from a function definition."""
        params = set()
        for arg in func.args.args + func.args.posonlyargs + func.args.kwonlyargs:
            if arg.arg != 'self':
                params.add(arg.arg)
        if func.args.vararg:
            params.add(func.args.vararg.arg)
        if func.args.kwarg:
            params.add(func.args.kwarg.arg)
        return params

    @staticmethod
    def get_required_params(
            target_nodes: list[ast.stmt],
            prev_nodes: list[ast.stmt],
            enclosing_function: Optional[ast.FunctionDef]
    ) -> list[str]:
        """Get variables that need to be passed as parameters."""
        target_load, target_store = RefactoringHelper.get_used_variables_in_stmts(target_nodes)
        _, prev_store = RefactoringHelper.get_used_variables_in_stmts(prev_nodes)

        func_params = set()
        if enclosing_function:
            func_params = RefactoringHelper.get_function_params(enclosing_function)

        external_deps = target_load - target_store
        params = external_deps & (prev_store | func_params)

        return sorted(params)

    @staticmethod
    def get_prev_stmts(
            target_node: ast.AST,
            attr_name: str,
            start_idx: int,
            enclosing_function: Optional[ast.FunctionDef]
    ) -> list[ast.stmt]:
        """Get statements before the extraction target."""
        prev_stmts = list(getattr(target_node, attr_name)[:start_idx])

        if enclosing_function and target_node is not enclosing_function:
            for stmt in enclosing_function.body:
                if stmt is target_node:
                    break
                prev_stmts.append(stmt)
                if any(child is target_node for child in ast.walk(stmt)):
                    break

        return prev_stmts

    # --- AST Node Creation ---

    @staticmethod
    def create_function(
            name: str,
            params: list[str],
            body: list[ast.stmt],
            include_self: bool
    ) -> ast.FunctionDef:
        """Create a new function definition."""
        args_list = [ast.arg(arg=p) for p in params]
        if include_self:
            args_list.insert(0, ast.arg(arg='self'))

        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=args_list,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=body,
            decorator_list=[],
            returns=None
        )

    @staticmethod
    def create_call_expr(func_name: str, params: list[str], use_self: bool) -> ast.Call:
        """Create a function call expression."""
        call_args: list = [ast.Name(id=p, ctx=ast.Load()) for p in params]

        if use_self:
            func = ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=func_name,
                ctx=ast.Load()
            )
        else:
            func = ast.Name(id=func_name, ctx=ast.Load())

        return ast.Call(func=func, args=call_args, keywords=[])
