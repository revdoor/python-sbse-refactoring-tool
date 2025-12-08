import ast
import copy
from typing import Optional


class ArgumentReplacer(ast.NodeTransformer):
    """Replace parameter names with actual arguments."""

    def __init__(self, param_to_arg: dict[str, ast.expr]):
        self.param_to_arg = param_to_arg

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.param_to_arg:
            return copy.deepcopy(self.param_to_arg[node.id])
        return node


class InlineMethodTransformer(ast.NodeTransformer):
    """Replace function calls with inlined expressions."""

    def __init__(
            self,
            func_name: str,
            func_args: ast.arguments,
            inline_expr: ast.expr,
            is_return: bool
    ):
        self.func_name = func_name
        self.func_args = func_args
        self.inline_expr = inline_expr
        self.is_return = is_return

    def _is_target_call(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == self.func_name)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if isinstance(node.value, ast.Tuple):
            return self._handle_tuple_assign(node)

        if self._is_target_call(node.value):
            assert isinstance(node.value, ast.Call)
            inlined = self._try_inline(node.value)
            if inlined is not None:
                if self.is_return:
                    node.value = inlined
                    return node
                else:
                    return [ast.Expr(value=inlined),
                            ast.Assign(targets=node.targets, value=ast.Constant(value=None))]

        self.generic_visit(node)
        return node

    def _handle_tuple_assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            self.generic_visit(node)
            return node

        target_tuple = node.targets[0]
        value_tuple = node.value
        assert isinstance(value_tuple, ast.Tuple)

        if len(target_tuple.elts) != len(value_tuple.elts):
            self.generic_visit(node)
            return node

        pre_stmts = []
        new_values = []

        for target_elt, value_elt in zip(target_tuple.elts, value_tuple.elts):
            if self._is_target_call(value_elt):
                assert isinstance(value_elt, ast.Call)
                inlined = self._try_inline(value_elt)
                if inlined is not None:
                    if self.is_return:
                        new_values.append(self.visit(inlined))
                    else:
                        pre_stmts.append(ast.Expr(value=self.visit(inlined)))
                        new_values.append(ast.Constant(value=None))
                else:
                    new_values.append(value_elt)
            else:
                new_values.append(self.visit(value_elt))

        value_tuple.elts = new_values
        return pre_stmts + [node] if pre_stmts else node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | list[ast.stmt]:
        if self._is_target_call(node.value) and self.is_return:
            assert isinstance(node.value, ast.Call)
            inlined = self._try_inline(node.value)
            if inlined is not None:
                node.value = self.visit(inlined)
                return node

        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self._is_target_call(node):
            inlined = self._try_inline(node)
            if inlined is not None:
                return self.visit(inlined)

        self.generic_visit(node)
        return node

    def _try_inline(self, call: ast.Call) -> Optional[ast.expr]:
        param_to_arg = self._build_param_mapping(call)
        if param_to_arg is None:
            return None

        inlined = copy.deepcopy(self.inline_expr)
        return ArgumentReplacer(param_to_arg).visit(inlined)

    def _build_param_mapping(self, call: ast.Call) -> Optional[dict[str, ast.expr]]:
        """Build parameter to argument mapping, handling all argument types."""
        param_to_arg: dict[str, ast.expr] = {}
        func_args = self.func_args

        # Parameter names
        param_names = [arg.arg for arg in func_args.args]

        # Default values
        num_defaults = len(func_args.defaults)
        defaults_start = len(param_names) - num_defaults
        param_defaults = {
            param_names[defaults_start + i]: default
            for i, default in enumerate(func_args.defaults)
        }

        # Keyword-only parameters
        kwonly_names = [arg.arg for arg in func_args.kwonlyargs]
        kwonly_defaults = {
            kwonly_names[i]: default
            for i, default in enumerate(func_args.kw_defaults)
            if default is not None
        }

        # Process positional arguments
        used_positional = 0
        for i, arg in enumerate(call.args):
            if i < len(param_names):
                param_to_arg[param_names[i]] = arg
                used_positional += 1
            elif func_args.vararg:
                break
            else:
                return None

        # *args handling
        if func_args.vararg:
            param_to_arg[func_args.vararg.arg] = ast.Tuple(
                elts=list(call.args[used_positional:]),
                ctx=ast.Load()
            )

        # Process keyword arguments
        remaining_kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                return None  # **kwargs unpacking not supported

            if kw.arg in param_names:
                if kw.arg in param_to_arg:
                    return None  # Duplicate argument
                param_to_arg[kw.arg] = kw.value
            elif kw.arg in kwonly_names:
                param_to_arg[kw.arg] = kw.value
            elif func_args.kwarg:
                remaining_kwargs[kw.arg] = kw.value
            else:
                return None

        # **kwargs handling
        if func_args.kwarg:
            param_to_arg[func_args.kwarg.arg] = ast.Dict(
                keys=[ast.Constant(value=k) for k in remaining_kwargs.keys()],
                values=list(remaining_kwargs.values())
            )

        # Apply default values
        for param in param_names:
            if param not in param_to_arg:
                if param in param_defaults:
                    param_to_arg[param] = copy.deepcopy(param_defaults[param])
                else:
                    return None

        for param in kwonly_names:
            if param not in param_to_arg:
                if param in kwonly_defaults:
                    param_to_arg[param] = copy.deepcopy(kwonly_defaults[param])
                else:
                    return None

        return param_to_arg
