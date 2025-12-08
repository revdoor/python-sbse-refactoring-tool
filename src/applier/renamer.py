import ast


class CallRenamer(ast.NodeTransformer):
    """Rename function/method calls throughout the AST."""

    def __init__(
            self,
            old_name: str,
            new_name: str,
            is_method: bool,
            super_only: bool = False
    ):
        self.old_name = old_name
        self.new_name = new_name
        self.is_method = is_method
        self.super_only = super_only

    def visit_Call(self, node: ast.Call):
        if self.is_method:
            self._handle_method_call(node)
        else:
            self._handle_function_call(node)
        self.generic_visit(node)
        return node

    def _handle_method_call(self, node: ast.Call) -> None:
        """Handle self.method() and super().method() calls."""
        if not isinstance(node.func, ast.Attribute):
            return

        if node.func.attr != self.old_name:
            return

        # super().method()
        if (isinstance(node.func.value, ast.Call) and
                isinstance(node.func.value.func, ast.Name) and
                node.func.value.func.id == 'super'):
            node.func.attr = self.new_name

        # self.method() (skip if super_only)
        elif (not self.super_only and
              isinstance(node.func.value, ast.Name) and
              node.func.value.id == 'self'):
            node.func.attr = self.new_name

    def _handle_function_call(self, node: ast.Call) -> None:
        """Handle standalone function calls."""
        if isinstance(node.func, ast.Name) and node.func.id == self.old_name:
            node.func.id = self.new_name


class FieldRenamer(ast.NodeTransformer):
    """Rename variable/parameter within a function scope, respecting nested scopes."""

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
        self.is_top_level = True

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self.old_name:
            node.id = self.new_name
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg == self.old_name:
            node.arg = self.new_name
        return node

    # --- Nested Scope Handling ---

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self.is_top_level:
            self.is_top_level = False
            self.generic_visit(node)
            return node
        return node  # Skip nested functions

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        if self.old_name in self._get_lambda_params(node):
            return node  # Skip if parameter shadows old_name
        node.body = self.visit(node.body)
        return node

    def _get_lambda_params(self, node: ast.Lambda) -> set[str]:
        params = set()
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            params.add(arg.arg)
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        return params

    # --- Comprehension Handling ---

    def _get_comprehension_targets(self, generators: list[ast.comprehension]) -> set[str]:
        targets = set()
        for gen in generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    targets.add(n.id)
        return targets

    def _handle_comprehension(self, node):
        if self.old_name in self._get_comprehension_targets(node.generators):
            return node  # Skip if target shadows old_name
        self.generic_visit(node)
        return node

    def visit_ListComp(self, node: ast.ListComp):
        return self._handle_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp):
        return self._handle_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp):
        return self._handle_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        return self._handle_comprehension(node)
