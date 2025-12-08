"""
Applier Module

This module provides the Applier class for applying refactoring operators
to Python source code using AST manipulation.

Supported refactoring operations:
    - RM  : Rename Method
    - RF  : Rename Field (variable/parameter)
    - RC  : Reverse Conditional
    - DC  : Decompose Conditional
    - CC  : Consolidate Conditional
    - RNC : Replace Nested Conditional
    - IM  : Inline Method
    - RDM : Remove Duplicate Method
    - EM  : Extract Method
    - EMR : Extract Method with Return
"""

import ast
import copy
from typing import Optional

from type_enums import RefactoringOperatorType
from refactoring_operator import (
    RefactoringOperator,
    InlineMethodOperator,
    DecomposeConditionalOperator,
    ReverseConditionalExpressionOperator,
    ConsolidateConditionalExpressionOperator,
    ReplaceNestedConditionalOperator,
    RenameMethodOperator,
    RenameFieldOperator,
    RemoveDuplicateMethodOperator,
    ExtractMethodOperator,
    ExtractMethodWithReturnOperator,
)
from util_ast import ast_equal, _is_recursive
from store_load_visitor import StoreLoadVisitor


# ============================================================
# Constants
# ============================================================

NODE_TYPE_TO_AST_NAME = {
    "FunctionDef": "FunctionDef",
    "If": "If",
    "IfElse": "If",
    "While": "While",
    "For": "For",
}


# ============================================================
# Scope Utility Functions
# ============================================================

def get_attr_name_from_node_type(node_type) -> str:
    """Get the attribute name ('body' or 'orelse') from node type."""
    return 'orelse' if node_type.value == "IfElse" else 'body'


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


# ============================================================
# Node Finding
# ============================================================

class TargetNodeFinder(ast.NodeVisitor):
    """Find a target node by type and sequential number."""

    def __init__(self, target_node_type, target_node_no: int):
        super().__init__()
        self.target_node_type = target_node_type
        self.target_node_no = target_node_no
        self.type_order: dict[str, int] = {}
        self.found_node: Optional[ast.AST] = None
    
    def generic_visit(self, node):
        if self.found_node:
            return
        
        typ = type(node).__name__
        expected = NODE_TYPE_TO_AST_NAME.get(
            self.target_node_type.value, 
            self.target_node_type.value
        )

        self.type_order[typ] = self.type_order.get(typ, 0) + 1

        if typ == expected and self.type_order[typ] == self.target_node_no:
            self.found_node = node
            return

        super().generic_visit(node)


def find_target_node(
    root: ast.Module, 
    target_node_type, 
    target_node_no: int
) -> Optional[ast.AST]:
    """Find target node using NodeVisitor (same traversal as node_order generation)."""
    finder = TargetNodeFinder(target_node_type, target_node_no)
    finder.visit(root)
    return finder.found_node


# ============================================================
# AST Transformers - Renaming
# ============================================================

class CallRenamer(ast.NodeVisitor):
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
    
    def visit_Call(self, node: ast.Call) -> None:
        if self.is_method:
            self._handle_method_call(node)
        else:
            self._handle_function_call(node)
        self.generic_visit(node)
    
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


# ============================================================
# AST Transformers - Inline Method
# ============================================================

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


# ============================================================
# Refactoring Helper
# ============================================================

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


# ============================================================
# Applier Class
# ============================================================

class Applier:
    """Apply refactoring operators to Python source code."""
    
    def apply_refactoring(
        self, 
        source_code: str, 
        refactoring_operator: RefactoringOperator
    ) -> str:
        """
        Apply a refactoring operator to source code.
        
        Args:
            source_code: The Python source code to refactor
            refactoring_operator: The refactoring operation to apply
            
        Returns:
            The refactored source code
        """
        root = ast.parse(source_code)
        root_backup = copy.deepcopy(root)

        target_node = find_target_node(
            root, 
            refactoring_operator.target_node_type, 
            refactoring_operator.target_node_no
        )

        if target_node is None:
            raise ValueError(f"Target node not found: {refactoring_operator}")
        
        try:
            self._dispatch_operator(root, target_node, refactoring_operator)
        except (TypeError, ValueError) as e:
            op_type = refactoring_operator.operator_type.name
            print(f"Error in applying {op_type}: {e}")
            root = root_backup
        
        return ast.unparse(root)
    
    def _dispatch_operator(
        self, 
        root: ast.Module, 
        target_node: ast.AST, 
        operator: RefactoringOperator
    ) -> None:
        """Dispatch to the appropriate apply method based on operator type."""
        dispatch_table = {
            RefactoringOperatorType.RM: self._apply_rm,
            RefactoringOperatorType.RF: self._apply_rf,
            RefactoringOperatorType.RC: self._apply_rc,
            RefactoringOperatorType.DC: self._apply_dc,
            RefactoringOperatorType.CC: self._apply_cc,
            RefactoringOperatorType.RNC: self._apply_rnc,
            RefactoringOperatorType.IM: self._apply_im,
            RefactoringOperatorType.RDM: self._apply_rdm,
            RefactoringOperatorType.EM: self._apply_em,
            RefactoringOperatorType.EMR: self._apply_emr,
        }
        
        handler = dispatch_table.get(operator.operator_type)
        if handler:
            handler(root, target_node, operator)
    
    # =========================================================
    # Rename Method (RM)
    # =========================================================
    
    def _apply_rm(self, root, node, operator: RenameMethodOperator):
        self._rename_method(root, node, operator)

    def _rename_method(
        self, 
        root, 
        node, 
        operator: RenameMethodOperator, 
        ignore_conflict: bool = False
    ):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected ast.FunctionDef, got {type(node).__name__}")
        
        old_name = node.name
        new_name = operator.new_name
        
        if not new_name:
            raise ValueError("Got empty name")
        
        enclosing_class = find_enclosing_class(root, node)
        
        if not ignore_conflict:
            if check_name_conflict(root, new_name, enclosing_class, exclude_node=node):
                scope = f"class '{enclosing_class.name}'" if enclosing_class else "module"
                raise ValueError(f"Function/method '{new_name}' already exists in {scope}")
        
        node.name = new_name

        if enclosing_class:
            # Rename self.method() calls
            CallRenamer(old_name, new_name, is_method=True).visit(enclosing_class)
            
            # Rename super().method() calls in subclasses
            for n in ast.walk(root):
                if isinstance(n, ast.ClassDef):
                    for base in n.bases:
                        if isinstance(base, ast.Name) and base.id == enclosing_class.name:
                            CallRenamer(old_name, new_name, is_method=True, super_only=True).visit(n)
        else:
            # Rename func() calls
            CallRenamer(old_name, new_name, is_method=False).visit(root)
        
        ast.fix_missing_locations(root)
    
    # =========================================================
    # Rename Field (RF)
    # =========================================================
    
    def _apply_rf(self, root, node, operator: RenameFieldOperator):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        old_name, new_name = operator.old_name, operator.new_name
        
        if not old_name or not new_name:
            raise ValueError("old_name and new_name must not be empty")
        
        if old_name == new_name:
            raise ValueError("old_name and new_name must be different")
        
        # Collect all names in function
        func_params = RefactoringHelper.get_function_params(node)
        func_params.add('self') if any(
            arg.arg == 'self' for arg in node.args.args
        ) else None
        
        visitor = StoreLoadVisitor()
        visitor.visit(node)
        all_names = visitor.store_ids | visitor.load_ids | func_params
        
        if old_name not in all_names:
            raise ValueError(f"Name '{old_name}' not found in function '{node.name}'")
        
        if new_name in (all_names - {old_name}):
            raise ValueError(f"Name '{new_name}' already exists in function '{node.name}'")
        
        FieldRenamer(old_name, new_name).visit(node)
        ast.fix_missing_locations(root)
    
    # =========================================================
    # Reverse Conditional (RC)
    # =========================================================
    
    def _apply_rc(self, root, node, operator: ReverseConditionalExpressionOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        node.body, node.orelse = node.orelse, node.body
    
    # =========================================================
    # Decompose Conditional (DC)
    # =========================================================
    
    def _apply_dc(self, root, node, operator: DecomposeConditionalOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        new_name = operator.new_name
        if not new_name:
            raise ValueError("new_name must not be empty")
        
        enclosing_function = find_enclosing_function(root, node)
        enclosing_class = find_enclosing_class(root, node)
        
        if check_name_conflict(root, new_name, enclosing_class):
            raise ValueError(f"Function/method '{new_name}' already exists")
        
        used_vars = RefactoringHelper.get_used_variables_in_expr(node.test)
        uses_self = 'self' in used_vars
        used_vars.discard('self')
        
        params = sorted(used_vars)
        include_self = enclosing_class is not None and uses_self
        
        new_func = RefactoringHelper.create_function(
            name=new_name,
            params=params,
            body=[ast.Return(value=copy.deepcopy(node.test))],
            include_self=include_self
        )
        
        node.test = RefactoringHelper.create_call_expr(new_name, params, use_self=include_self)
        insert_function_to_scope(root, new_func, enclosing_function, enclosing_class, node, uses_self)
        ast.fix_missing_locations(root)
    
    # =========================================================
    # Consolidate Conditional (CC)
    # =========================================================
    
    def _apply_cc(self, root, node, operator: ConsolidateConditionalExpressionOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        length = operator.length or 0
        if length < 2:
            raise ValueError("Got not enough length")
        
        conditions = [node.test]
        first_body = node.body
        cur_node = node
        
        for _ in range(length - 1):
            if not (cur_node.orelse and isinstance(cur_node.orelse[0], ast.If)):
                raise ValueError("Expected ast.If, got different node")
            
            next_node = cur_node.orelse[0]
            
            if not ast_equal(first_body, next_node.body):
                raise ValueError("Got different body")
            
            conditions.append(next_node.test)
            cur_node = next_node
        
        node.test = ast.BoolOp(op=ast.Or(), values=conditions)
        node.orelse = cur_node.orelse
    
    # =========================================================
    # Replace Nested Conditional (RNC)
    # =========================================================
    
    def _apply_rnc(self, root, node, operator: ReplaceNestedConditionalOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        length = operator.length or 0
        if length < 2:
            raise ValueError("Got not enough length")

        conditions = []
        cur_node = node
        
        for i in range(length):
            conditions.append(cur_node.test)
            
            if i < length - 1:
                if (cur_node.body and 
                    len(cur_node.body) == 1 and 
                    isinstance(cur_node.body[0], ast.If)):
                    cur_node = cur_node.body[0]
                else:
                    raise ValueError("Expected ast.If, got different node")
        
        node.test = ast.BoolOp(op=ast.And(), values=conditions)
        node.body = cur_node.body
        node.orelse = []
    
    # =========================================================
    # Inline Method (IM)
    # =========================================================
    
    def _apply_im(self, root, node, operator: InlineMethodOperator):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        if len(node.body) != 1:
            raise ValueError("Inline method only supports single-statement functions")
        
        stmt = node.body[0]
        
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            inline_expr, is_return = stmt.value, True
        elif isinstance(stmt, ast.Expr):
            inline_expr, is_return = stmt.value, False
        else:
            raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")
        
        if _is_recursive(node):
            raise ValueError("Cannot inline recursive function")
        
        func_name = node.name
        InlineMethodTransformer(func_name, node.args, inline_expr, is_return).visit(root)
        
        # Remove function if no remaining calls
        has_calls = any(
            isinstance(n, ast.Call) and 
            isinstance(n.func, ast.Name) and 
            n.func.id == func_name
            for n in ast.walk(root)
        )
        
        if not has_calls:
            root.body.remove(node)
        
        ast.fix_missing_locations(root)
    
    # =========================================================
    # Remove Duplicate Method (RDM)
    # =========================================================
    
    def _apply_rdm(self, root, node, operator: RemoveDuplicateMethodOperator):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        assert operator.reference_node_no
        ref_node = find_target_node(root, operator.target_node_type, operator.reference_node_no)
        
        if not isinstance(ref_node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef for reference, got {type(ref_node).__name__}")
        
        # Rename calls to duplicate → reference, then remove duplicate
        rm_operator = RenameMethodOperator(
            target_node_no=operator.target_node_no, 
            new_name=ref_node.name
        )
        self._rename_method(root, node, rm_operator, ignore_conflict=True)
        
        enclosing_class = find_enclosing_class(root, node)
        remove_function_from_scope(root, node, enclosing_class)
    
    # =========================================================
    # Extract Method (EM)
    # =========================================================
    
    def _apply_em(self, root, node, operator: ExtractMethodOperator):
        attr_name = get_attr_name_from_node_type(operator.target_node_type)
        idx, length = operator.start_idx, operator.length
        assert idx is not None and length is not None
        new_name = operator.new_name if operator.new_name != 'name' else "extracted"
        assert new_name is not None
        
        if not hasattr(node, attr_name):
            raise ValueError(f"Node does not have attribute '{attr_name}'")
        
        body = getattr(node, attr_name)
        
        if idx + length > len(body):
            raise ValueError("Invalid index or length for extraction")
        
        enclosing_function = find_enclosing_function(root, node)
        assert enclosing_function is not None
        enclosing_class = find_enclosing_class(root, node)

        if check_name_conflict(root, new_name, enclosing_class):
            raise ValueError(f"Function/method '{new_name}' already exists")
        
        target_stmts = body[idx:idx + length]
        prev_stmts = RefactoringHelper.get_prev_stmts(node, attr_name, idx, enclosing_function)
        params = RefactoringHelper.get_required_params(target_stmts, prev_stmts, enclosing_function)
        
        uses_self = RefactoringHelper.uses_self_in_stmts(target_stmts)
        include_self = enclosing_class is not None and uses_self

        new_func = RefactoringHelper.create_function(
            name=new_name,
            params=params,
            body=copy.deepcopy(target_stmts),
            include_self=include_self
        )
        
        call_expr = ast.Expr(
            value=RefactoringHelper.create_call_expr(new_name, params, use_self=include_self)
        )
        
        setattr(node, attr_name, body[:idx] + [call_expr] + body[idx + length:])
        insert_function_to_scope(root, new_func, enclosing_function, enclosing_class, node, uses_self)
        ast.fix_missing_locations(root)
    
    # =========================================================
    # Extract Method with Return (EMR)
    # =========================================================
    
    def _apply_emr(self, root, node, operator: ExtractMethodWithReturnOperator):
        attr_name = get_attr_name_from_node_type(operator.target_node_type)
        idx, length = operator.start_idx, operator.length
        assert idx and length
        new_name = operator.new_name if operator.new_name != 'name' else "extracted"
        assert new_name
        
        if not hasattr(node, attr_name):
            raise ValueError(f"Node does not have attribute '{attr_name}'")
        
        body = node.body
        
        if idx + length > len(body):
            raise ValueError("Invalid index or length for extraction")
        
        enclosing_function = find_enclosing_function(root, node)
        assert enclosing_function
        enclosing_class = find_enclosing_class(root, node)
        
        if check_name_conflict(root, new_name, enclosing_class):
            raise ValueError(f"Function/method '{new_name}' already exists")
        
        target_stmts = body[idx:idx + length]
        prev_stmts = body[:idx]
        last_stmt = target_stmts[-1]
        
        # Determine return handling
        if isinstance(last_stmt, ast.Return):
            is_return_stmt, return_var_name = True, None
        elif isinstance(last_stmt, ast.Assign):
            if len(last_stmt.targets) != 1 or not isinstance(last_stmt.targets[0], ast.Name):
                raise ValueError("Only simple single-variable assignment supported")
            is_return_stmt, return_var_name = False, last_stmt.targets[0].id
        elif isinstance(last_stmt, ast.AugAssign):
            if not isinstance(last_stmt.target, ast.Name):
                raise ValueError("Only simple variable assignment supported")
            is_return_stmt, return_var_name = False, last_stmt.target.id
        else:
            raise ValueError(f"Last statement must be Assign, AugAssign, or Return")
        
        prev_stmts = RefactoringHelper.get_prev_stmts(node, attr_name, idx, enclosing_function)
        params = RefactoringHelper.get_required_params(target_stmts, prev_stmts, enclosing_function)
        
        uses_self = RefactoringHelper.uses_self_in_stmts(target_stmts)
        include_self = enclosing_class is not None and uses_self
        
        # Build new function body
        new_func_body = copy.deepcopy(target_stmts)
        if not is_return_stmt:
            assert return_var_name
            new_func_body.append(ast.Return(value=ast.Name(id=return_var_name, ctx=ast.Load())))
        
        new_func = RefactoringHelper.create_function(
            name=new_name,
            params=params,
            body=new_body,
            include_self=include_self
        )
        
        # Build call statement
        call_expr = RefactoringHelper.create_call_expr(new_name, params, use_self=include_self)
        
        if is_return_stmt:
            call_stmt = ast.Return(value=call_expr)
        else:
            assert return_var_name
            call_stmt = ast.Assign(
                targets=[ast.Name(id=return_var_name, ctx=ast.Store())],
                value=call_expr
            )
        
        setattr(node, attr_name, body[:idx] + [call_stmt] + body[idx + length:])
        insert_function_to_scope(root, new_func, enclosing_function, enclosing_class, node, uses_self)
        ast.fix_missing_locations(root)
