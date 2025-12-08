"""
This module defines the Applier class,
which is used to apply the RefactoringOperator instances
into the given source code.
"""
import ast
import copy
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
from helpers.store_load_visitor import StoreLoadVisitor


# ============================================================
# Utility Functions
# ============================================================

def get_attr_name_from_node_type(node_type) -> str:
    """Get attribute name from node type."""
    if node_type.value == "IfElse":
        return 'orelse'
    else:
        return 'body'

def find_enclosing_class(root: ast.Module, target_node: ast.AST) -> ast.ClassDef | None:
    """Find the class that contains the target node."""
    # for every ClassDef
    for node in ast.walk(root):
        if isinstance(node, ast.ClassDef):
            # check target node is in Classdef
            for descendant in ast.walk(node):
                if descendant is target_node:
                    return node
    return None

def find_enclosing_function(root: ast.Module, target_node: ast.AST) -> ast.FunctionDef | None:
    """Find the function that contains the target node."""
    # target node is FunctionDef
    if isinstance(target_node, ast.FunctionDef):
        return target_node
    
    # find Functiondef including target node
    for n in ast.walk(root):
        if isinstance(n, ast.FunctionDef):
            for item in ast.walk(n):
                if item is target_node:
                    return n
    return None

def check_name_conflict(
    root: ast.Module, 
    new_name: str, 
    enclosing_class: ast.ClassDef | None,
    exclude_node: ast.AST | None = None
) -> bool:
    """
    Check if the new function/method name already exists in the same scope.
    
    Args:
        root: The module AST
        new_name: The name to check for conflicts
        enclosing_class: The class containing the target, or None for module level
        exclude_node: Optional node to exclude from conflict check (e.g., the node being renamed)
    
    Returns:
        True if there is a name conflict, False otherwise
    """
    if enclosing_class:
        # check in enclosing class
        for node in enclosing_class.body:
            if node is exclude_node:
                continue
            if isinstance(node, ast.FunctionDef):
                if node.name == new_name:
                    return True
    else:
        # check in module
        for node in root.body:
            if node is exclude_node:
                continue
            if isinstance(node, ast.FunctionDef):
                if node.name == new_name:
                    return True
    return False


def remove_function_from_scope(
    root: ast.Module, 
    node: ast.FunctionDef, 
    enclosing_class: ast.ClassDef | None
) -> None:
    """
    Remove a function/method from its enclosing scope.
    
    Args:
        root: The module AST
        node: The function to remove
        enclosing_class: The class containing the function, or None for module level
    """
    if enclosing_class:
        enclosing_class.body.remove(node)
    else:
        root.body.remove(node)

def find_direct_enclosing_method(enclosing_class: ast.ClassDef, target_node: ast.AST) -> ast.FunctionDef | None:
    """Find the method in enclosing_class that directly or indirectly contains target_node."""
    for item in enclosing_class.body:
        if isinstance(item, ast.FunctionDef):
            for n in ast.walk(item):
                if n is target_node:
                    return item
    return None

def insert_function_to_scope(
    root: ast.Module,
    new_func: ast.FunctionDef,
    enclosing_function: ast.FunctionDef | None,
    enclosing_class: ast.ClassDef | None,
    target_node: ast.AST,
    uses_self: bool
) -> None:
    """Insert a new function/method into the appropriate scope."""
    if enclosing_class and uses_self:
        # insert in class
        # find direct enclosing method including target_node
        direct_method = find_direct_enclosing_method(enclosing_class, target_node)
        if direct_method is not None:
            method_idx = enclosing_class.body.index(direct_method)
            enclosing_class.body.insert(method_idx, new_func)
        else:
            enclosing_class.body.insert(0, new_func)
    else:
        # insert in module
        if enclosing_class is not None and enclosing_class in root.body:
            class_idx = root.body.index(enclosing_class)
            root.body.insert(class_idx, new_func)
        elif enclosing_function is not None and enclosing_function in root.body:
            func_idx = root.body.index(enclosing_function)
            root.body.insert(func_idx, new_func)
        else:
            root.body.insert(0, new_func)


# ============================================================
# AST Visitors and Transformers
# ============================================================

NODE_TYPE_TO_AST_NAME = {
    "FunctionDef": "FunctionDef",
    "If": "If",
    "IfElse": "If",
    "While": "While",
    "For": "For",
}


def find_target_node(root: ast.Module, target_node_type, target_node_no) -> ast.AST | None:
    """Find target node using ast.walk (same traversal as node_order generation)."""
    expected_type = NODE_TYPE_TO_AST_NAME.get(target_node_type.value, target_node_type.value)
    
    type_order = {}
    for node in ast.walk(root):
        typ = type(node).__name__
        
        if typ not in type_order:
            type_order[typ] = 1
        else:
            type_order[typ] += 1
        
        if typ == expected_type and type_order[typ] == target_node_no:
            return node
    
    return None

class CallRenamer(ast.NodeVisitor):
    """Rename function/method calls."""
    
    def __init__(self, old_name: str, new_name: str, is_method: bool, super_only: bool = False):
        self.old_name = old_name
        self.new_name = new_name
        self.is_method = is_method
        self.super_only = super_only
    
    def visit_Call(self, node: ast.Call) -> None:
        if self.is_method: 
            # self.method()
            if (not self.super_only and
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'self' and
                node.func.attr == self.old_name):
                node.func.attr = self.new_name
            
            # super().method()
            elif (isinstance(node.func, ast.Attribute) and
                  isinstance(node.func.value, ast.Call) and
                  isinstance(node.func.value.func, ast.Name) and
                  node.func.value.func.id == 'super' and
                  node.func.attr == self.old_name):
                node.func.attr = self.new_name
        else:
            # func()
            if (isinstance(node.func, ast.Name) and
                node.func.id == self.old_name):
                node.func.id = self.new_name
        
        self.generic_visit(node)


class ArgumentReplacer(ast.NodeTransformer):
    """Replace parameter names with actual arguments."""
    
    def __init__(self, param_to_arg: dict[str, ast.expr]):
        self.param_to_arg = param_to_arg
    
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.param_to_arg:
            return copy.deepcopy(self.param_to_arg[node.id])
        return node


class FieldRenamer(ast.NodeTransformer):
    """Rename variable/parameter within a function scope."""
    
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variable references (both Load and Store)."""
        if node.id == self.old_name:
            node.id = self.new_name
        return node
    
    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename function argument."""
        if node.arg == self.old_name:
            node.arg = self.new_name
        return node
    

class InlineMethodTransformer(ast.NodeTransformer):
    """Replace function calls with inlined expression."""
    
    def __init__(self, func_name: str, func_args: ast.arguments, inline_expr: ast.expr, is_return: bool):
        self.func_name = func_name
        self.func_args = func_args
        self.inline_expr = inline_expr
        self.is_return = is_return
    
    def _is_target_call(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == self.func_name)
    
    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        """Handle: result = func() or a, b = func(), other()"""
        
        # tuple assignment
        if isinstance(node.value, ast.Tuple):
            return self._handle_tuple_assign(node)
        
        if self._is_target_call(node.value):
            assert isinstance(node.value, ast.Call)
            inlined = self._try_inline(node.value)
            if inlined is not None:
                # assign return value of function
                if self.is_return:
                    node.value = inlined
                    return node
                # call function without return value and assign None
                else:
                    expr_stmt = ast.Expr(value=inlined)
                    node.value = ast.Constant(value=None)
                    return [expr_stmt, node]
        
        self.generic_visit(node)
        return node
    
    def _handle_tuple_assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        """Handle: a, b = func(), other()"""
        # do not handle with multiple targets or not tuple
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            self.generic_visit(node)
            return node
        
        target_tuple = node.targets[0]
        value_tuple = node.value
        
        # do not handle with different number of elements
        assert isinstance(value_tuple, ast.Tuple)
        if len(target_tuple.elts) != len(value_tuple.elts):
            self.generic_visit(node)
            return node
        
        pre_stmts = []  # statements for function without return value
        new_values = []  # new values after IM
        
        # iterate tuple
        for target_elt, value_elt in zip(target_tuple.elts, value_tuple.elts):
            if self._is_target_call(value_elt):
                assert isinstance(value_elt, ast.Call)
                inlined = self._try_inline(value_elt)
                if inlined is not None:
                    if self.is_return:
                        # handle recursive calls with return value
                        new_values.append(self.visit(inlined))
                    else:
                        # handle recursive calls without return value
                        pre_stmts.append(ast.Expr(value=self.visit(inlined)))
                        new_values.append(ast.Constant(value=None))
                else:
                    # fail at IM
                    new_values.append(value_elt)
            else:
                # not direct target for IM
                new_values.append(self.visit(value_elt))
        
        value_tuple.elts = new_values
        
        if pre_stmts:
            return pre_stmts + [node]
        return node
    
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | list[ast.stmt]:
        """Handle: a += func()"""
        if self._is_target_call(node.value):
            if not self.is_return:
                # do not handle function without return value
                return node
            
            assert isinstance(node.value, ast.Call)
            inlined = self._try_inline(node.value)
            if inlined is not None:
                # handle recursive calls
                node.value = self.visit(inlined)
                return node
        
        self.generic_visit(node)
        return node
    
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handle other call contexts."""
        if self._is_target_call(node):
            inlined = self._try_inline(node)
            if inlined is not None:
                # handle recursive calls
                return self.visit(inlined)
        
        self.generic_visit(node)
        return node
    
    def _try_inline(self, call: ast.Call) -> ast.expr | None:
        # call(params) -> expr(args)
        param_to_arg = self._build_param_mapping(call)
        if param_to_arg is None:
            return None
        
        inlined = copy.deepcopy(self.inline_expr)
        replacer = ArgumentReplacer(param_to_arg)
        inlined = replacer.visit(inlined)
        return inlined
    
    def _build_param_mapping(self, call: ast.Call) -> dict[str, ast.expr] | None:
        """Build parameter to argument mapping, handling all argument types."""
        param_to_arg: dict[str, ast.expr] = {}
        func_args = self.func_args
        
        # parameters
        # e.g., def func(a, b, c) → param_names = ['a', 'b', 'c']
        param_names = [arg.arg for arg in func_args.args]
        
        # defaults
        # e.g., def func(a, b=1, c=2) → params=[a, b, c], defaults=[1, 2] → b=1, c=2
        num_defaults = len(func_args.defaults)
        defaults_start = len(param_names) - num_defaults
        param_defaults = {
            param_names[defaults_start + i]: default
            for i, default in enumerate(func_args.defaults)
        }
        
        # keyword-only parameters
        # e.g., def func(a, *, timeout=30, retry=3) → kwonly_names = ['timeout', 'retry']
        #       kw_defaults = [30, 3]
        kwonly_names = [arg.arg for arg in func_args.kwonlyargs]
        kwonly_defaults = {}
        for i, default in enumerate(func_args.kw_defaults):
            if default is not None:
                kwonly_defaults[kwonly_names[i]] = default
        
        # positional arguments
        # e.g., func(1, 2, 3) with def func(a, b, c)
        #       → param_to_arg = {'a': 1, 'b': 2, 'c': 3}
        used_positional = 0
        for i, arg in enumerate(call.args):
            if i < len(param_names):
                param_to_arg[param_names[i]] = arg
                used_positional += 1
            elif func_args.vararg:
                # *args
                break
            else:
                return None  # too many positional arguments
        
        # *args (vararg)
        # e.g., def func(a, *args) with func(1, 2, 3, 4)
        #       → param_to_arg = {'a': 1, 'args': (2, 3, 4)}
        if func_args.vararg:
            vararg_name = func_args.vararg.arg
            remaining_positional = call.args[used_positional:]
            param_to_arg[vararg_name] = ast.Tuple(
                elts=list(remaining_positional),
                ctx=ast.Load()
            )
        
        # keyword arguments
        # e.g., func(a=1, b=2) with def func(a, b)
        #       → param_to_arg = {'a': 1, 'b': 2}
        remaining_kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                # **kwargs
                return None
            
            if kw.arg in param_names:
                if kw.arg in param_to_arg:
                    return None  # positional arguments
                param_to_arg[kw.arg] = kw.value
            elif kw.arg in kwonly_names:
                param_to_arg[kw.arg] = kw.value
            elif func_args.kwarg:
                remaining_kwargs[kw.arg] = kw.value
            else:
                return None  # unknown keyword
        
        # **kwargs
        # e.g., def func(x, **kwargs) with func(x=0, a=1, b=2)
        #       → param_to_arg = {'x': 0, 'kwargs': {'a': 1, 'b': 2}}
        if func_args.kwarg:
            kwarg_name = func_args.kwarg.arg
            keys: list = [ast.Constant(value=k) for k in remaining_kwargs.keys()]
            values = list(remaining_kwargs.values())
            param_to_arg[kwarg_name] = ast.Dict(keys=keys, values=values)
        
        # default values in parameters
        # e.g., def func(a, b=10) with func(1)
        #       → param_to_arg = {'a': 1, 'b': 10}
        for param in param_names:
            if param not in param_to_arg:
                if param in param_defaults:
                    param_to_arg[param] = copy.deepcopy(param_defaults[param])
                else:
                    return None  # missing argument
        
        # default values in keyword-only parameters
        # e.g., def func(*, timeout=30) with func()
        #       → param_to_arg = {'timeout': 30}
        for param in kwonly_names:
            if param not in param_to_arg:
                if param in kwonly_defaults:
                    param_to_arg[param] = copy.deepcopy(kwonly_defaults[param])
                else:
                    return None  # missing keyword-only argument
        
        return param_to_arg


# ============================================================
# Extract Method Helper
# ============================================================

class ExtractMethodHelper:
    """Helper class for extract method refactoring."""
    
    @staticmethod
    def get_used_variables(nodes: list[ast.stmt]) -> tuple[set[str], set[str]]:
        """Get loaded and stored variable names in nodes."""
        visitor = StoreLoadVisitor()
        for node in nodes:
            visitor.visit(node)
        return visitor.load_ids, visitor.store_ids
    
    @staticmethod
    def get_required_params(
        target_nodes: list[ast.stmt], 
        prev_nodes: list[ast.stmt],
        enclosing_function: ast.FunctionDef | None
    ) -> list[str]:
        """Get variables that need to be passed as parameters."""
        target_load, target_store = ExtractMethodHelper.get_used_variables(target_nodes)
        prev_load, prev_store = ExtractMethodHelper.get_used_variables(prev_nodes)
        
        # 함수 파라미터도 사용 가능한 변수로 간주
        func_params = set()
        if enclosing_function:
            for arg in enclosing_function.args.args:
                if arg.arg != 'self':
                    func_params.add(arg.arg)
        
        # target에서 load하는데, target 내부에서 store되지 않은 것
        external_deps = target_load - target_store
        # 그 중 prev에서 store되었거나, 함수 파라미터인 것
        params = external_deps & (prev_store | func_params)
        
        return sorted(list(params))
    
    @staticmethod
    def get_prev_stmts(
        target_node: ast.AST,
        attr_name: str,
        start_idx: int,
        enclosing_function: ast.FunctionDef | None
    ) -> list[ast.stmt]:
        """Get statements before the extraction target, including function body if needed."""
        prev_stmts = []
        
        # target_node의 attr에서 idx 이전 statements
        attr = getattr(target_node, attr_name)
        prev_stmts.extend(attr[:start_idx])
        
        # target_node가 FunctionDef가 아니면, enclosing function의 body에서
        # target_node 이전의 모든 statements도 포함
        if enclosing_function and target_node is not enclosing_function:
            for stmt in enclosing_function.body:
                if stmt is target_node:
                    break
                prev_stmts.append(stmt)
                # 중첩 구조에서 target_node를 포함하는 경우
                for child in ast.walk(stmt):
                    if child is target_node:
                        break
        
        return prev_stmts

    @staticmethod
    def uses_self(nodes: list[ast.stmt]) -> bool:
        """Check if any node uses 'self'."""
        for node in nodes:
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id == 'self':
                    return True
                # self.attr pattern
                if isinstance(n, ast.Attribute):
                    if isinstance(n.value, ast.Name) and n.value.id == 'self':
                        return True
        return False
    
    @staticmethod
    def create_function(
        name: str,
        params: list[str],
        body: list[ast.stmt],
        include_self: bool
    ) -> ast.FunctionDef:
        """Create a new function definition."""
        if include_self:
            args_list = [ast.arg(arg='self')] + [ast.arg(arg=p) for p in params]
        else:
            args_list = [ast.arg(arg=p) for p in params]
        
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
    def create_call_expr(
        func_name: str,
        params: list[str],
        use_self: bool
    ) -> ast.Call:
        """Create a function call expression."""
        call_args: list = [ast.Name(id=p, ctx=ast.Load()) for p in params]
        
        if use_self:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=func_name,
                    ctx=ast.Load()
                ),
                args=call_args,
                keywords=[]
            )
        else:
            return ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=call_args,
                keywords=[]
            )


# ============================================================
# Applier Class
# ============================================================

class Applier:
    def __init__(self):
        pass

    def apply_refactoring(self, source_code, refactoring_operator: RefactoringOperator) -> str:
        root = ast.parse(source_code)
        root_backup = copy.deepcopy(root)

        # TargetNodeFinder 대신 ast.walk 기반 함수 사용
        target_node = find_target_node(
            root, 
            refactoring_operator.target_node_type, 
            refactoring_operator.target_node_no
        )

        if target_node is None:
            raise ValueError(f"Target node not found: {refactoring_operator}")
        
        match refactoring_operator.operator_type:
            case RefactoringOperatorType.RM:
                assert isinstance(refactoring_operator, RenameMethodOperator)
                try:
                    self._apply_rm(root, target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying RM: {e}")
                    root = root_backup

            case RefactoringOperatorType.RF:
                assert isinstance(refactoring_operator, RenameFieldOperator)
                try:
                    self._apply_rf(root, target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying RF: {e}")
                    root = root_backup

            case RefactoringOperatorType.RC:
                assert isinstance(refactoring_operator, ReverseConditionalExpressionOperator)
                try:
                    self._apply_rc(target_node)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying RC: {e}")
                    root = root_backup

            case RefactoringOperatorType.DC:
                pass

            case RefactoringOperatorType.CC:
                assert isinstance(refactoring_operator, ConsolidateConditionalExpressionOperator)
                try:
                    self._apply_cc(target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying CC: {e}")
                    root = root_backup

            case RefactoringOperatorType.RNC:
                assert isinstance(refactoring_operator, ReplaceNestedConditionalOperator)
                try:
                    self._apply_rnc(target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying RNC: {e}")
                    root = root_backup

            case RefactoringOperatorType.IM:
                assert isinstance(refactoring_operator, InlineMethodOperator)
                try:
                    self._apply_im(root, target_node)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying IM: {e}")
                    root = root_backup
            
            case RefactoringOperatorType.RDM:
                assert isinstance(refactoring_operator, RemoveDuplicateMethodOperator)
                try:
                    self._apply_rdm(root, target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying RDM: {e}")
                    root = root_backup
            
            case RefactoringOperatorType.EM:
                assert isinstance(refactoring_operator, ExtractMethodOperator)
                try:
                    self._apply_em(root, target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying EM: {e}")
                    root = root_backup

            case RefactoringOperatorType.EMR:
                assert isinstance(refactoring_operator, ExtractMethodWithReturnOperator)
                try:
                    self._apply_emr(root, target_node, refactoring_operator)
                except (TypeError, ValueError) as e:
                    print(f"Error in applying EMR: {e}")
                    root = root_backup

            case _:
                pass
        
        return ast.unparse(root)
    
    @staticmethod
    def _apply_rm(root, node, operator: RenameMethodOperator):
        Applier._rename_method(root, node, operator)

    @staticmethod
    def _rename_method(root, node, operator: RenameMethodOperator, ignore_conflict=False):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected ast.FunctionDef, got {type(node).__name__}")
        
        old_name = node.name
        new_name = operator.new_name if operator.new_name else ""

        if not operator.new_name:
            raise ValueError("Got empty name")
        
        enclosing_class = find_enclosing_class(root, node)
            
        # name conflict check
        if not ignore_conflict:
            if check_name_conflict(root, new_name, enclosing_class, exclude_node=node):
                scope_name = f"class '{enclosing_class.name}'" if enclosing_class else "module"
                raise ValueError(f"Function/method '{new_name}' already exists in {scope_name}")
        
        node.name = new_name

        if enclosing_class:
            # self.method()
            renamer = CallRenamer(old_name, new_name, is_method=True)
            renamer.visit(enclosing_class)

            # super().method() in subclasses
            subclasses = []
            for n in ast.walk(root):
                if isinstance(n, ast.ClassDef):
                    for base in n.bases:
                        if isinstance(base, ast.Name) and base.id == enclosing_class.name:
                            subclasses.append(n)
            
            for subclass in subclasses:
                renamer = CallRenamer(old_name, new_name, is_method=True, super_only=True)
                renamer.visit(subclass)
        else:
            # func()
            renamer = CallRenamer(old_name, new_name, is_method=False)
            renamer.visit(root)
        
        ast.fix_missing_locations(root)
    
    @staticmethod
    def _apply_rf(root, node, operator: RenameFieldOperator):
        """Apply Rename Field refactoring."""
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        old_name = operator.old_name
        new_name = operator.new_name
        
        if not old_name or not new_name:
            raise ValueError("old_name and new_name must not be empty")
        
        if old_name == new_name:
            raise ValueError("old_name and new_name must be different")
        
        # 1. 이름 충돌 체크 - 함수 내에서 new_name이 이미 사용되는지
        visitor = StoreLoadVisitor()
        visitor.visit(node)
        existing_names = visitor.store_ids | visitor.load_ids
        
        # 함수 파라미터도 포함
        for arg in node.args.args:
            existing_names.add(arg.arg)
        if node.args.vararg:
            existing_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            existing_names.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            existing_names.add(arg.arg)
        
        # old_name 제외하고 new_name이 있으면 충돌
        existing_names.discard(old_name)
        if new_name in existing_names:
            raise ValueError(f"Name '{new_name}' already exists in function '{node.name}'")
        
        # 2. old_name이 실제로 함수 내에 존재하는지 확인
        all_names = visitor.store_ids | visitor.load_ids
        arg_names = {arg.arg for arg in node.args.args}
        if node.args.vararg:
            arg_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            arg_names.add(node.args.kwarg.arg)
        arg_names.update(arg.arg for arg in node.args.kwonlyargs)
        
        if old_name not in all_names and old_name not in arg_names:
            raise ValueError(f"Name '{old_name}' not found in function '{node.name}'")
        
        # 3. 함수 내 모든 이름 변경
        renamer = FieldRenamer(old_name, new_name)
        renamer.visit(node)
        
        ast.fix_missing_locations(root)
    
    @staticmethod
    def _apply_rc(node):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        new_test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        node.test = new_test
        node.body, node.orelse = node.orelse, node.body

    @staticmethod
    def _apply_dc():
        pass

    @staticmethod
    def _apply_cc(node, operator: ConsolidateConditionalExpressionOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        length = operator.length if operator.length else 0
        
        if length < 2:
            raise ValueError(f"Got not enough length")
        
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
        
        final_orelse = cur_node.orelse

        new_test = ast.BoolOp(op=ast.Or(), values=conditions)
        
        node.test = new_test
        node.orelse = final_orelse
        
    @staticmethod
    def _apply_rnc(node, operator: ReplaceNestedConditionalOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        length = operator.length if operator.length else 0
        
        if length < 2:
            raise ValueError(f"Got not enough length")

        conditions = []
        cur_node = node
        
        for i in range(length):
            conditions.append(cur_node.test)

            if i < length - 1:
                if cur_node.body and len(cur_node.body) == 1 and isinstance(cur_node.body[0], ast.If):
                    cur_node = cur_node.body[0]
                else:
                    raise ValueError(f"Expected ast.If, got different node")
        
        final_body = cur_node.body

        new_test = ast.BoolOp(op=ast.And(), values=conditions)

        node.test = new_test
        node.body = final_body
        node.orelse = []

    @staticmethod
    def _apply_im(root, node):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        if len(node.body) != 1:
            raise ValueError("Inline method only supports single-statement functions")
        
        stmt = node.body[0]
        
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            inline_expr = stmt.value
            is_return = True
        elif isinstance(stmt, ast.Expr):
            inline_expr = stmt.value
            is_return = False
        else:
            raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")
        
        func_name = node.name
        
        if _is_recursive(node):
            raise ValueError("Cannot inline recursive function")
        
        transformer = InlineMethodTransformer(func_name, node.args, inline_expr, is_return)
        transformer.visit(root)
        
        # remove when nobody calls
        has_remaining_calls = False
        for n in ast.walk(root):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == func_name:
                    has_remaining_calls = True
                    break
        
        if not has_remaining_calls:
            root.body.remove(node)
        
        ast.fix_missing_locations(root)

    @staticmethod
    def _apply_rdm(root, node, operator: RemoveDuplicateMethodOperator):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef, got {type(node).__name__}")
        
        ref_node = find_target_node(root, operator.target_node_type, operator.reference_node_no)
       
        if not isinstance(ref_node, ast.FunctionDef):
            raise TypeError(f"Expected FunctionDef for reference, got {type(ref_node).__name__}")
        
        # rename before remove
        rm_operator = RenameMethodOperator(target_node_no=operator.target_node_no, new_name=ref_node.name) 
        Applier._rename_method(root, node, rm_operator, ignore_conflict=True)

        enclosing_class = find_enclosing_class(root, node)
        remove_function_from_scope(root, node, enclosing_class)

    def _apply_em(self, root, node, operator):
            """Apply Extract Method (without return)."""
            # 0. 기본 정보 추출
            attr_name = get_attr_name_from_node_type(operator.target_node_type)
            idx = operator.start_idx
            length = operator.length
            new_name = operator.new_name if operator.new_name != 'name' else "extracted"
            
            # 1. 대상 속성 확인
            if not hasattr(node, attr_name):
                raise ValueError(f"Node does not have attribute '{attr_name}'")
            
            body = getattr(node, attr_name)
            
            if idx + length > len(body):
                raise ValueError("Invalid index or length for extraction")
            
            # 2. enclosing function/class 찾기
            enclosing_function = find_enclosing_function(root, node)
            assert enclosing_function
            enclosing_class = find_enclosing_class(root, node)

            # 3. 이름 충돌 체크
            if check_name_conflict(root, new_name, enclosing_class):
                raise ValueError(f"Function/method '{new_name}' already exists")
            
            # 4. 추출할 statements
            target_stmts = body[idx:idx + length]
            
            # 5. 필요한 파라미터 계산
            prev_stmts = ExtractMethodHelper.get_prev_stmts(node, attr_name, idx, enclosing_function)
            params = ExtractMethodHelper.get_required_params(target_stmts, prev_stmts, enclosing_function)
            
            # 6. self 사용 여부 확인
            uses_self = ExtractMethodHelper.uses_self(target_stmts)
            include_self = enclosing_class is not None and uses_self

            # 7. 새 함수 생성
            new_func = ExtractMethodHelper.create_function(
                name=new_name,
                params=params,
                body=copy.deepcopy(target_stmts),
                include_self=include_self
            )
            
            # 8. 함수 호출 생성
            call_expr = ast.Expr(
                value=ExtractMethodHelper.create_call_expr(new_name, params, use_self=include_self)
            )
            
            # 9. 원본 body 수정
            new_body = body[:idx] + [call_expr] + body[idx + length:]
            setattr(node, attr_name, new_body)
            
            # 10. 새 함수 추가
            insert_function_to_scope(root, new_func, enclosing_function, enclosing_class, node, uses_self)
            
            ast.fix_missing_locations(root)

    def _apply_emr(self, root, node, operator):
        """Apply Extract Method with Return."""
        # 0. 기본 정보 추출
        attr_name = get_attr_name_from_node_type(operator.target_node_type)
        idx = operator.start_idx
        length = operator.length
        new_name = operator.new_name if operator.new_name != 'name' else "extracted"
        
        # 1. 대상 속성 확인
        if not hasattr(node, attr_name):
            raise ValueError(f"Node does not have attribute '{attr_name}'")
        
        body = getattr(node, attr_name)
        
        if idx + length > len(body):
            raise ValueError("Invalid index or length for extraction")
        
        # 2. enclosing function/class 찾기
        enclosing_function = find_enclosing_function(root, node)
        assert enclosing_function
        enclosing_class = find_enclosing_class(root, node)
        
        # 3. 이름 충돌 체크
        if check_name_conflict(root, new_name, enclosing_class):
            raise ValueError(f"Function/method '{new_name}' already exists")
        
        # 4. 추출할 statements
        target_stmts = body[idx:idx + length]
        last_stmt = target_stmts[-1]
        
        # 5. 마지막 statement 타입에 따라 분기
        if isinstance(last_stmt, ast.Return):
            is_return_stmt = True
            return_var_name = None
        elif isinstance(last_stmt, ast.Assign):
            is_return_stmt = False
            if len(last_stmt.targets) != 1:
                raise ValueError("Multiple assignment targets not supported")
            return_var = last_stmt.targets[0]
            if not isinstance(return_var, ast.Name):
                raise ValueError("Only simple variable assignment supported")
            return_var_name = return_var.id
        elif isinstance(last_stmt, ast.AugAssign):
            is_return_stmt = False
            if not isinstance(last_stmt.target, ast.Name):
                raise ValueError("Only simple variable assignment supported")
            return_var_name = last_stmt.target.id
        else:
            raise ValueError(f"Last statement must be Assign, AugAssign, or Return, got {type(last_stmt).__name__}")
        
        # 6. 필요한 파라미터 계산
        prev_stmts = ExtractMethodHelper.get_prev_stmts(node, attr_name, idx, enclosing_function)
        params = ExtractMethodHelper.get_required_params(target_stmts, prev_stmts, enclosing_function)
        
        # 7. self 사용 여부 확인
        uses_self = ExtractMethodHelper.uses_self(target_stmts)
        include_self = enclosing_class is not None and uses_self
        
        # 8. 새 함수 body 생성
        if is_return_stmt:
            new_func_body = copy.deepcopy(target_stmts)
        else:
            assert return_var_name is not None
            new_func_body = copy.deepcopy(target_stmts)
            new_func_body.append(ast.Return(value=ast.Name(id=return_var_name, ctx=ast.Load())))
        
        # 9. 새 함수 생성
        new_func = ExtractMethodHelper.create_function(
            name=new_name,
            params=params,
            body=new_func_body,
            include_self=include_self
        )
        
        # 10. 호출 expression 생성
        call_expr = ExtractMethodHelper.create_call_expr(new_name, params, use_self=include_self)
        
        # 11. 호출부 statement 생성
        if is_return_stmt:
            call_stmt = ast.Return(value=call_expr)
        else:
            assert return_var_name is not None
            call_stmt = ast.Assign(
                targets=[ast.Name(id=return_var_name, ctx=ast.Store())],
                value=call_expr
            )
        
        # 12. 원본 body 수정
        new_body = body[:idx] + [call_stmt] + body[idx + length:]
        setattr(node, attr_name, new_body)
        
        # 13. 새 함수 추가
        insert_function_to_scope(root, new_func, enclosing_function, enclosing_class, node, uses_self)
        
        ast.fix_missing_locations(root)
    