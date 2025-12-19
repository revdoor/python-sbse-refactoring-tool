import ast
import copy

from .target_node_finder import find_target_node
from .inline_method_transformer import InlineMethodTransformer
from .refactoring_helper import RefactoringHelper
from .renamer import CallRenamer, FieldRenamer
from .util_scope import (
    find_enclosing_class,
    find_enclosing_function,
    check_name_conflict,
    insert_function_to_scope,
    remove_function_from_scope,
    get_attr_name_from_node_type,
)

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


class Applier:
    """Apply refactoring operators to Python source code."""

    @staticmethod
    def apply_refactoring(
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
            Applier._dispatch_operator(root, target_node, refactoring_operator)
        except (TypeError, ValueError) as e:
            op_type = refactoring_operator.operator_type.name
            print(f"Error in applying {op_type}: {e}")
            root = root_backup

        ast.fix_missing_locations(root)

        return ast.unparse(root)

    @staticmethod
    def _dispatch_operator(
            root: ast.Module,
            target_node: ast.AST,
            operator: RefactoringOperator
    ) -> None:
        """Dispatch to the appropriate apply method based on operator type."""
        match operator:
            case RenameMethodOperator():
                Applier._apply_rm(root, target_node, operator)
            case RenameFieldOperator():
                Applier._apply_rf(root, target_node, operator)
            case ReverseConditionalExpressionOperator():
                Applier._apply_rc(root, target_node, operator)
            case DecomposeConditionalOperator():
                Applier._apply_dc(root, target_node, operator)
            case ConsolidateConditionalExpressionOperator():
                Applier._apply_cc(root, target_node, operator)
            case ReplaceNestedConditionalOperator():
                Applier._apply_rnc(root, target_node, operator)
            case InlineMethodOperator():
                Applier._apply_im(root, target_node, operator)
            case RemoveDuplicateMethodOperator():
                Applier._apply_rdm(root, target_node, operator)
            case ExtractMethodOperator():
                Applier._apply_em(root, target_node, operator)
            case ExtractMethodWithReturnOperator():
                Applier._apply_emr(root, target_node, operator)

    # =========================================================
    # Rename Method (RM)
    # =========================================================

    @staticmethod
    def _apply_rm(root: ast.Module, node: ast.AST, operator: RenameMethodOperator):
        Applier._rename_method(root, node, operator)

    @staticmethod
    def _rename_method(
            root: ast.Module,
            node: ast.AST,
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

    @staticmethod
    def _apply_rf(root: ast.Module, node: ast.AST, operator: RenameFieldOperator):
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

    @staticmethod
    def _apply_rc(root: ast.Module, node: ast.AST, operator: ReverseConditionalExpressionOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")

        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        node.body, node.orelse = node.orelse, node.body

    # =========================================================
    # Decompose Conditional (DC)
    # =========================================================

    @staticmethod
    def _apply_dc(root: ast.Module, node: ast.AST, operator: DecomposeConditionalOperator):
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

    @staticmethod
    def _apply_cc(root: ast.Module, node: ast.AST, operator: ConsolidateConditionalExpressionOperator):
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

    @staticmethod
    def _apply_rnc(root: ast.Module, node: ast.AST, operator: ReplaceNestedConditionalOperator):
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

    @staticmethod
    def _apply_im(root: ast.Module, node: ast.AST, operator: InlineMethodOperator):
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

    @staticmethod
    def _apply_rdm(root: ast.Module, node: ast.AST, operator: RemoveDuplicateMethodOperator):
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
        Applier._rename_method(root, node, rm_operator, ignore_conflict=True)

        enclosing_class = find_enclosing_class(root, node)
        remove_function_from_scope(root, node, enclosing_class)

    # =========================================================
    # Extract Method (EM)
    # =========================================================

    @staticmethod
    def _apply_em(root: ast.Module, node: ast.AST, operator: ExtractMethodOperator):
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

    @staticmethod
    def _apply_emr(root: ast.Module, node: ast.AST, operator: ExtractMethodWithReturnOperator):
        attr_name = get_attr_name_from_node_type(operator.target_node_type)
        idx, length = operator.start_idx, operator.length
        assert idx is not None and length is not None
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
            body=new_func_body,
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
