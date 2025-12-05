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
    RenameFieldOperator
)
from util_ast import ast_equal


class TargetNodeFinder(ast.NodeVisitor):
    """Find target node with type and number."""

    def __init__(self, target_node_type, target_node_no):
        super().__init__()
        self.target_node_type = target_node_type
        self.target_node_no = target_node_no
        self.type_order = {}
        self.found_node = None
        self.parent_node = None
    
    def generic_visit(self, node):
        if self.found_node:
            return
        
        typ = type(node).__name__

        if typ not in self.type_order:
            self.type_order[typ] = 1
        else:
            self.type_order[typ] += 1

        if typ == self.target_node_type.value and self.type_order[typ] == self.target_node_no:
            self.found_node = node
            return

        super().generic_visit(node)
    
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


class Applier:
    def __init__(self):
        pass

    def apply_refactoring(self, source_code, refactoring_operator: RefactoringOperator) -> str:
        # TODO: Implement refactoring application logic
        # It should apply the given refactoring operator to the source code,
        # and return the modified source code

        root = ast.parse(source_code)
        root_backup = copy.deepcopy(root)

        finder = TargetNodeFinder(refactoring_operator.target_node_type, refactoring_operator.target_node_no)
        finder.visit(root)

        target_node = finder.found_node

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
                pass

            case RefactoringOperatorType.RC:
                try:
                    self._apply_rc(target_node)
                except TypeError as e:
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
                pass

            case _:
                pass
        
        return ast.unparse(root)
    
    def _apply_rm(self, root, node, operator: RenameMethodOperator) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Expected ast.(Async)FunctionDef, got {type(node).__name__}")
        
        # def old_name(): -> def new_name():
        old_name = node.name
        new_name = operator.new_name if operator.new_name else ""

        if not operator.new_name:
            raise ValueError("Got empty name")
        
        node.name = new_name

        enclosing_class = None

        for n in ast.walk(root):
            if isinstance(n, ast.ClassDef):
                if node in n.body:
                    enclosing_class = n
                    break
            
        if enclosing_class:
            # name conflict in class
            for n in ast.walk(enclosing_class):
                if n is not node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == new_name:
                    raise ValueError(f"Method '{new_name}' already exists in class '{enclosing_class.name}'")
                
            # self.method()
            renamer = CallRenamer(old_name, new_name, is_method=True)
            renamer.visit(enclosing_class)

            subclasses = []
            for n in ast.walk(root):
                if isinstance(n, ast.ClassDef):
                    for base in n.bases:
                        if isinstance(base, ast.Name) and base.id == enclosing_class.name:
                            subclasses.append(n)
                        
            
            # super.method()
            for subclass in subclasses:
                renamer = CallRenamer(old_name, new_name, is_method=True, super_only=True)
                renamer.visit(subclass)

        else:
            # name conflict in module
            for n in ast.walk(root):
                if n is not node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == new_name:
                    raise ValueError(f"Function '{new_name}' already exists in module")
                
            # func()
            renamer = CallRenamer(old_name, new_name, is_method=False)
            renamer.visit(root)
        
        ast.fix_missing_locations(root)
    
    def _apply_rf(self):
        pass
    
    def _apply_rc(self, node):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        # if (test) [body] -> if (not test) else [body]
        new_test = ast.UnaryOp(op=ast.Not(), operand=node.test)

        node.test = new_test
        node.body, node.orelse = node.orelse, node.body

    def _apply_dc(self):
        pass

    def _apply_cc(self, node, operator: ConsolidateConditionalExpressionOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        # if (test_1) [body] ... elif (test_l) [body] else [orelse]
        # -> if (test_1 or  ... or test_l) [body] else [orelse]
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
        

    def _apply_rnc(self, node, operator: ReplaceNestedConditionalOperator):
        if not isinstance(node, ast.If):
            raise TypeError(f"Expected ast.If, got {type(node).__name__}")
        
        # if (test_1) [... if (test_l) [body]]
        # -> if (test_1 and  ... and test_l) [body]
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

