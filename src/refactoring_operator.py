"""
This module defines the RefactoringOperator class,
which are used to represent various refactoring operators that can be used
in SBSE process.
"""
from enum import EnumType

"""
impl order: DC -> IM -> CC -> RNC -> RV / RF -> EM
"""

class NodeType(EnumType):
    FunctionDef = "FunctionDef"
    If = "If"


class RefactoringOperator:
    pass


class ExtractMethodOperator(RefactoringOperator):
    pass


class RenameVariableOperator(RefactoringOperator):
    pass


class RenameFieldOperator(RefactoringOperator):
    pass


class DecomposeConditionalOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        self.target_node_type = NodeType.If
        self.target_node_no = target_node_no


class ReplaceNestedConditionalOperator(RefactoringOperator):
    pass


class InlineMethodOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        self.target_node_type = NodeType.FunctionDef
        self.target_node_no = target_node_no

    def __str__(self):
        return f"InlineMethodOperator(target_node_no={self.target_node_no})"


class ConsolidateConditionalExpressionOperator(RefactoringOperator):
    pass
