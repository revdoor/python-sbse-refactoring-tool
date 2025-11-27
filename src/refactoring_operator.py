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


class RefactoringOperator:
    pass


class ExtractMethodOperator(RefactoringOperator):
    pass


class RenameVariableOperator(RefactoringOperator):
    pass


class RenameFieldOperator(RefactoringOperator):
    pass


class DecomposeConditionalOperator(RefactoringOperator):
    pass


class ReplaceNestedConditionalOperator(RefactoringOperator):
    pass


class InlineMethodOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        self.target_node_type = NodeType.FunctionDef
        self.target_node_no = target_node_no


class ConsolidateConditionalExpressionOperator(RefactoringOperator):
    pass
