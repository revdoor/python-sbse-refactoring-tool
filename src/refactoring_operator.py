"""
This module defines the RefactoringOperator class,
which are used to represent various refactoring operators that can be used
in SBSE process.
"""

"""
impl order: DC -> IM -> CC -> RNC -> RM / RF -> EM
"""

from abc import ABC
from type_enums import RefactoringOperatorType, NodeType


class RefactoringOperator(ABC):
    def __init__(self, *, operator_type, target_node_type, target_node_no, length=None):
        self.operator_type = operator_type
        self.target_node_type = target_node_type
        self.target_node_no = target_node_no
        self.length = length

    def __str__(self):
        var_strs = [f"target={self.target_node_type.value}[{self.target_node_no}]"]

        if self.length:
            var_strs.append(f"length={self.length}")

        var_str = ", ".join(var_strs)

        return f"{self.operator_type.value}({var_str})"


class ExtractMethodOperator(RefactoringOperator):
    pass


class RenameMethodOperator(RefactoringOperator):
    pass


class RenameFieldOperator(RefactoringOperator):
    pass


class DecomposeConditionalOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(
            operator_type=RefactoringOperatorType.DC,
            target_node_type=NodeType.If,
            target_node_no=target_node_no
        )


class ReplaceNestedConditionalOperator(RefactoringOperator):
    def __init__(self, target_node_no, length):
        super().__init__(
            operator_type=RefactoringOperatorType.RNC,
            target_node_type=NodeType.If,
            target_node_no=target_node_no,
            length=length
        )


class InlineMethodOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(
            operator_type=RefactoringOperatorType.IM,
            target_node_type=NodeType.FunctionDef,
            target_node_no=target_node_no
        )


class ConsolidateConditionalExpressionOperator(RefactoringOperator):
    def __init__(self, target_node_no, length):
        super().__init__(
            operator_type=RefactoringOperatorType.CC,
            target_node_type=NodeType.If,
            target_node_no=target_node_no,
            length=length
        )


class ReverseConditionalExpressionOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(
            operator_type=RefactoringOperatorType.RC,
            target_node_type=NodeType.If,
            target_node_no=target_node_no
        )
