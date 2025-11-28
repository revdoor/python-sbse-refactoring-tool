"""
This module defines the RefactoringOperator class,
which are used to represent various refactoring operators that can be used
in SBSE process.
"""

"""
impl order: DC -> IM -> CC -> RNC -> RV / RF -> EM
"""

from type_enums import RefactoringOperatorType, NodeType


class RefactoringOperator:
    def __init__(self, operator_type, target_node_type, target_node_no):
        self.operator_type = operator_type
        self.target_node_type = target_node_type
        self.target_node_no = target_node_no

    def __str__(self):
        return f"{self.operator_type.value}(target={self.target_node_type.value}[{self.target_node_no}])"


class ExtractMethodOperator(RefactoringOperator):
    pass


class RenameVariableOperator(RefactoringOperator):
    pass


class RenameFieldOperator(RefactoringOperator):
    pass


class DecomposeConditionalOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(RefactoringOperatorType.DC, NodeType.If, target_node_no)


class ReplaceNestedConditionalOperator(RefactoringOperator):
    pass


class InlineMethodOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(RefactoringOperatorType.IM, NodeType.FunctionDef, target_node_no)


class ConsolidateConditionalExpressionOperator(RefactoringOperator):
    def __init__(self, target_node_no, length):
        super().__init__(RefactoringOperatorType.CC, NodeType.If, target_node_no)
        self.length = length

    def __str__(self):
        return f"{self.operator_type.value}(target={self.target_node_type.value}[{self.target_node_no}], length={self.length})"


class ReverseConditionalExpressionOperator(RefactoringOperator):
    def __init__(self, target_node_no):
        super().__init__(RefactoringOperatorType.RC, NodeType.If, target_node_no)
