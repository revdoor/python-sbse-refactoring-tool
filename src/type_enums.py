"""
Enumeration types for refactoring operators and AST node types.
"""
from enum import Enum


class RefactoringOperatorType(Enum):
    IM = "InlineMethod"
    DC = "DecomposeConditional"
    CC = "ConsolidateConditionalExpression"
    RC = "ReverseConditionalExpression"
    RNC = "ReplaceNestedConditional"
    RF = "RenameField"
    RM = "RenameMethod"
    EM = "ExtractMethod"
    EMR = "ExtractMethodWithReturn"
    RDM = "RemoveDuplicateMethod"

    def uses_llm(self):
        return self in {
            RefactoringOperatorType.RF,
            RefactoringOperatorType.RM,
            RefactoringOperatorType.EM,
            RefactoringOperatorType.EMR,
            RefactoringOperatorType.DC
        }

    def is_implemented(self):
        return self in {
            RefactoringOperatorType.IM,
            RefactoringOperatorType.DC,
            RefactoringOperatorType.CC,
            RefactoringOperatorType.RC,
            RefactoringOperatorType.RNC,
            RefactoringOperatorType.RM,
            RefactoringOperatorType.RDM,
        }


class NodeType(Enum):
    FunctionDef = "FunctionDef"
    If = "If"
    IfElse = "IfElse"
    While = "While"
    For = "For"
    Assign = "Assign"
    AugAssign = "AugAssign"
    Arg = "Arg"
    VarArg = "VarArg"
    KwArg = "KwArg"
    KwOnlyArg = "KwOnlyArg"
