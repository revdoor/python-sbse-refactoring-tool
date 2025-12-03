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


class NodeType(Enum):
    FunctionDef = "FunctionDef"
    If = "If"
    Assign = "Assign"
