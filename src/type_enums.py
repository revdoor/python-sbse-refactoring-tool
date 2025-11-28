"""
Enumeration types for refactoring operators and AST node types.
"""
from enum import Enum


class RefactoringOperatorType(Enum):
    IM = "InlineMethod"
    DC = "DecomposeConditional"


class NodeType(Enum):
    FunctionDef = "FunctionDef"
    If = "If"
