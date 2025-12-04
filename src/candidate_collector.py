"""
Module for collecting unique refactoring operator candidates.
"""
from abc import ABC
from typing import Generic, TypeVar, Type

from refactoring_operator import (
    RefactoringOperator,
    InlineMethodOperator,
    DecomposeConditionalOperator,
    ReverseConditionalExpressionOperator,
    ConsolidateConditionalExpressionOperator,
    ReplaceNestedConditionalOperator,
    RenameMethodOperator,
    RemoveDuplicateMethodOperator
)

T = TypeVar('T', bound=RefactoringOperator)


class CandidateCollector(ABC, Generic[T]):
    """Abstract base class for type-safe candidate collection"""

    def __init__(self, operator_class: Type[T]):
        self._operator_class = operator_class
        self._candidates: list[T] = []
        self._seen_args = set()

    def _add_unique(self, *args) -> None:
        key = args if len(args) > 1 else args[0]
        if key not in self._seen_args:
            self._candidates.append(self._operator_class(*args))
            self._seen_args.add(key)

    def get_candidates(self) -> list[T]:
        return self._candidates


class IMCollector(CandidateCollector[InlineMethodOperator]):
    def __init__(self):
        super().__init__(InlineMethodOperator)

    def add(self, no: int) -> None:
        self._add_unique(no)


class DCCollector(CandidateCollector[DecomposeConditionalOperator]):
    def __init__(self):
        super().__init__(DecomposeConditionalOperator)

    def add(self, no: int) -> None:
        self._add_unique(no)


class RCCollector(CandidateCollector[ReverseConditionalExpressionOperator]):
    def __init__(self):
        super().__init__(ReverseConditionalExpressionOperator)

    def add(self, no: int) -> None:
        self._add_unique(no)


class CCCollector(CandidateCollector[ConsolidateConditionalExpressionOperator]):
    def __init__(self):
        super().__init__(ConsolidateConditionalExpressionOperator)

    def add(self, no: int, length: int) -> None:
        self._add_unique(no, length)


class RNCCollector(CandidateCollector[ReplaceNestedConditionalOperator]):
    def __init__(self):
        super().__init__(ReplaceNestedConditionalOperator)

    def add(self, no: int, length: int) -> None:
        self._add_unique(no, length)


class RMCollector(CandidateCollector[RenameMethodOperator]):
    def __init__(self):
        super().__init__(RenameMethodOperator)

    def add(self, no: int, new_name: str) -> None:
        self._add_unique(no, new_name)


class RDMCollector(CandidateCollector[RemoveDuplicateMethodOperator]):
    def __init__(self):
        super().__init__(RemoveDuplicateMethodOperator)

    def add(self, no: int, ref_no: int) -> None:
        self._add_unique(no, ref_no)
