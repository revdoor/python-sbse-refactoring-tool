"""
This module defines the CandidateGenerator class,
which is used to find the possible RefactoringOperator instances
for the given source code.
"""

import ast
from refactoring_operator import InlineMethodOperator


class CandidateGenerator:
    def __init__(self):
        pass

    def generate_candidates(self, source_code):
        # TODO: Implement candidate generation logic
        # It should find various valid refactoring operators applicable to the source code
        pass

    def generate_inline_method_candidates(self, source_code):
        root = ast.parse(source_code)

        candidates = []

        no = 1
        for node in root.body:
            if isinstance(node, ast.FunctionDef):
                # consider functions with single statement only
                # to handle simple inline method refactoring
                if len(node.body) == 1:
                    candidates.append(InlineMethodOperator(no))
                    no += 1

        return candidates
