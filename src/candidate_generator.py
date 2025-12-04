"""
This module defines the CandidateGenerator class,
which is used to find the possible RefactoringOperator instances
for the given source code.
"""

import ast
import random
import string

from type_enums import RefactoringOperatorType
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
from util_ast import ast_equal, ast_similar, find_same_level_ifs
from util_llm import get_recommendations_for_function_rename


class OrderVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.type_order = {}
        self.node_order = {}

    def generic_visit(self, node):
        typ = type(node).__name__

        if typ not in self.type_order:
            self.type_order[typ] = 1
        else:
            self.type_order[typ] += 1

        self.node_order[node] = self.type_order[typ]

        super().generic_visit(node)


class CandidateGenerator:
    @staticmethod
    def generate_candidates(source_code: str) -> list[RefactoringOperator]:
        # It should find various valid refactoring operators applicable to the source code
        overall_result = []

        root, node_order = CandidateGenerator._parse_and_order(source_code)

        for operator in RefactoringOperatorType:
            candidates = CandidateGenerator._generate_candidates_by_operator(root, node_order, operator)
            overall_result.extend(candidates)

        return overall_result

    @staticmethod
    def generate_candidates_by_operator(source_code: str, operator: RefactoringOperatorType) -> list[RefactoringOperator]:
        root, node_order = CandidateGenerator._parse_and_order(source_code)
        return CandidateGenerator._generate_candidates_by_operator(root, node_order, operator)

    @staticmethod
    def _generate_candidates_by_operator(root: ast.Module, node_order: dict[ast.AST, int], operator: RefactoringOperatorType) -> list[RefactoringOperator]:
        match operator:
            case RefactoringOperatorType.DC:
                return CandidateGenerator._generate_dc_candidates(root, node_order)
            case RefactoringOperatorType.IM:
                return CandidateGenerator._generate_im_candidates(root, node_order)
            case RefactoringOperatorType.RC:
                return CandidateGenerator._generate_rc_candidates(root, node_order)
            case RefactoringOperatorType.CC:
                return CandidateGenerator._generate_cc_candidates(root, node_order)
            case RefactoringOperatorType.RNC:
                return CandidateGenerator._generate_rnc_candidates(root, node_order)
            case RefactoringOperatorType.RM:
                return CandidateGenerator._generate_rm_candidates(root, node_order)
            case _:
                return []

    @staticmethod
    def _parse_and_order(source_code: str) -> tuple[ast.Module, dict[ast.AST, int]]:
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        return root, order_visitor.node_order

    @staticmethod
    def _generate_dc_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[DecomposeConditionalOperator]:
        candidates = []

        for node in ast.walk(root):
            if not isinstance(node, ast.If):
                continue

            # consider If nodes with BoolOp of And/Or only
            # for a valid complex conditional expression
            # that would be considered as a target for DC operator
            if isinstance(node.test, ast.BoolOp) and \
                    (isinstance(node.test.op, ast.And) or isinstance(node.test.op, ast.Or)):
                no = node_order[node]
                candidates.append(DecomposeConditionalOperator(no))

        return candidates

    @staticmethod
    def _generate_im_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[InlineMethodOperator]:
        candidates = []

        for node in root.body:
            if isinstance(node, ast.FunctionDef):
                # consider functions with single statement only
                # to handle simple inline method refactoring
                if len(node.body) == 1:
                    no = node_order[node]
                    candidates.append(InlineMethodOperator(no))

        return candidates

    @staticmethod
    def _generate_rc_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[ReverseConditionalExpressionOperator]:
        candidates = []

        for node in ast.walk(root):
            if isinstance(node, ast.If):
                no = node_order[node]
                candidates.append(ReverseConditionalExpressionOperator(no))

        return candidates

    @staticmethod
    def _generate_cc_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[ConsolidateConditionalExpressionOperator]:
        # Consider only about single if-elif-else structure,
        # with same body in each branch
        # so, we should check whether the body of each branch is same or not
        candidates = []

        for node in ast.walk(root):
            if not isinstance(node, ast.If):
                continue

            branch_infos = find_same_level_ifs(node)

            # consider only when the first body (If body) is same for consequent branches
            # how can we save? save as the length of continuous same bodies
            first_body = branch_infos[0][1]
            length = 1
            for cond, body in branch_infos[1:]:
                if ast_equal(first_body, body):
                    length += 1
                else:
                    break

            if length >= 2:
                no = node_order[node]
                candidates.append(ConsolidateConditionalExpressionOperator(no, length))

        return candidates

    @staticmethod
    def _generate_rnc_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[ReplaceNestedConditionalOperator]:
        # consider continuous If statements, with no 'orelse' block
        candidates = []

        for node in ast.walk(root):
            if not isinstance(node, ast.If):
                continue

            # check whether it has 'orelse' block or not
            if node.orelse:
                continue

            # continuously check whether the next statement is also an If node without orelse
            length = 1

            cur_node = node
            while True:
                body = cur_node.body

                if len(body) != 1:
                    break

                first_body = body[0]
                if not isinstance(first_body, ast.If):
                    break

                if first_body.orelse:
                    break

                length += 1
                cur_node = body[0]

            if length >= 2:
                no = node_order[node]
                candidates.append(ReplaceNestedConditionalOperator(no, length))

        return candidates

    @staticmethod
    def _generate_rm_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[RenameMethodOperator]:
        candidates = []

        for node in ast.walk(root):
            if not isinstance(node, ast.FunctionDef):
                continue

            no = node_order[node]

            orig_name = node.name
            length = 10
            node.name = ''.join(random.choice(string.ascii_letters) for _ in range(length))

            code = ast.unparse(node)
            recommendations = get_recommendations_for_function_rename(code).split('\n')[0]

            for _name in recommendations.split(","):
                name = _name.strip()
                if not name:
                    continue

                candidates.append(RenameMethodOperator(no, name))

            node.name = orig_name

        return candidates

    @staticmethod
    def _generate_rdm_candidates(root: ast.Module, node_order: dict[ast.AST, int]) -> list[RemoveDuplicateMethodOperator]:
        function_nodes = []

        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                function_nodes.append(node)

        candidates = []

        for i in range(len(function_nodes)):
            for j in range(i+1, len(function_nodes)):
                node1 = function_nodes[i]
                node2 = function_nodes[j]

                if ast_similar(node1, node2):
                    # found duplicate functions
                    # remove the latter one only
                    no = node_order[node2]
                    candidates.append(RemoveDuplicateMethodOperator(no))

        return candidates
