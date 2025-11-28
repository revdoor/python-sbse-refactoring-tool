"""
This module defines the CandidateGenerator class,
which is used to find the possible RefactoringOperator instances
for the given source code.
"""

import ast
from type_enums import RefactoringOperatorType
from refactoring_operator import (
    InlineMethodOperator,
    DecomposeConditionalOperator,
    ConsolidateConditionalExpressionOperator
)


def ast_equal(node1, node2):
    # recursively compare two AST nodes for equality
    if type(node1) != type(node2):
        return False

    if isinstance(node1, ast.AST):
        for field in node1._fields:
            if not ast_equal(getattr(node1, field, None),
                             getattr(node2, field, None)):
                return False
        return True

    elif isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        return all(ast_equal(n1, n2) for n1, n2 in zip(node1, node2))

    else:
        return node1 == node2


def find_same_level_ifs(if_node):
    # if-elif-elif-else structure -> if-orelse, in orelse if-orelse, ...
    # so, we need to traverse orelse till it is not an If node
    # collect the conditions and bodies too

    branches = []

    cur_node = if_node
    while isinstance(cur_node, ast.If):
        branches.append((cur_node.test, cur_node.body))

        if cur_node.orelse and len(cur_node.orelse) == 1 and isinstance(cur_node.orelse[0], ast.If):
            cur_node = cur_node.orelse[0]
        else:
            if cur_node.orelse:
                branches.append((None, cur_node.orelse))
            break

    return branches


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
    def __init__(self):
        pass

    def generate_candidates(self, source_code):
        # TODO: Implement candidate generation logic
        # It should find various valid refactoring operators applicable to the source code
        pass

    def generate_candidates_by_operator(self, source_code, operator):
        match operator:
            case RefactoringOperatorType.DC:
                return self.generate_decompose_conditional_candidates(source_code)
            case RefactoringOperatorType.IM:
                return self.generate_inline_method_candidates(source_code)
            case RefactoringOperatorType.RC:
                return self.generate_reverse_conditional_expression_candidates(source_code)
            case RefactoringOperatorType.CC:
                return self.generate_consolidate_conditional_expression_candidates(source_code)
            case _:
                return []

    @staticmethod
    def generate_decompose_conditional_candidates(source_code):
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        order = order_visitor.node_order

        candidates = []

        for node in ast.walk(root):
            if isinstance(node, ast.If):
                # consider If nodes with BoolOp of And/Or only
                # for a valid complex conditional expression
                # that would be considered as a target for DC operator
                if isinstance(node.test, ast.BoolOp) and \
                        (isinstance(node.test.op, ast.And) or isinstance(node.test.op, ast.Or)):
                    no = order[node]
                    candidates.append(DecomposeConditionalOperator(no))

        return candidates

    @staticmethod
    def generate_inline_method_candidates(source_code):
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        order = order_visitor.node_order

        candidates = []

        for node in root.body:
            if isinstance(node, ast.FunctionDef):
                # consider functions with single statement only
                # to handle simple inline method refactoring
                if len(node.body) == 1:
                    no = order[node]
                    candidates.append(InlineMethodOperator(no))

        return candidates

    @staticmethod
    def generate_reverse_conditional_expression_candidates(source_code):
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        order = order_visitor.node_order

        candidates = []

        for node in ast.walk(root):
            if isinstance(node, ast.If):
                no = order[node]
                candidates.append(DecomposeConditionalOperator(no))

        return candidates


    @staticmethod
    def generate_consolidate_conditional_expression_candidates(source_code):
        root = ast.parse(source_code)

        order_visitor = OrderVisitor()
        order_visitor.visit(root)

        order = order_visitor.node_order

        # Consider only about single if-elif-else structure,
        # with same body in each branch
        # so, we should check whether the body of each branch is same or not
        candidates = []

        for node in ast.walk(root):
            if isinstance(node, ast.If):
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
                    no = order[node]
                    candidates.append(ConsolidateConditionalExpressionOperator(no, length))

        return candidates
