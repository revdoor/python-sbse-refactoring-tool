"""
This module defines the MetricCalculator class,
which is used to calculate various software metrics
for the given source code.
"""

import ast

class MetricCalculator:
    def __init__(self):
        pass

    def calculate_metric(self, source_code):
        # TODO: Implement metric calculation logic
        # It should calculate various metrics for the given source code and return them
        # Metrics: cyclomatic complexity, lines of code, fan-in, LLM readability score
        root = ast.parse(source_code)
        code = ast.unparse(root)
        pass

    def cyclomatic_complexity(self, source_code, ast):
        # TODO:(#. edges) - (#. nodes) + 2*(#. components) in the control flow graph.
        # revisit 'radon' module: use number of decision points in AST
        # instead of control flow graph.

        class CCVisitor(ast.NodeVisitor):
            def __init__(self):
                self.cc = 1

            def generic_visit(self, node):
                # decision point rules
                if isinstance(node, (ast.If, ast.IfExp)):
                    self.cc += 1
                elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                    self.cc += 1
                    if getattr(node, 'orelse', None):
                        self.cc += 1
                elif isinstance(node, ast.Try):
                    self.cc += len(node.handlers)
                    if node.orelse:
                        self.cc += 1
                elif isinstance(node, ast.BoolOp):
                    self.cc += len(node.values) - 1
                elif isinstance(node, ast.comprehension):
                    self.cc += 1 + len(node.ifs)
                elif isinstance(node, ast.Assert):
                    self.cc += 1

                super().generic_visit(node)

        visitor = CCVisitor()
        visitor.visit(ast)
        return visitor.cc

    def SLOC(self, source_code, ast):
        '''source line of code'''
        lines = source_code.split("\n")
        return len(lines)
    
    def num_incoming_invocation(self, source_code, ast):
        #TODO:
        pass

    def num_refactoring(self, source_code, ast):
        #TODO:
        pass

    def num_failed_op(self, source_code, ast):
        #TODO:
        pass

    def LLM_readability_score(self, source_code, ast):
        #TODO:
        pass
