"""
This module defines the MetricCalculator class,
which is used to calculate various software metrics
for the given source code.
"""

import ast
from util_llm import llm_readability_score


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


class MetricCalculator:
    def calculate_metric(self, source_code):
        # Metric calculation logic
        # It should calculate various metrics for the given source code and return them
        # Metrics: cyclomatic complexity, lines of code, fan-in, LLM readability score

        root = ast.parse(source_code)
        code = ast.unparse(root)
        
        score_cyclomatic = MetricCalculator.cyclomatic_complexity(root)
        score_fan_in, _dict_fan_in = MetricCalculator.fan_in(root)

        score_sloc = MetricCalculator.sloc(code)
        score_llm = llm_readability_score(code)

        # print("cyclomatic", score_cyclomatic)
        # print("SLOC", score_SLOC)
        # print("fan-in", score_fan_in)
        # print("LLM readability", score_LLM)

        return score_cyclomatic, score_sloc, score_fan_in, score_llm

    @staticmethod
    def cyclomatic_complexity(root):
        # (#. edges) - (#. nodes) + 2*(#. components) in the control flow graph.
        # idea from 'radon' module: use number of decision points in AST to calculate.

        visitor = CCVisitor()
        visitor.visit(root)

        # visitor.cc is number of decision points, which becomes the score of cyclomatic_complexity.
        return visitor.cc

    @staticmethod
    def sloc(source_code):
        # number of source line of code
        lines = source_code.split("\n")
        return len(lines)

    @staticmethod
    def fan_in(root):
        """
        Fan-in metric calculation.
        Fan-in represents how many other function/modules call that function/modules.
        The overall score is the total fan-in numbers
        """

        fan_in_dict = {}

        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                temp_attr_list = []
                cur = node.func

                while isinstance(cur, ast.Attribute):
                    temp_attr_list.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    temp_attr_list.append(cur.id)
                temp_attr_list.reverse()
                func_name = ".".join(temp_attr_list)

            fan_in_dict[func_name] = fan_in_dict.get(func_name, 0) + 1
        
        score = 0
        for func_name in fan_in_dict:
            score += fan_in_dict[func_name]
        
        return score, fan_in_dict

    # def num_refactoring(self, source_code, root):
    #     pass

    # def num_failed_op(self, source_code, root):
    #     pass


if __name__ == "__main__":
    test_code = """def hehehe(a, b, c):
        if a == 0:
            return True
        if b == 0:
            return True
        if c == 0:
            return True
        return False
    
    def complex_formula(a, b, c):
        temp = a + b + c
        
        if temp % 2 == 0:
            return hehehe(a, b, temp)
        if temp % 3 == 0:
            return hehehe(a, temp, c)
        return hehehe(temp, b, c)
        
    def monte_carlo_pi(num_samples):
        v = 0
    
        for _ in range(num_samples):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if in_circle(x, y):
                v += 1
    
        return (v / num_samples) * 4
        """

    MetricCalculator().calculate_metric(test_code)
