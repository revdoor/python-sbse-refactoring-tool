"""
This module defines the MetricCalculator class,
which is used to calculate various software metrics
for the given source code.
"""

import ast
import ollama


class MetricCalculator:
    def __init__(self):
        pass

    def calculate_metric(self, source_code):
        # TODO: Implement metric calculation logic
        # It should calculate various metrics for the given source code and return them
        # Metrics: cyclomatic complexity, lines of code, fan-in, LLM readability score

        root = ast.parse(source_code)
        code = ast.unparse(root)
        
        score_cyclomatic = self.cyclomatic_complexity(code, root)
        score_SLOC = self.SLOC(code, root)
        score_fan_in = self.num_incoming_invocation(code, root)

        print("cyclomatic", score_cyclomatic)
        print("SLOC", score_SLOC)
        print("fan-in", score_fan_in)

        pass

    def cyclomatic_complexity(self, source_code, root):
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
        visitor.visit(root)
        return visitor.cc

    def SLOC(self, source_code, root):
        # number of source line of code
        lines = source_code.split("\n")
        return len(lines)
    
    def num_incoming_invocation(self, source_code, root):
        #Fan-In: it represents how many other function/modules calls that function/modules.

        #Version: number of ast.Call
        # num_fan_in = 0

        # for node in ast.walk(root):
        #     if isinstance(node, ast.Call):

        #         num_fan_in += 1
        
        #return num_fan_in

        #Version: make a dict. of fan-in for each callable node

        fan_in_dict = {}

        for node in ast.walk(root):
            if isinstance(node, ast.Call):

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
        
        
        return fan_in_dict

    def num_refactoring(self, source_code, root):
        #TODO:
        pass

    def num_failed_op(self, source_code, root):
        #TODO:
        pass

    def LLM_readability_score(self, source_code, root):
        #calculate the LLM readability score with local LLMs such as llama-3

        prompt = f"""The assistant is a seasoned senior software engineer,
        with deep Python Language expertise,
        doing source code evaluation as part of a due diligence process,
        these source codes are presented in the form of a Python code snippet.
        Your task is to emit a score from 0 to 100 based on the readability level of the source code presented.
        The response should start with first line with 'Score: xx/100' format without any extra characters,
        and there would be some detailed explanations about that score.
        
        Following is a Python source code to emit a score.
        
        {source_code}"""

        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': 'You are a strict Python code readability evaluator.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        content = response['message']['content']

        lines = content.split('\n')
        first = lines[0].strip()

        # First line of response should follow 'Score: xx/100' format.
        # If not, this code consider it as a miss of llama-3 and ignore it.
        # If first line of response not followes given format, it considered as 0 point.

        try:
            score_str = first.replace("Score:", "").replace("/100", "").strip()
            score = int(score_str)
        except:
            score = 0

        return score


# test_code = """def hehehe(a, b, c):
#     if a == 0:
#         return True
#     if b == 0:
#         return True
#     if c == 0:
#         return True
#     return False

# def complex_formula(a, b, c):
#     temp = a + b + c
    
#     if temp % 2 == 0:
#         return hehehe(a, b, temp)
#     if temp % 3 == 0:
#         return hehehe(a, temp, c)
#     return hehehe(temp, b, c)
    
# def monte_carlo_pi(num_samples):
#     v = 0

#     for _ in range(num_samples):
#         x = random.uniform(-1, 1)
#         y = random.uniform(-1, 1)
#         if in_circle(x, y):
#             v += 1

#     return (v / num_samples) * 4
#     """

test_code = """def hehehe(a, b, c):
    if a == 0:
        return True
    if b == 0:
        return True
    if c == 0:
        return True
    return False"""

temp = MetricCalculator()
temp.calculate_metric(test_code)