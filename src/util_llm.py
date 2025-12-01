from cmath import isnan

import ollama
import time
import ast


MODEL = 'llama3.2:3b'


def get_recommendations_for_rename(context_code, target_lines):
    prompt = f"""Suggest better names for the target lines.

Context Code:
{context_code}

The target lines are:
{target_lines}

In each line, suggest 3 better names only, with the order of preference, separated by commas.
Do not include any additional text or formatting. Just response as "name1, name2, name3 (original line)" format.
Names only, no 'def' or '='
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def get_recommendations_for_function_rename(function_code):
    print(function_code)
    prompt = f"""Have a break, and then suggest better names for the given function. (Currently the name is random string)

Code:
{function_code}

Suggest 3 better names only, with the order of preference, separated by commas.
Do not include any additional text or formatting. Just response as "name1, name2, name3" format.
Names only, no 'def'.
Give reasons simply, in the next line.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def readability_score(code):
    prompt = """The assistant is a seasoned senior software engineer, with deep Python Language expertise, doing source code evaluation as part of a due diligence process, these source codes are presented in the form of a Python code snippet.
Your task is to emit a score from 0 to 100 based on the readability level of the source code presented.
The response should start with 'Score: xx/100' format, and there would be some detailed explanations about that score. """

    chat = f"""My code is here:
{code}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': chat}
        ]
    )

    print(response['message']['content'].split('\n')[0])


if __name__ == "__main__":
#     start_time = time.time()
#
#     context = """def hehehe(a, b, c):
#     if a == 0:
#         return True
#     if b == 0:
#         return True
#     if c == 0:
#         return True
#     return False
#
# def complex_formula(a, b, c):
#     temp = a + b + c
#
#     if temp % 2 == 0:
#         return True
#     if temp % 3 == 0:
#         return True
#     return False
#
# def monte_carlo_pi(num_samples):
#     v = 0
#
#     for _ in range(num_samples):
#         x = random.uniform(-1, 1)
#         y = random.uniform(-1, 1)
#         if in_circle(x, y):
#             v += 1
#
#     return (v / num_samples) * 4
#     """
#
#     target_line_no = [0, 10, 19]
#
#     lines = context.split('\n')
#     target_lines = [lines[no].strip() for no in target_line_no]
#
#     print("target lines are")
#     for line in target_lines:
#         print(line)
#
#     recommendations = get_recommendations_for_rename(context, target_lines)
#
#     print("Recommended names for the target line:")
#     print(recommendations)
#
#     print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    ft_code = """def hehehe(a, b, c):
    if a == 0:
        return True
    if b == 0:
        return True
    if c == 0:
        return True
    return False"""

    start_time = time.time()

    recommendations = get_recommendations_for_function_rename(ft_code)

    print("Recommended names:")
    print(recommendations)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    root = ast.parse(ft_code)

    ft_node = None
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            ft_node = node
            break

    for _name in recommendations.split('\n')[0].split(','):
        name = _name.strip()

        ft_node.name = name
        edited_code = ast.unparse(root)

        start_time = time.time()
        print(f"Check readability score for {name}...")
        readability_score(edited_code)
        print(f"Execution Time: {time.time() - start_time:.2f} seconds")
        print()
