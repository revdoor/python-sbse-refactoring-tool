import ollama
import time
import ast
import re


MODEL = 'llama3'


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
    prompt = f"""Have a break, and then suggest better names for the given function. (Currently the name is random string)

Code:
{function_code}

Suggest 3 better snake-case names only, with the order of preference, separated by commas.
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

    chat = f"""Mainly consider whether the name of the function is proper or not.
My code is here:
{code}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': chat}
        ]
    )

    return response['message']['content']


def get_score(score_str):
    try:
        raw_val = re.search(r'Score: (\d+)/100', score_str).group(1)
        return int(raw_val)
    except:
        return 0


if __name__ == "__main__":
    ft_codes = [
        """def IlwRKbZAMcQnHzuWQiIy(a, b, c):
    if a == 0:
        return True
    if b == 0:
        return True
    if c == 0:
        return True
    return False""",
        """def check_val_does_not_0(a, b, c):
    if a == 0:
        return True
    if b == 0:
        return True
    if c == 0:
        return True
    return False
    """,
        """def check(num_str, key):
    cnt = 0
    length = len(key)

    for i in range(length):
        if key[i] == num_str[i]:
            cnt += 1

    return cnt""",
        """def KvLsfVjrJXoRzwkdwyYG(num_str, key):
    cnt = 0
    length = len(key)

    for i in range(length):
        if key[i] == num_str[i]:
            cnt += 1

    return cnt
        """,
        """def phi_guess(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if in_circle(x, y):
            inside_circle += 1

    return (inside_circle / num_samples) * 4""",
        """def BJvoXXfgvszIIOfFaIAn(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if in_circle(x, y):
            inside_circle += 1

    return (inside_circle / num_samples) * 4"""
    ]

    for i in range(len(ft_codes)):
        ft_code = ft_codes[i]

        with open(f"score_result_{i}.txt", 'w') as f:
            results = set()

            for _ in range(10):
                start_time = time.time()

                recommendations = get_recommendations_for_function_rename(ft_code)

                print("Recommended names:")
                print(recommendations)

                print(f"Execution Time: {time.time() - start_time:.2f} seconds")

                for _name in recommendations.split('\n')[0].split(','):
                    name = _name.strip()
                    results.add(name)

            print("The collected names are:")
            print(results)

            root = ast.parse(ft_code)

            ft_node = None
            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef):
                    ft_node = node
                    break

            std_scores = [get_score(readability_score(ft_code).split('\n')[0]) for _ in range(3)]
            recommendation_scores = []

            for name in results:
                if not name:
                    continue

                ft_node.name = name
                edited_code = ast.unparse(root)

                print(f"Check readability score for {name}...")
                for _ in range(3):
                    start_time = time.time()
                    res = readability_score(edited_code)
                    print(res)
                    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
                    print()

                    score = get_score(res.split('\n')[0])
                    recommendation_scores.append(score)

            f.write(f"Score for origin: {std_scores}\n")
            f.write(f"Score for recommendations: {recommendation_scores}\n\n")
