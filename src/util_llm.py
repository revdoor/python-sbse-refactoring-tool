import time
import re
import ollama


def get_recommendation_for_rename(context_code, target_lines):
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
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def get_recommendation_for_function_rename(function_code):
    prompt = f"""Suggest better names for the given function.

    Code:
    {function_code}

    Suggest 3 better names only, with the order of preference, separated by commas.
    Do not include any additional text or formatting. Just response as "name1, name2, name3" format.
    Names only, no 'def'.
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def get_recommendation_for_field_rename(function_code, field_name):
    prompt = f"""Suggest better names for the given field.

    Code:
    {function_code}
    Target field:
    {field_name}

    Suggest 3 better names only, with the order of preference, separated by commas.
    Do not include any additional text or formatting. Just response as "name1, name2, name3" format.
    Names only, no 'def'.
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def get_recommendation_for_function_name(function_code):
    prompt = f"""Suggest names for the given function.

    Code:
    {function_code}

    Suggest 3 names only, with the order of preference, separated by commas.
    Do not include any additional text or formatting. Just response as "name1, name2, name3" format.
    Names only, no 'def'.
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are a Python naming expert.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


def extract_names_from_recommendation(recommendation):
    result_line = recommendation.split('\n')[0]

    names = []

    for _name in result_line.split(","):
        name = _name.strip()
        if not name:
            continue
        names.append(name)

    return names


def llm_readability_score(source_code):
    """
    LLM readability score calculation.
    It would calculate the 'readability score' with local LLM such as llama-3.
    """

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
    first_line = lines[0].strip()

    # First line of response should follow 'Score: xx/100' format.
    # Returns 'xx' as points.
    # If not, this code consider it as a miss of llama-3 and ignore it.
    # In this case, we consider it as 0 point.

    match = re.search(r'Score: (\d+)/100', first_line)

    if match:
        try:
            score = int(match.group(1))
        except ValueError:
            score = 0
    else:
        score = 0

    return score


if __name__ == "__main__":
    start_time = time.time()

    context = """def hehehe(a, b, c):
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
        return True
    if temp % 3 == 0:
        return True
    return False
    
def monte_carlo_pi(num_samples):
    v = 0

    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if in_circle(x, y):
            v += 1

    return (v / num_samples) * 4
    """

    target_line_no = [0, 10, 19]

    lines = context.split('\n')
    target_lines = [lines[no].strip() for no in target_line_no]

    print("target lines are")
    for line in target_lines:
        print(line)

    recommendations = get_recommendation_for_rename(context, target_lines)

    print("Recommended names for the target line:")
    print(recommendations)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    ft_code = """def hehehe(a, b, c):
if a == 0:
    return True
if b == 0:
    return True
if c == 0:
    return True
return False"""

    start_time = time.time()

    recommendations = get_recommendation_for_function_rename(ft_code)

    print("Recommended names:")
    print(recommendations)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
