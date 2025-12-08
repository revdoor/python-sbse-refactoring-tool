import re
import ollama


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


def get_recommendation_for_decompose_conditional(conditional_code):
    prompt = f"""Suggest names for the given conditional, which would be extracted as a method.

    Code:
    {conditional_code}

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
