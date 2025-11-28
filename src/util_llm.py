import ollama
import time


def get_recommendations_for_rename(context_code, target_lines):
    prompt = f"""The assistant is a seasoned senior software engineer, with deep Python Language expertise.
With the given context, your task is suggesting better names for the target lines (which is included in the context)
If the line is function definition, suggest better function names, and if the line is a variable assignment, suggest better variable names.
(Assume that the target lines are either function definitions or variable assignments only.)

Context Code:
{context_code}

The target lines are:
{target_lines}

In each line, suggest 3 better names only, with the order of preference, separated by commas.
Do not include any additional text or formatting. Just response as "line_no: name1, name2, name3 (original line)" format.
Do not print 'def' or '=' in your response. (Just the name only.)
"""

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']


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

    recommendations = get_recommendations_for_rename(context, target_lines)

    print("Recommended names for the target line:")
    print(recommendations)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
