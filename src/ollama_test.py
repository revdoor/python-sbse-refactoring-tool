import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': '안녕하세요!'}
    ]
)

print(response['message']['content'])
