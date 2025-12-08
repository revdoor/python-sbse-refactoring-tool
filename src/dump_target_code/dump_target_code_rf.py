# # Before
def outer(x):
    y = x + 1
    
    def inner(x):  # 내부 함수에서 같은 이름 'x' 사용
        return x * 2
    
    result = inner(y)
    return result + x

def process(items):
    x = 10
    result = [x * 2 for x in items]  # comprehension 내부의 x는 별도 scope
    return x  # 여전히 10