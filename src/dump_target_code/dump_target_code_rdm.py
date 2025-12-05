def my_func_1(a, b):
    v = a + b

    if v < 0:
        return "Negative"
    elif v == 0:
        return "Zero"
    else:
        return "Positive"


def my_func_2(x, y):
    sum_value = x + y

    if sum_value < 0:
        return "Negative"
    elif sum_value == 0:
        return "Zero"
    else:
        return "Positive"


def my_func_3(m, n):
    total = m + n

    if total < 0:
        return "Negative"
    elif total == 0:
        return "Zero"
    else:
        return "Positive"


def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)


def fact(val):
    return 1 if val == 0 else val * fact(val - 1)


def not_exact_fact(num):
    return 1 if num == 0 else num * fact(num - 1)


def foo():
    return bar()


def fos():
    return baz()


def bar():
    return 42


def baz():
    return 43
