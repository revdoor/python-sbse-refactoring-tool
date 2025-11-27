def simple_add(a, b):
    return a + b


def simple_multiply(a, b):
    return a * b


def simple_print(value):
    print(value)


def complex_formula(a, b, c, d):
    v1 = simple_add(a, b) - simple_multiply(c, d)
    v2 = simple_multiply(a, d) + simple_add(b, c)

    simple_print(v1)

    result = simple_add(v1, v2)

    return result


if __name__ == "__main__":
    a = 1
    b = 2
    c = 3
    d = 4

    result = complex_formula(a, b, c, d)
    print("The result of the complex formula is:", result)
