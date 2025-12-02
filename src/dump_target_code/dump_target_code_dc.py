def my_func(a, b, c):
    if a % 2 == 0 and (b % 3 == 1 or c % 3 == 1):
        return False

    v = a + b + c
    return v % 5 == 0
