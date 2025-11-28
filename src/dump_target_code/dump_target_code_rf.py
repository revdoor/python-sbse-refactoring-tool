import random


def in_circle(x, y):
    return x * x + y * y <= 1


def hehehe(a, b, c):
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
