import random
import string


def get_random_name(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))
