def hehehe(a, b, c):
    if a == 0:
        return True
    if b == 0:
        return True
    if c == 0:
        return True
    return False


def is_zero_or_one(a):
    if a == 0:
        return True
    return False


def aa(num_str, key):
    cnt = 0
    length = len(key)

    for i in range(length):
        if key[i] == num_str[i]:
            cnt += 1

    return cnt
