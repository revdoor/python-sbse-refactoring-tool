def my_func(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                print("They are all zero!")

    print(a, b, c)


def also_my_func(a, b, c, d):
    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    if i + j == 0:
                        if k + l == 0:
                            print("i+j and k+l are zero")


def my_3rd_func(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                print("They are all zero!")
    else:
        print("a is not zero")
