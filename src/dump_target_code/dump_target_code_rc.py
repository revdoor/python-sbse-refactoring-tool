def some_complex_function(a, b, c, d, e):
    if a == 0:
        print("a is zero")
    elif b == 0:
        print("b is zero")
    else:
        if a == 1:
            print("a is one")
        else:
            if c == 0:
                print("c is zero")
            elif d == 0 and e == 0:
                print("d and e are zero")
            else:
                if b == 1 or c == 1:
                    print("either b or c is one")
                elif d + e == 0:
                    print("d and e sum to zero")
                else:
                    print("no conditions met")
