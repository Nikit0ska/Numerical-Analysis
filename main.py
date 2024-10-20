def phi(x):
    return (0.37 * x ** 5 - 0.86 * x ** 4 - 0.72 * x ** 3 + 2.7 * x ** 2 - 10.9) / 8.3


def simple_iteration(x0, epsilon):
    xn_2 = x0
    xn_1 = phi(xn_2)
    xn = phi(xn_1)
    iters = 1
    while (((xn - xn_1) ** 2) / abs(2 * xn_1 - xn - xn_2)) >= epsilon:
        xn_1 = xn
        xn = phi(xn_1)
        iters += 1

    return xn_1, iters


print(simple_iteration(0.0, 10 ** -29))

