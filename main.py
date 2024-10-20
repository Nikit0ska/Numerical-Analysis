import sympy as sp
import numpy as np


def f1(x, y, z):
    return x ** 5 - 2.1 * z ** 2 - 3 * x ** 2 * y ** 4 - 17.9


def f2(x, y, z):
    return 0.6 * y * z ** 3 + 1.7 * x ** 2 * y ** 3 - 20.9 + 14.7


def f3(x, y, z):
    return 5.2 * y ** 5 - 2.5 * z ** 4 * x ** 2 + 4.8


def f1dx(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(x ** 5 - 2.1 * z ** 2 - 3 * x ** 2 * y ** 4 - 17.9, x).subs([(x, x0), (y, y0), (z, z0)])


def f2dx(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(0.6 * y * z ** 3 + 1.7 * x ** 2 * y ** 3 - 20.9 + 14.7, x).subs([(x, x0), (y, y0), (z, z0)])


def f3dx(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(5.2 * y ** 5 - 2.5 * z ** 4 * x ** 2 + 4.8, x).subs([(x, x0), (y, y0), (z, z0)])


def f1dy(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(x ** 5 - 2.1 * z ** 2 - 3 * x ** 2 * y ** 4 - 17.9, y).subs([(x, x0), (y, y0), (z, z0)])


def f2dy(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(0.6 * y * z ** 3 + 1.7 * x ** 2 * y ** 3 - 20.9 + 14.7, y).subs([(x, x0), (y, y0), (z, z0)])


def f3dy(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(5.2 * y ** 5 - 2.5 * z ** 4 * x ** 2 + 4.8, y).subs([(x, x0), (y, y0), (z, z0)])


def f1dz(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(x ** 5 - 2.1 * z ** 2 - 3 * x ** 2 * y ** 4 - 17.9, z).subs([(x, x0), (y, y0), (z, z0)])


def f2dz(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(0.6 * y * z ** 3 + 1.7 * x ** 2 * y ** 3 - 20.9 + 14.7, z).subs([(x, x0), (y, y0), (z, z0)])


def f3dz(x0, y0, z0):
    x, y, z = sp.symbols('x y z')
    return sp.diff(5.2 * y ** 5 - 2.5 * z ** 4 * x ** 2 + 4.8, z).subs([(x, x0), (y, y0), (z, z0)])


def Nwt(x0, y0, z0, eps):
    xi = x0 + 2 * eps
    yi = y0 + 2 * eps
    zi = z0 + 2 * eps
    i = 0
    while abs(xi - x0) > eps or abs(yi - y0) > eps or abs(zi - z0) > eps:
        i += 1
        W = np.array(((f1dx(x0, y0, z0), f1dy(x0, y0, z0), f1dz(x0, y0, z0)),
                      (f2dx(x0, y0, z0), f2dy(x0, y0, z0), f2dz(x0, y0, z0)),
                      (f3dx(x0, y0, z0), f3dy(x0, y0, z0), f3dz(x0, y0, z0))), dtype='float')
        F = np.array((-f1(x0, y0, z0), -f2(x0, y0, z0), -f3(x0, y0, z0)), dtype='float')
        dx, dy, dz = np.linalg.solve(W, F)
        xi = x0
        yi = y0
        zi = z0
        x0 += dx
        y0 += dy
        z0 += dz
        print('Шаг №', i, ': x = ', x0, ', y = ', y0, ',z = ', z0, sep='')
    print('Значения функций:')
    print('f1(x,y,z) = ', f1(x0, y0, z0))
    print('f2(x,y,z) = ', f2(x0, y0, z0))
    print('f3(x,y,z) = ', f3(x0, y0, z0))
    print('x = ', x0)
    print('y = ', y0)
    print('z = ', z0)


Nwt(2, 1, 1, 0.0001)
