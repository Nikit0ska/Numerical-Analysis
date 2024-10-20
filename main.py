import sympy as sp
import numpy as np

# Определение переменных и функций
x, y, z = sp.symbols('x y z')

f1_expr = x ** 5 - 2.1 * z ** 2 - 3 * x ** 2 * y ** 4 - 17.9
f2_expr = 0.6 * y * z ** 3 + 1.7 * x ** 2 * y ** 3 - 20.9 + 14.7
f3_expr = 5.2 * y ** 5 - 2.5 * z ** 4 * x ** 2 + 4.8

# Производные по x, y, z для всех функций
f1_dx = sp.diff(f1_expr, x)
f1_dy = sp.diff(f1_expr, y)
f1_dz = sp.diff(f1_expr, z)

f2_dx = sp.diff(f2_expr, x)
f2_dy = sp.diff(f2_expr, y)
f2_dz = sp.diff(f2_expr, z)

f3_dx = sp.diff(f3_expr, x)
f3_dy = sp.diff(f3_expr, y)
f3_dz = sp.diff(f3_expr, z)


# Функция для вычисления значений функций
def f1(x0, y0, z0):
    return f1_expr.subs([(x, x0), (y, y0), (z, z0)])


def f2(x0, y0, z0):
    return f2_expr.subs([(x, x0), (y, y0), (z, z0)])


def f3(x0, y0, z0):
    return f3_expr.subs([(x, x0), (y, y0), (z, z0)])


# Функция для вычисления Якобиана
def jacobian(x0, y0, z0):
    return np.array([
        [f1_dx.subs([(x, x0), (y, y0), (z, z0)]), f1_dy.subs([(x, x0), (y, y0), (z, z0)]),
         f1_dz.subs([(x, x0), (y, y0), (z, z0)])],
        [f2_dx.subs([(x, x0), (y, y0), (z, z0)]), f2_dy.subs([(x, x0), (y, y0), (z, z0)]),
         f2_dz.subs([(x, x0), (y, y0), (z, z0)])],
        [f3_dx.subs([(x, x0), (y, y0), (z, z0)]), f3_dy.subs([(x, x0), (y, y0), (z, z0)]),
         f3_dz.subs([(x, x0), (y, y0), (z, z0)])]
    ], dtype='float')


# Реализация метода Ньютона
def newton_method(x0, y0, z0, eps):
    xi, yi, zi = x0 + 2 * eps, y0 + 2 * eps, z0 + 2 * eps
    i = 0
    while abs(xi - x0) > eps or abs(yi - y0) > eps or abs(zi - z0) > eps:
        i += 1
        J = jacobian(x0, y0, z0)  # Якобиан (матрица производных)
        F = np.array([-f1(x0, y0, z0), -f2(x0, y0, z0), -f3(x0, y0, z0)], dtype='float')  # Значения функций

        # Решаем линейную систему для нахождения поправок
        try:
            dx, dy, dz = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            print("Ошибка: Якобиан вырожден. Метод не сходится.")
            return

        xi, yi, zi = x0, y0, z0
        x0 += dx
        y0 += dy
        z0 += dz
        print(f'Шаг {i}: x = {x0}, y = {y0}, z = {z0}')

    print('Значения функций в корне:')
    print(f'f1(x, y, z) = {f1(x0, y0, z0)}')
    print(f'f2(x, y, z) = {f2(x0, y0, z0)}')
    print(f'f3(x, y, z) = {f3(x0, y0, z0)}')
    print(f'Решение: x = {x0}, y = {y0}, z = {z0}')


# Пример использования
newton_method(2, 1, 1, 0.000001)
