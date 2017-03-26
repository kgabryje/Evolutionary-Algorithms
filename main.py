import sys
import matplotlib
if sys.platform == 'linux':
    matplotlib.use('TkAgg')

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math


def rosenbrock(x, a, b):
    return (1 - x[0]+a)**2 + 100*(x[1]-b-(x[0]-a)**2)**2


def jacobian(x, a, b):
    dx = 2 * (-200 * (a - x[0]) * (a**2 - 2*a*x[0] +b + x[0]**2 - x[1]) - a + x[0] - 1)
    dy = 200 * (-(x[0] - a)**2 - b + x[1])
    return np.array([dx, dy])


def hess(x, a, b):
    dxx = 2 * (600 * a**2 - 1200*a*x[0] + 200*b + 600 * x[0]**2 - 200 * x[1] + 1)
    dxy = 400 * (a - x[0])
    dyy = 200
    return np.array([[dxx, dxy], [dxy, dyy]])


def get_params():
    return int(4 * np.random.uniform(-1, 1)) / 2, int(4 * np.random.uniform(-1, 1)) / 2


def get_start_point(a, b):
    return np.array([a + 2 * np.random.uniform(-1, 1), b + 2 * np.random.uniform(-1, 1)])


def nelder_mead(x0, a, b):
    callback = []
    print('Nelder-Mead optimization: ')
    optim = minimize(rosenbrock, x0=x0, args=(a, b), method='Nelder-Mead', callback=callback.append)
    print(str(optim) + '\n')
    return callback


def powell(x0, a, b):
    callback = []
    print('Powell optimization: ')
    optim = minimize(rosenbrock, x0=x0, args=(a, b), method='Powell', callback=callback.append)
    print(str(optim) + '\n')
    return callback


def newton(x0, a, b):
    callback = []
    print('Newton optimization: ')
    optim = minimize(rosenbrock, x0=x0, args=(a, b), method='Newton-CG', jac=jacobian, hess=hess,
                     callback=callback.append)
    print(str(optim) + '\n')
    return callback


def cg(x0, a, b):
    callback = []
    print('CG optimization: ')
    optim = minimize(rosenbrock, x0=x0, args=(a, b), method='CG', jac=jacobian, callback=callback.append)
    print(str(optim) + '\n')
    return callback


def plot_optim(res, params, start_point, method_name):
    # x = np.arange(abs(params[0]) + abs(start_point[0]) - 2, abs(params[0]) + abs(start_point[0]) + 2, 0.05)
    # y = np.arange(abs(params[1]) + abs(start_point[1]) - 2, abs(params[1]) + abs(start_point[1]) + 2, 0.05)
    x = np.arange(min(params[0], start_point[0]) - 2, max(params[0], start_point[0]) + 2, 0.05)
    y = np.arange(min(params[1], start_point[1]) - 2, max(params[1], start_point[1]) + 2, 0.05)
    x, y = np.meshgrid(x, y)
    r = rosenbrock((x, y), *params)
    levels = np.arange(-100.0, 1000, 1.0)

    fig = plt.figure()
    fig.suptitle(method_name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contour(x, y, r, levels=levels)
    plt.text(x[0, 0] + 0.3, y[0, 0] + 0.3, 'Punkt startowy: ({:.5f}, {:.5f})\nPunkt optymalny: ({:.5f}, {:.5f})'
             .format(*(np.append(start_point, res[:, -1]))))
    plt.plot(res[0], res[1], color='r', label='Przebieg optymalizacji')
    plt.legend(loc='best')


def main():
    params = get_params()
    print("Parameters: a = {0}  b = {1}".format(*params))
    plt.ion()
    for i in range(4):
        start_point = get_start_point(*params)
        print("Start point: x = {0}  y = {1}\n".format(start_point[0], start_point[1]))

        res = nelder_mead(start_point, *params)
        powell(start_point, *params)
        newton(start_point, *params)
        cg(start_point, *params)
        res = np.transpose(res)

        plot_optim(res, params, start_point, 'Metoda Neldera-Meada')
    plt.show(block=True)
if __name__ == "__main__":
    main()
