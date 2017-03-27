import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib
if sys.platform == 'linux':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def rosenbrock(x, a, b):
    return (1 - x[0]+a)**2 + 100*(x[1]-b-(x[0]-a)**2)**2


def jacobian(x, a, b):
    dx = 2 * (-200 * (a - x[0]) * (a**2 - 2*a*x[0] + b + x[0]**2 - x[1]) - a + x[0] - 1)
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


def optimize(method, x0, a, b):
    callback = [x0]
    print('Newton optimization: ')
    optim = minimize(rosenbrock, x0=x0, args=(a, b), method=method, jac=jacobian, hess=hess,
                     callback=callback.append)
    print(str(optim) + '\n')
    return np.transpose(callback)


def plot_optim(params, start_point, method_name, filename):
    res = optimize(method_name, start_point, *params)
    x = np.arange(res[0, -1] - 4, res[0, -1] + 2, 0.05)
    y = np.arange(res[1, -1] - 4, res[1, -1] + 2, 0.05)
    x, y = np.meshgrid(x, y)
    r = rosenbrock((x, y), *params)
    levels = np.arange(-100.0, 2000, 1.0)

    fig = plt.figure().gca()
    # fig.suptitle(method_name)
    plt.subplot(211)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contour(x, y, r, levels=levels)
    plt.text(x[0, 0] + 0.3, y[0, 0] + 0.3, 'Punkt startowy: ({:.5f}, {:.5f})\nPunkt końcowy: ({:.5f}, {:.5f})'
             .format(*(np.append(start_point, res[:, -1]))))
    plt.plot(res[0], res[1], color='r', label='Przebieg optymalizacji')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.xlabel('iter')
    plt.ylabel('log(rosenbrock(iter))')
    plt.semilogy(rosenbrock(res, *params))
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('doc/wykresy/' + filename + '.svg', format='svg', dpi=1200)


def main():
    params = get_params()
    print("Parameters: a = {0}  b = {1}".format(*params))
    plt.ion()

    for i in range(4):
        start_point = get_start_point(*params)
        print("Start point: x = {}  y = {}\n".format(start_point[0], start_point[1]))
        plot_optim(params, start_point, 'Nelder-Mead', 'nelder-mead_' + str(i))
        plot_optim(params, start_point, 'Powell', 'powell_' + str(i))
        plot_optim(params, start_point, 'CG', 'cg_' + str(i))
        plot_optim(params, start_point, 'Newton-CG', 'newton_' + str(i))
    # plt.show(block=True)


if __name__ == "__main__":
    main()
