import numpy
from scipy.optimize import fmin, least_squares, minimize


def rosenbrock(x, a, b):
    return (1-x[0]+a)**2 + 100*(x[1]-b-(x[0]-a)**2)**2


def jacobian(x, a, b):
    dx = 2 * (-200 * (a - x[0]) * (a**2 - 2*a*x[0] +b + x[0]**2 - x[1]) - a + x[0] - 1)
    dy = 200 * (-(x[0] - a)**2 - b + x[1])
    return numpy.array([dx, dy])


def get_params():
    return int(4 * numpy.random.uniform(-1, 2)) / 2, int(4 * numpy.random.uniform(-1, 2)) / 2


def get_start_point(a, b):
    return numpy.array([a + 2 * numpy.random.uniform(-1, 1), b + 2 * numpy.random.uniform(-1, 1)])


def nelder_mead(x0, a, b):
    print('Nelder-Mead optimization: ')
    print(minimize(rosenbrock, x0=x0, args=(a, b), method='Nelder-Mead'))
    print()


def powell(x0, a, b):
    print('Powell optimization: ')
    print(minimize(rosenbrock, x0=x0, args=(a, b), method='Powell'))
    print()


def newton(x0, a, b):
    print('Newton optimization: ')
    print(minimize(rosenbrock, x0=x0, args=(a, b), method='Newton-CG', jac=jacobian))
    print()


def cg(x0, a, b):
    print('CG optimization: ')
    print(minimize(rosenbrock, x0=x0, args=(a, b), method='CG', jac=jacobian))
    print()


def fmin_optim(x0, a, b):
    print('fmin optimization: ')
    print(fmin(rosenbrock, x0=x0, args=(a, b)))
    print()


def main():
    a, b = get_params()
    start_point = get_start_point(a, b)
    print("Parameters: a = {0}  b = {1}".format(a, b))
    print("Start point: x = {0}  y = {1}".format(start_point[0], start_point[1]))

    nelder_mead(start_point, a, b)
    powell(start_point, a, b)
    newton(start_point, a, b)
    cg(start_point, a, b)

if __name__ == "__main__":
    main()
    # x = numpy.array([1, 2])
