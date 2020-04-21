import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt


def function_for_minimization(params, x, y):

    b_m_0, b_m_1 = params
    res = 0

    for i in range(len(x)):
        res += abs(b_m_0 * x[i] + b_m_1 - y[i])

    return res


def mnk(x, y):

    b_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    b_0 = np.mean(y) - b_1 * np.mean(x)

    return b_0, b_1


def plot(x, y, type):

    b_0, b_1 = mnk(x, y)

    print('MNK')
    print('b_0 = ' + str(np.around(b_0, decimals=2)))
    print('b_1 = ' + str(np.around(b_1, decimals=2)))

    result = opt.minimize(function_for_minimization, [b_0, b_1], args=(x, y), method='SLSQP')
    coef = result.x
    b_m_0, b_m_1 = coef[0], coef[1]


    print('MNA')
    print('B_M_0 = ' + str(np.around(b_m_0, decimals=2)))
    print('B_M_1 = ' + str(np.around(b_m_1, decimals=2)))


    plt.scatter(x[1:-2], y[1:-2], label='Выборка', edgecolor='darkblue')

    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='springgreen')
    plt.plot(x, x * (b_1 * np.ones(len(x))) + b_0 * np.ones(len(x)), label='МHK', color='red')
    plt.plot(x, x * (b_m_1 * np.ones(len(x))) + b_m_0 * np.ones(len(x)), label='МHM', color='dodgerblue')

    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim([-1.8, 2])
    plt.legend()
    plt.title(type)

    plt.savefig(type + '.png', format='png')
    plt.show()


if __name__ == '__main__':

    x = np.arange(-1.8, 2, 0.2)
    y = 2 * x + 2 * np.ones(len(x)) + np.random.normal(0, 1, size=len(x))

    first = 'Без возмущений'
    second = 'С возмущениями'

    plot(x, y, first)
    y[0] += 10
    y[-1] -= 10
    plot(x, y, second)
