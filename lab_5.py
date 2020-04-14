import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse
from prettytable import PrettyTable
import statistics
import os

size = [20, 60, 100]
rho = [0, 0.5, 0.9]


def quad_coef_cor(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    new_x = np.empty(size, dtype=float)
    new_x.fill(med_x)
    new_x = x - new_x
    new_y = np.empty(size, dtype=float)
    new_y.fill(med_y)
    new_y = y - new_y
    n = [0, 0, 0, 0]
    for i in range(size):
        if new_x[i] >= 0 and new_y[i] >= 0: n[0] += 1
        if new_x[i] < 0 and new_y[i] > 0: n[1] += 1
        if new_x[i] < 0 and new_y[i] < 0: n[2] += 1
        if new_x[i] > 0 and new_y[i] < 0: n[3] += 1
    return (n[0] + n[2] - n[1] - n[3]) / size


def find_coefficients(size_, rho):
    mean_rv = [0, 0]
    cov_rv = [[1.0, rho], [rho, 1.0]]
    p_coef = np.empty(1000, dtype=float)
    s_coef = np.empty(1000, dtype=float)
    q_coef = np.empty(1000, dtype=float)
    for i in range(1000):
        rv = stats.multivariate_normal.rvs(mean_rv, cov_rv, size=size_)
        x = rv[:, 0]
        y = rv[:, 1]
        p_coef[i], t = stats.pearsonr(x, y)
        s_coef[i], t = stats.spearmanr(x, y)
        q_coef[i] = quad_coef_cor(x, y)
    return p_coef, s_coef, q_coef

def ellipse_data(x, y, ax, n_std=3.0, facecolor='none', **kw_args):

    cov = np.cov(x, y)
    p_coef = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ellipse_radius_x = np.sqrt(1 + p_coef)
    ellipse_radius_y = np.sqrt(1 - p_coef)
    ellipse = Ellipse((0, 0), width=ellipse_radius_x * 2, height=ellipse_radius_y * 2, facecolor=facecolor, **kw_args)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transform = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transform + ax.transData)

    return ax.add_patch(ellipse)


def plot_range(size):
    mean = [0, 0]
    figure, ax = plt.subplots(1, 3)
    figure.suptitle("n = " + str(size))
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    num = 0
    for r in rho:
        cov = [[1.0, r], [r, 1.0]]
        rv = stats.multivariate_normal.rvs(mean, cov, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[num].scatter(x, y, s=3)
        ellipse_data(x, y, ax[num], edgecolor='black')
        ax[num].scatter(np.mean(x), np.mean(y), c='red', s=3)
        ax[num].set_title(titles[num])
        num += 1
    plt.savefig("n" + str(size) + ".png", format='png')
    plt.show()


def table_data(pearson_coef, spearman_coef, quadrant_coef, rho, size):
    header = []
    
    if rho != -1:
        header = ["rho = " + str(rho), 'r', 'r_S', 'r_Q']
    else:
        header = ["n = " + str(size), 'r', 'r_S', 'r_Q']

    table = PrettyTable(header)

    p = np.median(pearson_coef)
    s = np.median(spearman_coef)
    q = np.median(quadrant_coef)
    table.add_row(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pearson_coef[k] ** 2 for k in range(1000)])
    s = np.median([spearman_coef[k] ** 2 for k in range(1000)])
    q = np.median([quadrant_coef[k] ** 2 for k in range(1000)])
    table.add_row(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson_coef)
    s = statistics.variance(spearman_coef)
    q = statistics.variance(quadrant_coef)
    table.add_row(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    print(table)
    print('\n')

if __name__ == '__main__':
    os.system('color f0')
    for j in size:
        for i in rho:
            p_coef, s_coef, q_coef =  find_coefficients(j, i)
            table_data(p_coef, s_coef, q_coef, i, j)
        p_coef = np.empty(1000, dtype=float)
        s_coef = np.empty(1000, dtype=float)
        q_coef = np.empty(1000, dtype=float)
        for k in range(1000):
            rv = []
            for l in range(2):
                x = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], j) + 0.1 * stats. multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], j)
                rv += list(x)
            rv = np.array(rv)
            x = rv[:, 0]
            y = rv[:, 1]
            p_coef[k], t = stats.pearsonr(x, y)
            s_coef[k], t = stats.spearmanr(x, y)
            q_coef[k] = quad_coef_cor(x, y)
        table_data(p_coef, s_coef, q_coef, -1, j)
        plot_range(j)
