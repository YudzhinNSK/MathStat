import numpy as np
from tabulate import tabulate
import scipy.stats as stats
from prettytable import PrettyTable




def print_table_(n_i,p_i,result):

    headers = ["i", "boarders", "n_i", "p_i", "np_i", "n_i - np_i", "((n_i - np_i)^2)/np_i"]
    
    table = PrettyTable(headers)
    for i in range(0, len(n_i)):
       if i == 0:
           boarders = ['-inf', np.around(boarders[0], decimals=2)]
       elif i == len(n_i) - 1:
           boarders = [np.around(boarders[-1], decimals=2), 'inf']
       else:
           boarders = [np.around(boarders[i - 1], decimals=2), np.around(boarders[i], decimals=2)]
       table.add_row([i + 1, boarders, n_i[i], np.around(p_i[i], decimals=4), np.around(p_i[i] * 100, decimals = 2), np.around(n_i[i] - 100 * p_i[i], decimals=2), np.around(result[i], decimals=2)])


    table.add_row([len(n_i), "-", np.sum(n_i), np.around(np.sum(p_i), decimals=4), np.around(np.sum(p_i * 100), decimals=2), -np.around(np.sum(n_i - 100 * p_i), decimals=2), np.around(np.sum(result), decimals=2)])

    print(table)




if __name__ == '__main__':

    distr = np.random.normal(0, 1, size=100)
    mu_n = np.mean(distr)
    sigma_n = np.std(distr)
    print('mu_n = ',np.around(mu_n, decimals=2), ' sigma_n = ', np.around(sigma_n, decimals=2),'\n')

    alpha = 0.05
    p = 1 - alpha
    k = 6

    boarders = np.linspace(-1.2, 1.2, num=k-1)
    sample = stats.chi2.ppf(p, k-1)
    p_i = np.array([stats.norm.cdf(boarders[0])])
    n_i = np.array([len(distr[distr <= boarders[0]])])

    k = len(boarders) - 1

    for i in range(0, k):
        new_ar = stats.norm.cdf(boarders[i + 1]) - stats.norm.cdf(boarders[i])
        p_i = np.append(p_i, new_ar)
        n_i = np.append(n_i, len(distr[(distr <= boarders[i + 1]) & (distr >= boarders[i])]))

    p_i = np.append(p_i, 1 - stats.norm.cdf(boarders[-1]))
    n_i = np.append(n_i, len(distr[distr >= boarders[-1]]))
    result = np.divide(np.multiply((n_i - 100 * p_i), (n_i - 100 * p_i)), p_i * 100)

    print_table_(n_i,p_i,result) 

    print('\n',len(n_i))
