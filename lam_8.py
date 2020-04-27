import numpy as np
import scipy.stats as stats
from prettytable import PrettyTable


def confidence_interval_m_(m,s,n,gamma):
    interval = s * stats.t.ppf((1 + gamma) / 2, n - 1) / (n - 1) ** 0.5
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def confidence_interval_var_(m,s,n,gamma):
    low_boarder = s * (n / stats.chi2.ppf((1 + gamma) / 2, n - 1)) ** 0.5
    high_boarder = s * (n / stats.chi2.ppf((1 - gamma) / 2, n - 1)) ** 0.5
    return np.around(low_boarder, decimals=2), np.around(high_boarder, decimals=2)


def confidence_asimpt_m_(m,s,n,gamma):
    u = stats.norm.ppf((1 + gamma) / 2)
    interval = s * u / (n ** 0.5)
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def confidence_asimpt_var_(m,s,n,gamma):
    moment_4 = stats.moment(distr, 4)
    e_ = moment_4 / s**4 - 3
    u = stats.norm.ppf((1 + gamma) / 2)
    U = u * (((e_ + 2) / n) ** 0.5)
    low_boarder = s * (1 + 0.5 * U) ** (-0.5)
    high_boarder = s * (1 - 0.5 * U) ** (-0.5)
    return np.around(low_boarder, decimals=2), np.around(high_boarder, decimals=2)


if __name__ == '__main__':

    size = [20, 100]

    gamma = 0.95

    mean_ =[]
    variance_ = []
    asimpt_mean_ = []
    asimpt_variance_ = []
    size_ = []

    headers = [" "," n = 20 ", " n = 100 "]
    table = PrettyTable(headers)
    for s in size:
        distr = np.random.normal(0, 1, size=s)
        mean = np.mean(distr)
        s = np.std(distr)
        n = len(distr)
        mean_.append(confidence_interval_m_(mean,s,n,gamma))
        variance_.append(confidence_interval_var_(mean,s,n,gamma))
        asimpt_mean_.append(confidence_asimpt_m_(mean,s,n,gamma))
        asimpt_variance_.append(confidence_asimpt_var_(mean,s,n,gamma))

    table.add_row(["mean",mean_[0],mean_[1]])
    table.add_row(["variance",variance_[0],variance_[1]])
    table.add_row(["asimpt_mean",asimpt_mean_[0],asimpt_mean_[1]])
    table.add_row(["asimpt_variance",asimpt_variance_[0],asimpt_variance_[1]])
    print(table)
