import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math as m
import scipy.optimize as opt
import tabulate


alpha = 0.05
p = 1 - alpha

#метод максимального правдоподобия
def MLE(distribution):
    mu, sigma = sps.norm.fit(data = distribution)
    print('mu = ' + str(np.around(mu, decimals=2)))
    print('sigma = ' + str(np.around(sigma, decimals=2)))


#число k для оценки пирсона
def get_k(size):
    return m.ceil(1.72 * size ** (1 / 3))



def get_limits(k):

    #равномерная сетка
    limits = np.linspace(-1.1, 1.1, k-1)

    return limits


def get_n_p(distribution, limits, size):
    p_list = np.array([])
    n_list = np.array([])

    for i in range(-1, len(limits)):
        if i != -1:
            prev = sps.norm.cdf(limits[i])
        else:
            prev = 0


        if i != len(limits) - 1:
            cur = sps.norm.cdf(limits[i + 1])
        else:
            cur = 1

        p_list = np.append(p_list, cur - prev)


        if i == -1:
            n_list = np.append(n_list, len(distribution[distribution <= limits[0]]))
        elif i == len(limits) - 1:
            n_list = np.append(n_list, len(distribution[distribution >= limits[-1]]))
        else:
            n_list = np.append(n_list, len(distribution[(distribution <= limits[i + 1]) & (distribution >= limits[i])]))


    result = np.divide(np.multiply((n_list - size * p_list), (n_list - size * p_list)), p_list * size)
    return n_list, p_list, result


def print_table(n_list, p_list, result, size, limits):
    print('\\hline')
    print("$i$ & limits & $n_i$ & $p_i$ & $np_i$ & $n_i - np_i$ & $\dfrac{(n_i-np_i)^2}{np_i}$ \\\\")
    print("\\hline")
    rows = []
    for i in range(0, len(n_list)):
        if i == 0:
            boarders = ['$-\infty$', np.around(limits[0], decimals=2)]
        elif i == len(n_list) - 1:
            boarders = [np.around(limits[len(n_list) - 2], decimals=2), '$\infty$']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]

        print(f'{i + 1} & {boarders} & {n_list[i]} & {np.around(p_list[i], decimals=4)} & {np.around(p_list[i] * size, decimals=2)} & {np.around(n_list[i] - size * p_list[i], decimals=2)} & {np.around(result[i], decimals=2)} \\\\')

    print('\\hline')
    print(f'$\sum$ & - & {np.sum(n_list)} & {np.around(np.sum(p_list))} & {np.around(np.sum(p_list * size), decimals=2)} & {np.around(np.sum(n_list - size * p_list), decimals=2)} & {np.around(np.sum(result), decimals=2)}\\\\')    
    print('\\hline')


def print_result(size, distribution):
    k = get_k(size)

    MLE(distribution)

    #квантиль для хи квадрат
    chi_2 = sps.chi2.ppf(p, k - 1)
    print('chi_2 = ' + str(chi_2))

    limits = get_limits(k)

    n_list, p_list, result = get_n_p(distribution, limits, size)

    print_table(n_list, p_list, result, size, limits)


if __name__ == '__main__':
    
    x_1 = sps.norm.rvs(size=100, loc=0, scale = 1)
    x_2 = sps.laplace.rvs(size=20, scale=1 / m.sqrt(2), loc=0)
    x_3 = sps.uniform.rvs(size=20, loc=-m.sqrt(3), scale=2 * m.sqrt(3))

    print_result(100, x_1)
    print_result(20, x_2)
    print_result(20, x_3)



