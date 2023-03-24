import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

def statistical_series(x):
    x.sort()
    z = []
    n = []
    z.append(x[0])
    temp = x[0]
    n.append(1)
    k = 0
    for i in range(1, len(x)):
        if x[i] == temp:
            n[k] += 1
        else: 
            temp = x[i]
            z.append(x[i])
            n.append(1)
            k += 1
    return z, n

def filter_func(x, value):
    if x < value:
        return True
    else: 
        return False


def empirical_distribution_function_in_point(z, n, size, x):
    result = 0
    i = 0

    for value in z:
        if value >= x:
            return result / size
        else:
            result += n[i]
            i += 1
    
    return result / size


def normal_distribution_function(size, left, right):
    dist = sps.norm(loc=0, scale=1).rvs(size=size)
    z, n = statistical_series(dist)
    x = np.linspace(start=left, stop=right, num=10000)
    y_em = []
    for value in x: 
        y_em.append(empirical_distribution_function_in_point(z, n, size, value))

    plt.figure()
    plt.plot(x, sps.norm.cdf(x), color='blue', linewidth=1)
    plt.plot(x, y_em, color='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Normal n = {size}')
    plt.savefig(f'normal_F{size}.png')
    #plt.show()

def cauchy_distribution_function(size, left, right):
    dist = sps.cauchy().rvs(size=size)
    z, n = statistical_series(dist)
    x = np.linspace(start=left, stop=right, num=10000)
    y_em = []
    for value in x: 
        y_em.append(empirical_distribution_function_in_point(z, n, size, value))

    plt.figure()
    plt.plot(x, sps.cauchy.cdf(x), color='blue', linewidth=1)
    plt.plot(x, y_em, color='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Cauchy n = {size}')
    plt.savefig(f'cauchy_F{size}.png')
    #plt.show()

def laplace_distribution_function(size, left, right):
    dist = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size)
    z, n = statistical_series(dist)
    x = np.linspace(start=left, stop=right, num=10000)
    y_em = []
    for value in x: 
        y_em.append(empirical_distribution_function_in_point(z, n, size, value))

    plt.figure()
    plt.plot(x, sps.laplace(loc=0, scale=1/2**0.5).cdf(x), color='blue', linewidth=1)
    plt.plot(x, y_em, color='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Laplace n = {size}')
    plt.savefig(f'laplace_F{size}.png')
    #plt.show()

def poisson_distribution_function(size, left, right):
    dist = sps.poisson(mu=10).rvs(size=size)
    z, n = statistical_series(dist)
    x = np.linspace(start=left, stop=right, num=10000)
    y_em = []
    for value in x: 
        y_em.append(empirical_distribution_function_in_point(z, n, size, value))

    plt.figure()
    plt.plot(x, sps.poisson(mu=10).cdf(x), color='blue', linewidth=1)
    plt.plot(x, y_em, color='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Poisson n = {size}')
    plt.savefig(f'poisson_F{size}.png')
    #plt.show()

def uniform_distribution_function(size, left, right):
    dist = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size)
    z, n = statistical_series(dist)
    x = np.linspace(start=left, stop=right, num=10000)
    y_em = []
    for value in x: 
        y_em.append(empirical_distribution_function_in_point(z, n, size, value))

    plt.figure()
    plt.plot(x, sps.uniform(loc=-3**0.5, scale=2*3**0.5).cdf(x), color='blue', linewidth=1)
    plt.plot(x, y_em, color='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Uniform n = {size}')
    plt.savefig(f'uniform_F{size}.png')
    #plt.show()



normal_distribution_function(20, -4, 4)
normal_distribution_function(60, -4, 4)
normal_distribution_function(100, -4, 4)
cauchy_distribution_function(20, -4, 4)
cauchy_distribution_function(60, -4, 4)
cauchy_distribution_function(100, -4, 4)
laplace_distribution_function(20, -4, 4)
laplace_distribution_function(60, -4, 4)
laplace_distribution_function(100, -4, 4)
poisson_distribution_function(20, 6, 14) 
poisson_distribution_function(60, 6, 14)   
poisson_distribution_function(100, 6, 14)   
uniform_distribution_function(20, -4, 4)   
uniform_distribution_function(60, -4, 4)   
uniform_distribution_function(100, -4, 4)   


    
    

