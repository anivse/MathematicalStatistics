import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns

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
    plt.close()
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
    plt.close()
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
    plt.close()
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
    plt.close()
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
    plt.close()
    #plt.show()

def normal_kde(size, adjust, left, right):
    dist = sps.norm(loc=0, scale=1).rvs(size=size)
    dist.sort()
    x = np.linspace(start=left, stop=right, num=10000)
    plt.figure()
    plt.plot(x, sps.norm.pdf(x), color='red', linewidth=1)
    sns.kdeplot(data = dist, color = 'black', bw_method = 'silverman', bw_adjust=adjust, linewidth=1)
    plt.xlim(left, right)
    plt.xlabel("x")
    plt.title(f'Normal n={size}, adjust = {adjust}')
    plt.savefig(f'normal_n{size}_adjust{adjust}.png')
    plt.close()
    #plt.show()
    
def cauchy_kde(size, adjust, left, right):
    dist = sps.cauchy.rvs(size=size)
    dist.sort()
    x = np.linspace(start=left, stop=right, num=10000)
    plt.figure()
    plt.plot(x, sps.cauchy.pdf(x), color='red', linewidth=1)
    sns.kdeplot(data = dist, color = 'black', bw_method = 'silverman', bw_adjust=adjust, linewidth=1)
    plt.xlim(left, right)
    plt.xlabel("x")
    plt.title(f'Cauchy n={size}, adjust = {adjust}')
    plt.savefig(f'cauchy_n{size}_adjust{adjust}.png')
    plt.close()
    #plt.show()

def laplace_kde(size, adjust, left, right):
    dist = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size)
    dist.sort()
    x = np.linspace(start=left, stop=right, num=10000)
    plt.figure()
    plt.plot(x, sps.laplace(loc=0, scale=1/2**0.5).pdf(x), color='red', linewidth=1)
    sns.kdeplot(data = dist, color = 'black', bw_method = 'silverman', bw_adjust=adjust, linewidth=1)
    plt.xlim(left, right)
    plt.xlabel("x")
    plt.title(f'Laplace n={size}, adjust = {adjust}')
    plt.savefig(f'laplace_n{size}_adjust{adjust}.png')
    plt.close()
    #plt.show()

def poisson_kde(size, adjust, left, right):
    dist = sps.poisson(mu=10).rvs(size=size)
    dist.sort()
    x = np.arange(sps.poisson.ppf(0.001, mu=10), sps.poisson.ppf(0.999, mu=10))
    plt.figure()
    plt.plot(x, sps.poisson(mu=10).pmf(x), color='red', linewidth=1)
    sns.kdeplot(data = dist, color = 'black', bw_method = 'silverman', bw_adjust=adjust, linewidth=1)
    plt.xlim(left, right)
    plt.xlabel("x")
    plt.title(f'Poisson n={size}, adjust = {adjust}')
    plt.savefig(f'poisson_n{size}_adjust{adjust}.png')
    plt.close()
    #plt.show()

def uniform_kde(size, adjust, left, right):
    dist = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size)
    dist.sort()
    x = np.linspace(start=left, stop=right, num=10000)
    plt.figure()
    plt.plot(x, sps.uniform(loc=-3**0.5, scale=2*3**0.5).pdf(x), color='red', linewidth=1)
    sns.kdeplot(data = dist, color = 'black', bw_method = 'silverman', bw_adjust=adjust, linewidth=1)
    plt.xlim(left, right)
    plt.xlabel("x")
    plt.title(f'Uniform n={size}, adjust = {adjust}')
    plt.savefig(f'uniform_n{size}_adjust{adjust}.png')
    plt.close()
    #plt.show()

# normal_distribution_function(20, -4, 4)
# normal_distribution_function(60, -4, 4)
# normal_distribution_function(100, -4, 4)
# cauchy_distribution_function(20, -4, 4)
# cauchy_distribution_function(60, -4, 4)
# cauchy_distribution_function(100, -4, 4)
# laplace_distribution_function(20, -4, 4)
# laplace_distribution_function(60, -4, 4)
# laplace_distribution_function(100, -4, 4)
# poisson_distribution_function(20, 6, 14) 
# poisson_distribution_function(60, 6, 14)   
# poisson_distribution_function(100, 6, 14)   
# uniform_distribution_function(20, -4, 4)   
# uniform_distribution_function(60, -4, 4)   
# uniform_distribution_function(100, -4, 4)   


adjusts = [0.5, 1, 2]
sizes = [20, 60, 100]  

for adjust in adjusts:
    for size in sizes:
        normal_kde(size, adjust, -4, 4)
        cauchy_kde(size, adjust, -4, 4)
        laplace_kde(size, adjust, -4, 4)
        poisson_kde(size, adjust, 6, 14)
        uniform_kde(size, adjust, -4, 4)



    

