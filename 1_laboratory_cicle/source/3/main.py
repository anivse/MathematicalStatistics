import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt 
import seaborn as sns
import math

def share_of_emissions(x):
    n = len(x)
    p1 = 0.25
    p2 = 0.75
    Q1 = x[math.ceil(n*p1)-1]
    Q3 = x[math.ceil(n*p2)-1]
    X1 = Q3 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    count = 0 
    for i in range(0, n):
        if ((x[i] < X1) or (x[i] > X2)):
            count += 1
    return round((count / n), 2)


def Normal():
    size_1 = 20
    size_2 = 100
    x_1 = sps.norm(loc=0, scale=1).rvs(size=size_1)
    x_2 = sps.norm(loc=0, scale=1).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    x_1_em = share_of_emissions(x_1)
    x_2_em = share_of_emissions(x_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Normal")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    print("Normal: ")
    print(f'20: {x_1_em}')
    print(f'100: {x_2_em}')
    
def Cauchy():
    size_1 = 20
    size_2 = 100
    x_1 = sps.cauchy().rvs(size=size_1)
    x_2 = sps.cauchy().rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    x_1_em = share_of_emissions(x_1)
    x_2_em = share_of_emissions(x_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Cauchy")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    print("Cauchy: ")
    print(f'20: {x_1_em}')
    print(f'100: {x_2_em}')


def Laplace():
    size_1 = 20
    size_2 = 100
    x_1 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_1)
    x_2 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    x_1_em = share_of_emissions(x_1)
    x_2_em = share_of_emissions(x_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Laplace")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    print("Laplace: ")
    print(f'20: {x_1_em}')
    print(f'100: {x_2_em}')


def Poisson():
    size_1 = 20
    size_2 = 100
    x_1 = sps.poisson(mu=10).rvs(size=size_1)
    x_2 = sps.poisson(mu=10).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    x_1_em = share_of_emissions(x_1)
    x_2_em = share_of_emissions(x_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Poisson")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    print("Poisson: ")
    print(f'20: {x_1_em}')
    print(f'100: {x_2_em}')


def Uniform():
    size_1 = 20
    size_2 = 100
    x_1 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_1)
    x_2 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    x_1_em = share_of_emissions(x_1)
    x_2_em = share_of_emissions(x_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Uniform")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    print("Uniform: ")
    print(f'20: {x_1_em}')
    print(f'100: {x_2_em}')

Normal()
Cauchy()
Laplace()
Poisson()
Uniform()