import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt 
import seaborn as sns
import math

def count_of_emissions(x):
    n = len(x)
    Q1 = np.quantile(x, 0.25)
    Q3 = np.quantile(x, 0.75)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    count = 0 
    for i in range(0, n):
        if (x[i] < X1 or x[i] > X2):
            count += 1
    return count 

def normal_boxplot():
    size_1 = 20
    size_2 = 100
    x_1 = sps.norm(loc=0, scale=1).rvs(size=size_1)
    x_2 = sps.norm(loc=0, scale=1).rvs(size=size_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Normal")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()

def normal_share_of_emissions():
    size_1 = 20
    size_2 = 100
    repeat = 1000
    count_1 = 0
    count_2 = 0
    for i in range (0, repeat):
        x_1 = sps.norm(loc=0, scale=1).rvs(size=size_1)
        x_2 = sps.norm(loc=0, scale=1).rvs(size=size_2)
        x_1.sort()
        x_2.sort()
        count_1 += count_of_emissions(x_1)
        count_2 += count_of_emissions(x_2)
    
    x_1_share = count_1 / (size_1 * repeat)
    x_2_share = count_2 / (size_2 * repeat)

    print("Normal: ")
    print(f'20: {x_1_share}')
    print(f'100: {x_2_share}')
    
def cauchy_share_of_emissions():
    size_1 = 20
    size_2 = 100
    repeat = 1000
    count_1 = 0
    count_2 = 0
    for i in range (0, repeat):
        x_1 = sps.cauchy().rvs(size=size_1)
        x_2 = sps.cauchy().rvs(size=size_2)
        x_1.sort()
        x_2.sort()
        count_1 += count_of_emissions(x_1)
        count_2 += count_of_emissions(x_2)
    
    x_1_share = count_1 / (size_1 * repeat)
    x_2_share = count_2 / (size_2 * repeat)

    print("Cauchy: ")
    print(f'20: {x_1_share}')
    print(f'100: {x_2_share}')

def cauchy_boxplot():
    size_1 = 20
    size_2 = 100
    x_1 = sps.cauchy().rvs(size=size_1)
    x_2 = sps.cauchy().rvs(size=size_2)
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Cauchy")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()


def laplace_share_of_emissions():
    size_1 = 20
    size_2 = 100
    repeat = 1000
    count_1 = 0
    count_2 = 0
   
    for i in range (0, repeat):
        x_1 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_1)
        x_2 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_2)
        x_1.sort()
        x_2.sort()
        count_1 += count_of_emissions(x_1)
        count_2 += count_of_emissions(x_2)
    
    x_1_share = count_1 / (size_1 * repeat)
    x_2_share = count_2 / (size_2 * repeat)
    
    print("Laplace: ")
    print(f'20: {x_1_share}')
    print(f'100: {x_2_share}')


def laplace_boxplot():
    size_1 = 20
    size_2 = 100
    x_1 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_1)
    x_2 = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Laplace")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()

def poisson_share_of_emissions():
    size_1 = 20
    size_2 = 100
    repeat = 1000
    count_1 = 0
    count_2 = 0
    
    for i in range (0, repeat):
        x_1 = sps.poisson(mu=10).rvs(size=size_1)
        x_2 = sps.poisson(mu=10).rvs(size=size_2)
        x_1.sort()
        x_2.sort()
        count_1 += count_of_emissions(x_1)
        count_2 += count_of_emissions(x_2)
    
    x_1_share = count_1 / (size_1 * repeat)
    x_2_share = count_2 / (size_2 * repeat)

    print("Poisson: ")
    print(f'20: {x_1_share}')
    print(f'100: {x_2_share}')

def poisson_boxplot():
    size_1 = 20
    size_2 = 100
    x_1 = sps.poisson(mu=10).rvs(size=size_1)
    x_2 = sps.poisson(mu=10).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Poisson")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()

def uniform_share_of_emissions():
    size_1 = 20
    size_2 = 100
    repeat = 1000
    count_1 = 0
    count_2 = 0

    for i in range (0, repeat):
        x_1 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_1)
        x_2 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_2)
        x_1.sort()
        x_2.sort()
        count_1 += count_of_emissions(x_1)
        count_2 += count_of_emissions(x_2)
    
    x_1_share = count_1 / (size_1 * repeat)
    x_2_share = count_2 / (size_2 * repeat)

    print("Uniform: ")
    print(f'20: {x_1_share}')
    print(f'100: {x_2_share}')

def uniform_boxplot():
    size_1 = 20
    size_2 = 100
    x_1 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_1)
    x_2 = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size_2)
    x_1.sort()
    x_2.sort()
    plt.figure()
    plt.boxplot(x = [x_1, x_2], vert = False, labels = [20, 100])
    plt.title("Uniform")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()

# normal_boxplot()
# cauchy_boxplot()
# laplace_boxplot()
# poisson_boxplot()
# uniform_boxplot()


normal_share_of_emissions()
cauchy_share_of_emissions()
laplace_share_of_emissions()
poisson_share_of_emissions()
uniform_share_of_emissions()