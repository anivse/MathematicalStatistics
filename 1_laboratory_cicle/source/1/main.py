import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

def Normal(size):
    x = sps.norm(loc=0, scale=1).rvs(size=size)
    xf = np.arange(-3, 3, 0.001)
    plt.figure()
    plt.hist(x, bins=15, color='white', edgecolor='black', density=True)
    plt.plot(xf, sps.norm.pdf(xf, 0, 1), color='blue')
    plt.title(f"NormalNumbers={size}")
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

def Cauchy(size):
    x = sps.cauchy().rvs(size=size)
    xf = np.arange(-15, 15, 0.001)
    plt.figure()
    plt.hist(x, bins=20, color='white', edgecolor='black', density=True)
    plt.plot(xf, sps.cauchy.pdf(xf, loc=0, scale=1), color='blue')
    plt.title(f"CauchyNumbers={size}")
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

def Laplace(size):
    x = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size)
    xf = np.arange(-4, 4, 0.001)
    plt.figure()
    plt.hist(x, bins=50, color='white', edgecolor='black', density=True)
    plt.plot(xf, sps.laplace.pdf(xf, 0, 1/2**0.5), color='blue')
    plt.title(f"LaplaceNumbers={size}")
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

def Poisson(size):
    x = sps.poisson(mu=10).rvs(size=size)
    xf = np.arange(sps.poisson.ppf(0.01, mu=10), sps.poisson.ppf(0.99, mu=10))
    plt.figure()
    plt.hist(x, bins=10, color='white', edgecolor='black', density=True)
    plt.plot(xf, sps.poisson.pmf(xf, mu=10), color='blue')
    plt.title(f"PoissonNumbers={size}")
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

def Uniform(size):
    x = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size)
    xf = np.arange(-2, 2, 0.001)
    plt.figure()
    plt.hist(x, bins=20, color='white', edgecolor='black', density=True)
    plt.plot(xf, sps.uniform.pdf(xf,loc=-3**0.5, scale=2*3**0.5), color='blue')
    plt.title(f"UniformNumbers={size}")
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()



Normal(10)
Normal(50)
Normal(1000)
Cauchy(10)
Cauchy(50)
Cauchy(1000)
Laplace(10)
Laplace(50)
Laplace(1000)
Poisson(10)
Poisson(50)
Poisson(1000)
Uniform(10)
Uniform(50)
Uniform(1000)

