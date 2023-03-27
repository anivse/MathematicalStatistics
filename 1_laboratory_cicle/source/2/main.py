import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import characteristics as c


def f(e, d):
    for i in range(len(e)):
        print(f'& [%.5f;%.5f] ' % ((e[i] - np.sqrt(d[i]), e[i] + np.sqrt(d[i]))), end = ' ')


def Normal(size):
    means, medians, zRs, zQs, ztrs = [], [], [], [], []
    for i in range(1000):
        x = sps.norm(loc=0, scale=1).rvs(size=size)
        x.sort()
        means.append(c.mean(x))
        medians.append(c.median(x))
        zRs.append(c.zR(x))
        zQs.append(c.zQ(x))
        ztrs.append(c.ztr(x))

    print(f'Normal {size}')
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)))
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)))
    e = [c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)]
    d = [c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)]
    f(e, d)
    print('\\\\')


        

def Cauchy(size):
    means, medians, zRs, zQs, ztrs = [], [], [], [], []
    for i in range(1000):
        x = sps.cauchy().rvs(size=size)
        x.sort()
        means.append(c.mean(x))
        medians.append(c.median(x))
        zRs.append(c.zR(x))
        zQs.append(c.zQ(x))
        ztrs.append(c.ztr(x))
    
    print(f'Cauchy {size}')
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)))
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)))
    e = [c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)]
    d = [c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)]
    f(e, d)
    print('\\\\')


def Laplace(size):
    means, medians, zRs, zQs, ztrs = [], [], [], [], []
    for i in range(1000):
        x = sps.laplace(loc=0, scale=1/2**0.5).rvs(size=size)
        x.sort()
        means.append(c.mean(x))
        medians.append(c.median(x))
        zRs.append(c.zR(x))
        zQs.append(c.zQ(x))
        ztrs.append(c.ztr(x))
    
    print(f'Laplace {size}')
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)))
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)))
    e = [c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)]
    d = [c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)]
    f(e, d)
    print('\\\\')


def Poisson(size):
    means, medians, zRs, zQs, ztrs = [], [], [], [], []
    for i in range(1000):
        x = sps.poisson(mu=10).rvs(size=size)
        x.sort()
        means.append(c.mean(x))
        medians.append(c.median(x))
        zRs.append(c.zR(x))
        zQs.append(c.zQ(x))
        ztrs.append(c.ztr(x))
    
    print(f'Poisson {size}')
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)))
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)))
    e = [c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)]
    d = [c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)]
    f(e, d)
    print('\\\\')


def Uniform(size):
    means, medians, zRs, zQs, ztrs = [], [], [], [], []
    for i in range(1000):
        x = sps.uniform(loc=-3**0.5, scale=2*3**0.5).rvs(size=size)
        x.sort()
        means.append(c.mean(x))
        medians.append(c.median(x))
        zRs.append(c.zR(x))
        zQs.append(c.zQ(x))
        ztrs.append(c.ztr(x))
    
    print(f'Uniform {size}')
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)))
    print(f'& %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)))
    e = [c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs)]
    d = [c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)]
    f(e, d)
    print('\\\\')




Normal(10)



Normal(100)



Normal(1000)



Cauchy(10)



Cauchy(100)



Cauchy(1000)



Laplace(10)



Laplace(100)



Laplace(1000)


Poisson(10)



Poisson(100)


Poisson(1000)


Uniform(10)


Uniform(100)



Uniform(1000)




