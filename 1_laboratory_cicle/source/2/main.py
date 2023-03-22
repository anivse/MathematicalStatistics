import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import characteristics as c

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
    
    return c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs), c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)

        

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
    
    return c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs), c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)


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
    
    return c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs), c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)


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
    
    return c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs), c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)


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
    
    return c.E(means), c.E(medians), c.E(zRs), c.E(zQs), c.E(ztrs), c.D(means), c.D(medians), c.D(zRs), c.D(zQs), c.D(ztrs)

N10 = Normal(10)

print('Normal 10')
print(f'E: mean = {N10[0]:.{5}}, median = {N10[1]:.{5}}, zR = {N10[2]:.{5}}, zQ = {N10[3]:.{5}}, ztr = {N10[4]:.{5}}') 
print(f'D: mean = {N10[5]:.{5}}, median = {N10[6]:.{5}}, zR = {N10[7]:.{5}}, zQ = {N10[8]:.{5}}, ztr = {N10[9]:.{5}}') 

N100 = Normal(100)

print('Normal 100')
print(f'E: mean = {N100[0]:.{5}}, median = {N100[1]:.{5}}, zR = {N100[2]:.{5}}, zQ = {N100[3]:.{5}}, ztr = {N100[4]:.{5}}') 
print(f'D: mean = {N100[5]:.{5}}, median = {N100[6]:.{5}}, zR = {N100[7]:.{5}}, zQ = {N100[8]:.{5}}, ztr = {N100[9]:.{5}}') 

N1000 = Normal(1000)

print('Normal 1000')
print(f'E: mean = {N1000[0]:.{5}}, median = {N1000[1]:.{5}}, zR = {N1000[2]:.{5}}, zQ = {N1000[3]:.{5}}, ztr = {N1000[4]:.{5}}') 
print(f'D: mean = {N1000[5]:.{5}}, median = {N1000[6]:.{5}}, zR = {N1000[7]:.{5}}, zQ = {N1000[8]:.{5}}, ztr = {N1000[9]:.{5}}') 

C10 = Cauchy(10)

print('Cauchy 10')
print(f'E: mean = {C10[0]:.{5}}, median = {C10[1]:.{5}}, zR = {C10[2]:.{5}}, zQ = {C10[3]:.{5}}, ztr = {C10[4]:.{5}}') 
print(f'D: mean = {C10[5]:.{5}}, median = {C10[6]:.{5}}, zR = {C10[7]:.{5}}, zQ = {C10[8]:.{5}}, ztr = {C10[9]:.{5}}') 

C100 = Cauchy(100)

print('Cauchy 100')
print(f'E: mean = {C100[0]:.{5}}, median = {C100[1]:.{5}}, zR = {C100[2]:.{5}}, zQ = {C100[3]:.{5}}, ztr = {C100[4]:.{5}}') 
print(f'D: mean = {C100[5]:.{5}}, median = {C100[6]:.{5}}, zR = {C100[7]:.{5}}, zQ = {C100[8]:.{5}}, ztr = {C100[9]:.{5}}') 

C1000 = Cauchy(1000)

print('Cauchy 1000')
print(f'E: mean = {C1000[0]:.{5}}, median = {C1000[1]:.{5}}, zR = {C1000[2]:.{5}}, zQ = {C1000[3]:.{5}}, ztr = {C1000[4]:.{5}}') 
print(f'D: mean = {C1000[5]:.{5}}, median = {C1000[6]:.{5}}, zR = {C1000[7]:.{5}}, zQ = {C1000[8]:.{5}}, ztr = {C1000[9]:.{5}}') 

L10 = Laplace(10)

print('Laplace 10')
print(f'E: mean = {L10[0]:.{5}}, median = {L10[1]:.{5}}, zR = {L10[2]:.{5}}, zQ = {L10[3]:.{5}}, ztr = {L10[4]:.{5}}') 
print(f'D: mean = {L10[5]:.{5}}, median = {L10[6]:.{5}}, zR = {L10[7]:.{5}}, zQ = {L10[8]:.{5}}, ztr = {L10[9]:.{5}}') 

L100 = Laplace(100)

print('Laplace 100')
print(f'E: mean = {L100[0]:.{5}}, median = {L100[1]:.{5}}, zR = {L100[2]:.{5}}, zQ = {L100[3]:.{5}}, ztr = {L100[4]:.{5}}') 
print(f'D: mean = {L100[5]:.{5}}, median = {L100[6]:.{5}}, zR = {L100[7]:.{5}}, zQ = {L100[8]:.{5}}, ztr = {L100[9]:.{5}}') 

L1000 = Laplace(1000)

print('Laplace 1000')
print(f'E: mean = {L1000[0]:.{5}}, median = {L1000[1]:.{5}}, zR = {L1000[2]:.{5}}, zQ = {L1000[3]:.{5}}, ztr = {L1000[4]:.{5}}') 
print(f'D: mean = {L1000[5]:.{5}}, median = {L1000[6]:.{5}}, zR = {L1000[7]:.{5}}, zQ = {L1000[8]:.{5}}, ztr = {L1000[9]:.{5}}') 

P10 = Poisson(10)

print('Poisson 10')
print(f'E: mean = {P10[0]:.{5}}, median = {P10[1]:.{5}}, zR = {P10[2]:.{5}}, zQ = {P10[3]:.{5}}, ztr = {P10[4]:.{5}}') 
print(f'D: mean = {P10[5]:.{5}}, median = {P10[6]:.{5}}, zR = {P10[7]:.{5}}, zQ = {P10[8]:.{5}}, ztr = {P10[9]:.{5}}') 

P100 = Poisson(100)

print('Poisson 100')
print(f'E: mean = {P100[0]:.{5}}, median = {P100[1]:.{5}}, zR = {P100[2]:.{5}}, zQ = {P100[3]:.{5}}, ztr = {P100[4]:.{5}}') 
print(f'D: mean = {P100[5]:.{5}}, median = {P100[6]:.{5}}, zR = {P100[7]:.{5}}, zQ = {P100[8]:.{5}}, ztr = {P100[9]:.{5}}') 

P1000 = Poisson(1000)

print('Poisson 1000')
print(f'E: mean = {P1000[0]:.{5}}, median = {P1000[1]:.{5}}, zR = {P1000[2]:.{5}}, zQ = {P1000[3]:.{5}}, ztr = {P1000[4]:.{5}}') 
print(f'D: mean = {P1000[5]:.{5}}, median = {P1000[6]:.{5}}, zR = {P1000[7]:.{5}}, zQ = {P1000[8]:.{5}}, ztr = {P1000[9]:.{5}}') 

U10 = Uniform(10)

print('Uniform 10')
print(f'E: mean = {U10[0]:.{5}}, median = {U10[1]:.{5}}, zR = {U10[2]:.{5}}, zQ = {U10[3]:.{5}}, ztr = {U10[4]:.{5}}') 
print(f'D: mean = {U10[5]:.{5}}, median = {U10[6]:.{5}}, zR = {U10[7]:.{5}}, zQ = {U10[8]:.{5}}, ztr = {U10[9]:.{5}}') 

U100 = Uniform(100)

print('Uniform 100')
print(f'E: mean = {U100[0]:.{5}}, median = {U100[1]}, zR = {U100[2]:.{5}}, zQ = {U100[3]:.{5}}, ztr = {U100[4]:.{5}}') 
print(f'D: mean = {U100[5]:.{5}}, median = {U100[6]}, zR = {U100[7]:.{5}}, zQ = {U100[8]:.{5}}, ztr = {U100[9]:.{5}}') 

U1000 = Uniform(1000)

print('Uniform 1000')
print(f'E: mean = {U1000[0]:.{5}}, median = {U1000[1]}, zR = {U1000[2]:.{5}}, zQ = {U1000[3]:.{5}}, ztr = {U1000[4]:.{5}}') 
print(f'D: mean = {U1000[5]:.{5}}, median = {U1000[6]}, zR = {U1000[7]:.{5}}, zQ = {U1000[8]:.{5}}, ztr = {U1000[9]:.{5}}') 


