import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math

def mean(x): #среднее
    n = len(x)
    sum = np.sum(x)
    return sum / n

def median(x): #медиана
    n = len(x)
    if n % 2 == 0:
        return (x[(int)(n/2) - 1] + x[(int)(n/2)]) / 2
    else:
        return x[(int)((n-1) / 2)]

def zR(x): #полусумма экстремальных выборочных элементов
    n = len(x)
    return (x[0] + x[n-1]) / 2

def zQ(x): #полусумма квартилей
    n = len(x)
    p1 = 0.25
    p2 = 0.75
    z1 = x[math.ceil(n*p1)-1]
    z2 = x[math.ceil(n*p2)-1]
    return (z1 + z2) / 2

def ztr(x): #усеченное среднее
    n = len(x)
    r = math.ceil(n / 4)
    sum = 0 
    for i in range (r, n-r):
        sum += x[i]
    return sum / (n - 2*r)

def E(z): #среднее характеристик положения
    return mean(z)

def D(z): #оценка дисперсии 
    n = len(z)
    zquad = []
    for i in range (0, n):
        zquad.append(z[i]**2)
    
    return mean(zquad) - mean(z)**2






