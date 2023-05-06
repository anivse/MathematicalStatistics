import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math as m
import scipy.optimize as opt
import tabulate

gamma = 0.95
alpha = 0.05 



def classic(x):
    n = len(x)
    m = np.mean(x)
    s = np.sqrt(np.var(x))
    

    m_int = [m - s * (sps.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1),
        m + s * (sps.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1)]
    s_int = [s * np.sqrt(n) / np.sqrt(sps.chi2.ppf(1 - alpha / 2, n - 1)),
        s * np.sqrt(n) / np.sqrt(sps.chi2.ppf(alpha / 2, n - 1))]
    
    print("classic n = %i :  m = [%.2f, %.2f], s = [%.2f, %.2f]" % (n, m_int[0], m_int[1], s_int[0], s_int[1]))
    
    return m_int, s_int



def asymptotic(x):
    n = len(x)
    m = np.mean(x)
    s = np.sqrt(np.var(x))

    m_int = [m - sps.norm.ppf(1 - alpha / 2) / np.sqrt(n), m + sps.norm.ppf(1 - alpha / 2) / np.sqrt(n)]
    e = (sum(list(map(lambda el: (el - m) ** 4, x))) / n) / s ** 4 - 3
    s_int = [s / np.sqrt(1 + sps.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n)),
            s / np.sqrt(1 - sps.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n))]
    
    print("asymptotic n = %i :  m = [%.2f, %.2f], s = [%.2f, %.2f]" % (n, m_int[0], m_int[1], s_int[0], s_int[1]))

    return m_int, s_int

    

def hist(x):
    size = len(x)
    plt.figure()
    plt.hist(x, bins=15, color='white', edgecolor='black', density=True)
    plt.title(f'hist n = {size}')
    plt.savefig(f'hist_n_{size}.png')


if __name__ == "__main__":
    x_20 = sps.norm.rvs(size=20, loc=0, scale = 1)
    x_100 = sps.norm.rvs(size=100, loc=0, scale = 1)
    hist(x_20)
    hist(x_100)

    mc20, sc20 = classic(x_20)
    mc100, sc100 = classic(x_100)

    ma20, sa20 = asymptotic(x_20)
    ma100, sa100 = asymptotic(x_100)

    plt.figure()
    plt.ylim(0.9, 1.4)
    plt.plot(mc100, [1.1, 1.1], label = "n100")
    plt.plot(mc20, [1, 1], label = "n20")
    plt.legend()
    plt.title('classical confidence interval for m')
    plt.savefig('classic_m.png')

    plt.figure()
    plt.ylim(0.9, 1.4)
    plt.plot(sc100, [1.1, 1.1], label = "n100")
    plt.plot(sc20, [1, 1], label = "n20")
    plt.legend()
    plt.title('classical confidence interval for s')
    plt.savefig('classic_s.png')

    plt.figure()
    plt.ylim(0.9, 1.4)
    plt.plot(ma100, [1.1, 1.1], label = "n100")
    plt.plot(ma20, [1, 1], label = "n20")
    plt.legend()
    plt.title('asymptotic confidence interval for m')
    plt.savefig('asymptotic_m.png')

    plt.figure()
    plt.ylim(0.9, 1.4)
    plt.plot(sa100, [1.1, 1.1], label = "n100")
    plt.plot(sa20, [1, 1], label = "n20")
    plt.legend()
    plt.title('asymptotic confidence interval for s')
    plt.savefig('asymptotic_s.png')





    