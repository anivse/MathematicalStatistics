import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sb 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def quadrant_coeff(x,y):
    n = len(x)
    n1, n2, n3, n4 = 0, 0, 0, 0
    x_mean = np.median(x)
    y_mean = np.median(y)
    for i in range(0,n):
        if x[i] >= x_mean and y[i] >= y_mean:
            n1 += 1
        if x[i] < x_mean and y[i] >= y_mean:
            n2 += 1
        if x[i] < x_mean and y[i] < y_mean:
            n3 += 1
        if x[i] >= x_mean and y[i] < y_mean:
            n4 += 1
    return ((n1 + n3) - (n2 + n4)) / n

ros = [0, 0.5, 0.9, 'mix']
numbers = [20, 60, 100]

def mix_multivariate_normal(size):
    return 0.9 * sps.multivariate_normal.rvs([0, 0], [[1.0, 0.9], [0.9, 1.0]], size=size) + 0.1 * sps.multivariate_normal.rvs([0, 0], [[10.0, -0.9], [-0.9, 10.0]], size=size)

def compute_corr_coeffs():
    file = open("result.txt", 'w')
    for ro in ros:
        file.write(f'rho = {ro} \n')
        file.write(f'\\hline\n')
        for n in numbers:
            r, rs, rq= [], [], []
            for i in range (0, 1000):
               if ro == 'mix':
                 select = mix_multivariate_normal(n)
               else: 
                 select = sps.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=n)
               x = select[:, 0]
               y = select[:, 1]
               r.append(sps.pearsonr(x,y)[0])
               rs.append(sps.spearmanr(x,y)[0])
               rq.append(quadrant_coeff(x,y))
            rr = np.power(r, 2)
            rrs = np.power(rs, 2)
            rrq = np.power(rq, 2)
            Er = np.mean(r)
            Err = np.mean(rr)
            Ers = np.mean(rs)
            Errs = np.mean(rrs)
            Erq = np.mean(rq)
            Errq = np.mean(rrq)
            Dr = np.var(r)
            Drs = np.var(rs)
            Drq = np.var(rq)
            file.write(f'$size = {n}$ & $r$ & $r_s$ & $r_q$ \\\\ \n')
            file.write(f'\\hline \n')
            file.write(f'$E(z)$ & %.5f & %.5f & %.5f \\\\ \n' %(Er, Ers, Erq))
            file.write(f'\\hline \n')
            file.write(f'$E(z^2)$ & %.5f & %.5f & %.5f \\\\ \n' %(Err, Errs, Errq))
            file.write(f'\\hline \n')
            file.write(f'$D(z)$ & %.5f & %.5f & %.5f \\\\ \n' %(Dr, Drs, Drq))
            file.write(f'\\hline \\hline \n')
    file.close()

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_ellipses():
    for ro in ros:
        for n in numbers:
            if ro == 'mix':
                 select = mix_multivariate_normal(n)
            else: 
                 select = sps.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=n)
            x = select[:,0]
            y = select[:,1]
            fig = plt.figure()
            ax = plt.subplot(111)
            confidence_ellipse(x, y, ax, edgecolor='blue')
            ax.grid() 
            ax.scatter(x,y)
            plt.xlabel('x')
            plt.ylabel('y')
            if ro == 'mix':
                plt.title(f'ellipse size={n} mix')
            else:
                plt.title(f'ellipse size={n} rho={ro}')
            plt.savefig(f'ellipse_{n}_{ro}.png')
            plt.close()


if __name__ == '__main__':
    # plot_ellipses()
    compute_corr_coeffs()
   


        




            




