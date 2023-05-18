import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

# модель 
def func(x): 
    return 2 * x + 2

# внесение стандартно распределенной ошибки в модель 
def reference(x):
    return func(x) + sps.norm.rvs(loc = 0, scale = 1)

# генерация выборки без возмущений
def generate_points_without_perturbations(n, x_1, x_2):
    x = np.linspace(x_1, x_2, n)
    y = np.array(list(map(reference, x)))
    return x, y

# минимизируемая функция для метода наименьших модулей
def minimize_func_lad(param, x, y):
    beta_0, beta_1 = param
    sum = 0 
    for i in range (0, len(x)):
        sum += abs(y[i] - beta_0 - beta_1 * x[i])
    return sum 

#поиск коэффициентов линейной регрессии методом наименьших квадратов
def LSM(x, y):
    x_loc = np.mean(x) 
    y_loc = np.mean(y) 
    x_y = x * y
    x_y_loc = np.mean(x_y)
    beta_1 = (x_y_loc - x_loc * y_loc) / np.var(x)
    beta_0 = y_loc - x_loc * beta_1 
    return beta_0 , beta_1

# поиск коэффициентов линейной регрессии методом наименьших модулей
def LAD(x, y):
    beta_0_0, beta_1_0 = LSM(x, y)
    result = opt.minimize(fun=minimize_func_lad, x0 = [beta_0_0, beta_1_0], args=(x, y), method='SLSQP').x
    return result[0], result[1]

#отображение результатов
def plot_result():
    x, y = generate_points_without_perturbations(20, -1.8, 2.0)

    beta_0_LSM, beta_1_LSM = LSM(x, y)
    beta_0_LAD, beta_1_LAD = LAD(x, y)

    print(f'LSM: beta_0 = %.5f, beta_1 = %.5f'%(beta_0_LSM, beta_1_LSM)) 
    print(f'LAD: beta_0 = %.5f, beta_1 = %.5f'%(beta_0_LAD, beta_1_LAD)) 

    y_lsm = list(map(lambda x: beta_0_LSM + beta_1_LSM * x, x))
    y_lad = list(map(lambda x: beta_0_LAD + beta_1_LAD * x, x))

    fig1 = plt.figure()
    plt.scatter(x, y, s = 2, label = 'sampling')
    plt.plot(x, func(x), label='model', color = 'black')
    plt.plot(x, y_lsm, label='lsm', color = 'teal')
    plt.plot(x, y_lad, label='lad', color = 'blue')
    plt.title('Linear regression without perturbations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('regression_without_pert.png')


    # внесение возмущение в y1 и y20 
    y_pert = y 
    y_pert[0] += 10 
    y_pert[19] += -10

    beta_0_pert_LSM, beta_1_pert_LSM = LSM(x, y_pert)
    beta_0_pert_LAD, beta_1_pert_LAD = LAD(x, y_pert)

    print(f'with perturbations LSM: beta_0 = %.5f, beta_1 = %.5f'%(beta_0_pert_LSM, beta_1_pert_LSM)) 
    print(f'with perturbations LAD: beta_0 = %.5f, beta_1 = %.5f'%(beta_0_pert_LAD, beta_1_pert_LAD)) 

    y_lsm_pert = list(map(lambda x: beta_0_pert_LSM + beta_1_pert_LSM * x, x))
    y_lad_pert = list(map(lambda x: beta_0_pert_LAD + beta_1_pert_LAD * x, x))

    fig2 = plt.figure()
    plt.scatter(x, y, s = 2, label = 'sampling')
    plt.plot(x, func(x), label='model', color = 'black')
    plt.plot(x, y_lsm_pert, label='lsm', color = 'teal')
    plt.plot(x, y_lad_pert, label='lad', color = 'blue')
    plt.title('Linear regression with perturbations in y0 and y20')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('regression_with_pert.png')





if __name__ == '__main__':
    plot_result()

