import numpy as np
from tabulate import tabulate




def iteration(alpha, beta, x, eps):
    k=1
    err = eps + 1
    while err > eps and k < 500:
        err = np.linalg.norm(np.dot(alpha, x) + beta - x)
        x = np.dot(alpha, x) + beta
        k += 1
    x = np.dot(alpha, x) + beta
    return x, k


def zeidel(A, b, eps):
    k = 0
    x = np.array(np.zeros((b.shape[0])))
    err = eps + 1
    while err > eps:
        x_new = x.copy()
        for i in range(A.shape[0]):
            x1 = sum(A[i][j] * x_new[j] for j in range(i))
            x2 = sum(A[i][j] * x[j] for j in range(i + 1, A.shape[0]))
            x_new[i] = (b[i] - x1 - x2)/A[i][i]
        err = np.linalg.norm(x_new - x)
        k += 1
        x = x_new
    return x, k

def calculate_alpha_beta(A, b):
    alpha = np.array(np.zeros((A.shape[0], A.shape[0])))
    beta = np.array(np.zeros(b.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j:
                alpha[i][j] = - A[i][j] / A[i][i]
                beta[i] = b[i] / A[i][i]
            else:
                alpha[i][i] = 0
    return alpha, beta
def iter_form(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + 1 + j + 1 - 1)
    return A


A2 = np.array([[1, 1 / 2,],
                  [1 / 2, 1 / 3]],dtype=float)
A3 = np.array([[1, 1 / 2, 1 / 3, ],
                  [1 / 2, 1 / 3, 1 / 4],
                  [1 / 3, 1 / 4, 1 / 5]],dtype=float)
A4 = np.array([[-500.7, 120.7],
                  [ 890.3, -550.6]],dtype=float)
x2 = np.random.uniform(0, 100, size=A2.shape[0])
x4 = np.random.uniform(0, 100, size=A4.shape[0])
x3 = np.random.uniform(0, 100, size=A3.shape[0])
p_A2=A2
p_A3=A3
p_A4=A4
A2 = iter_form(A2)
A4 = iter_form(A4)
A3 = iter_form(A3)
b2 = np.dot(A2,x2)
b4 = np.dot(A4,x4)
b3 = np.dot(A3,x3)
alpha2, beta2 = calculate_alpha_beta(A2, b2)
alpha4, beta4 = calculate_alpha_beta(A4, b4)
alpha3, beta3 = calculate_alpha_beta(A3, b3)
print(p_A4)
print(tabulate([[10**(-5),iteration(alpha4, beta4, beta4, 10**(-5))[1],zeidel(A4, b4,10**(-5))[1],np.linalg.norm(x4 - iteration(alpha4, beta4, beta4, 10**(-5))[0]),np.linalg.norm(x4 - zeidel(A4, b4, 10**(-5))[0])],
                [10**(-8),iteration(alpha4, beta4, beta4, 10**(-8))[1],zeidel(A4, b4,10**(-8))[1],np.linalg.norm(x4 - iteration(alpha4, beta4, beta4, 10**(-8))[0]),np.linalg.norm(x4 - zeidel(A4, b4, 10**(-8))[0])],
                [10**(-11),iteration(alpha4, beta4, beta4, 10**(-11))[1],zeidel(A4, b4,10**(-11))[1],np.linalg.norm(x4 - iteration(alpha4, beta4, beta4, 10**(-11))[0]),np.linalg.norm(x4 - zeidel(A4, b4, 10**(-11))[0])],
                [10**(-14),iteration(alpha4, beta4, beta4, 10**(-14))[1],zeidel(A4, b4,10**(-14))[1],np.linalg.norm(x4 - iteration(alpha4, beta4, beta4, 10**(-14))[0]),np.linalg.norm(x4 - zeidel(A4, b4, 10**(-14))[0])]], headers=['Погрешность','#Итерации простого','#Итерации Зейделя','|x-x_pr|','|x-x_zei|'],tablefmt='orgtbl'))
print(p_A3)
print(tabulate([[10**(-5),iteration(alpha3, beta3, beta3, 10**(-5))[1],zeidel(A3, b3,10**(-5))[1],np.linalg.norm(x3 - iteration(alpha3, beta3, beta3, 10**(-5))[0]),np.linalg.norm(x3 - zeidel(A3, b3, 10**(-5))[0])],
                [10**(-8),iteration(alpha3, beta3, beta3, 10**(-8))[1],zeidel(A3, b3,10**(-8))[1],np.linalg.norm(x3 - iteration(alpha3, beta3, beta3, 10**(-8))[0]),np.linalg.norm(x3 - zeidel(A3, b3, 10**(-8))[0])],
                [10**(-11),iteration(alpha3, beta3, beta3, 10**(-11))[1],zeidel(A3, b3,10**(-11))[1],np.linalg.norm(x3 - iteration(alpha3, beta3, beta3, 10**(-11))[0]),np.linalg.norm(x3 - zeidel(A3, b3, 10**(-11))[0])],
                [10**(-14),iteration(alpha3, beta3, beta3, 10**(-14))[1],zeidel(A3, b3,10**(-14))[1],np.linalg.norm(x3 - iteration(alpha3, beta3, beta3, 10**(-14))[0]),np.linalg.norm(x3 - zeidel(A3, b3, 10**(-14))[0])]], headers=['Погрешность','#Итерации простого','#Итерации Зейделя','|x-x_pr|','|x-x_zei|'],tablefmt='orgtbl'))
print(p_A2)
print(tabulate([[10**(-5),iteration(alpha2, beta2, beta2, 10**(-5))[1],zeidel(A2, b2,10**(-5))[1],np.linalg.norm(x2 - iteration(alpha2, beta2, beta2, 10**(-5))[0]),np.linalg.norm(x2 - zeidel(A2, b2, 10**(-5))[0])],
                [10**(-8),iteration(alpha2, beta2, beta2, 10**(-8))[1],zeidel(A2, b2,10**(-8))[1],np.linalg.norm(x2 - iteration(alpha2, beta2, beta2, 10**(-8))[0]),np.linalg.norm(x2 - zeidel(A2, b2, 10**(-8))[0])],
                [10**(-11),iteration(alpha2, beta2, beta2, 10**(-11))[1],zeidel(A2, b2,10**(-11))[1],np.linalg.norm(x2 - iteration(alpha2, beta2, beta2, 10**(-11))[0]),np.linalg.norm(x2 - zeidel(A2, b2, 10**(-11))[0])],
                [10**(-14),iteration(alpha2, beta2, beta2, 10**(-14))[1],zeidel(A2, b2,10**(-14))[1],np.linalg.norm(x2 - iteration(alpha2, beta2, beta2, 10**(-14))[0]),np.linalg.norm(x2 - zeidel(A2, b2, 10**(-14))[0])]], headers=['Погрешность','#Итерации простого','#Итерации Зейделя','|x-x_pr|','|x-x_zei|'],tablefmt='orgtbl'))



