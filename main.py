import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm

M1 = np.array([0, 1]).reshape(2,1)
M2 = np.array([-1, -1]).reshape(2,1)
M3 = np.array([1, 0]).reshape(2,1)
B1 = np.array(([0.1, 0.02], [0.02, 0.2]))
B2 = np.array(([0.3, 0.03], [0.03, 0.25]))
B3 = np.array(([0.09, 0.01], [0.01, 0.05]))

def mahalanobis(M0, M1, B):
    M1_M0 = M1 - M0
    M1_M0_T = M1_M0.reshape(1,2)
    B_1 = np.linalg.inv(B)
    result = np.dot(M1_M0_T, B_1)
    result = np.dot(result, M1_M0)
    return result

def bayes_classificator_1(X, M1, M2, B1, threshold=0):
    M_diff = M1 - M2
    M_diff_T = M_diff.reshape(1,2)
    M_summ = M1 + M2
    M_summ_T = M_summ.reshape(1,2)
    B_inv = np.linalg.inv(B1)
    B = np.dot(M_diff_T, B_inv)
    c = 0.5 * np.dot(np.dot(M_summ_T,B_inv),M_diff) - threshold
    Y=[]
    for x in X:
        y = (c - B[0,0]*x)/B[0,1]
        Y.append(np.float_(y))
    return np.array(Y)

def calc_errors(M1, M2, B):
    r_m = mahalanobis(M1, M2, B)
    p1 = 1 - scipy.stats.norm.cdf(0.5*np.sqrt((r_m)))
    p2 = scipy.stats.norm.cdf(-0.5*np.sqrt((r_m)))
    return p1, p2

def bayes_classificator(X, Mi, Mj, Bi, Bj, threshold):
    Bi_inv = np.linalg.inv(Bi)
    Bj_inv = np.linalg.inv(Bj)
    B_dif_1 = Bj_inv - Bi_inv
    Mi_T = Mi.reshape(1,2)
    Mj_T = Mj.reshape(1, 2)
    B_dif_2 = np.dot(Mi_T, Bi_inv) - np.dot(Mj_T, Bj_inv)
    c = 0.5 * np.dot(np.dot(Mj_T,Bj_inv),Mj) - 0.5 * np.dot(np.dot(Mi_T,Bi_inv),Mi) + 0.5 * np.log((np.linalg.det(Bj) / np.linalg.det(Bi))) + threshold
    A = B_dif_1[1,1]

    bound = []
    for x in X:
        B = B_dif_1[0,1] * x + B_dif_1[1,0] * x + 2 * B_dif_2[0,1]
        C = 2 * c + 2 * B_dif_2[0,0] * x + B_dif_1[0,0] * x * x
        D = B ** 2 - 4 * A * C
        if D >= 0 :
            y1 = (-B + np.sqrt(D))/(2 * A)
            y2 = (-B - np.sqrt(D)) / (2 * A)
            if y1 == y2:
                bound += [x, y1.item()]
            else:
                bound += [[x, y1.item()], [x, y2.item()]]
    return np.array((bound))

def diff_errors(X, M1, M2, B1, B2):
    count = 0
    for i in range(0, len(X[0])):
        x = X[:, i].reshape(2,1)
        d1 = np.log((0.5)) - np.log(np.sqrt(np.linalg.det(B1))) - \
            0.5 * np.dot(np.dot(np.transpose(x - M1), np.linalg.inv(B1)), (x - M1))
        d2 = np.log((0.5)) - np.log(np.sqrt(np.linalg.det(B2))) - \
             0.5 * np.dot(np.dot(np.transpose(x - M2), np.linalg.inv(B2)), (x - M2))
        if d2 > d1:
            count += 1
    return count/len(X[0])

def task1():
    S1, S2 = np.load("task1.npy")
    x_array = np.linspace(-2, 2, 100)

    p1, p2 = calc_errors(M1, M2, B1)
    sum_p = p1+p2
    print("Вероятность ошибочной классификации для первого класса: ", p1)
    print("Вероятность ошибочной классификации для второго класса: ", p2)
    print("Суммарная вероятность ошибочной классификации: ", sum_p)
    y_array = bayes_classificator_1(x_array, M1, M2, B1)
    fig, ax = plt.subplots()
    ax.scatter(S1[0, :], S1[1, :], c='b')
    ax.scatter(S2[0, :], S2[1, :], c='r')
    ax.scatter(x_array, y_array)
    plt.show()

def task2():
    S1, S2 = np.load("task1.npy")
    x_array = np.linspace(-2, 2, 100)

    r_m = mahalanobis(M1, M2, B1)
    lambda_ = -0.5*r_m+np.sqrt(r_m)*1.645
    y_array = bayes_classificator_1(x_array, M1, M2, B1, lambda_)
    fig, ax = plt.subplots()
    ax.scatter(S1[0, :], S1[1, :], c='b')
    ax.scatter(S2[0, :], S2[1, :], c='r')
    ax.scatter(x_array, y_array)
    plt.show()

def task3():
    S1, S2, S3 = np.load("task2.npy")

    x_array = np.linspace(-2, 2, 100)

    bound_1_2 = bayes_classificator(x_array, M1, M2, B1, B2, 0)
    bound_1_3 = bayes_classificator(x_array, M1, M3, B1, B3, 0)
    bound_2_3 = bayes_classificator(x_array, M2, M3, B2, B3, 0)

    fig, ax = plt.subplots()
    ax.scatter(bound_1_2[:, 0], bound_1_2[:, 1], c='black')
    ax.scatter(bound_1_3[:, 0], bound_1_3[:, 1], c='orange')
    ax.scatter(bound_2_3[:, 0], bound_2_3[:, 1], c='yellow')

    ax.scatter(S1[0, :], S1[1, :], c='b')
    ax.scatter(S2[0, :], S2[1, :], c='r')
    ax.scatter(S3[0, :], S3[1, :], c='g')

    plt.ylim(-3, 2)
    plt.xlim(-3, 2)

    error1 = diff_errors(S1, M1, M2, B1, B2)
    error2 = diff_errors(S2, M2, M1, B2, B1)

    print("Ошибка классификации для первой выборки: ", error1)
    print("Ошибка классификации для второй выборки: ", error2)

    p = (error1 + error2)/2
    E = np.sqrt((1 - p) / (200 * p))
    p0 = 0.05
    N = (1 - p0) / (E * E * p0)

    print("Относительная погрешность для объема выборки 200: ", E)
    print("Объем выборки с погрешностью 5%: ", N)

    plt.show()


task1()
task2()
task3()