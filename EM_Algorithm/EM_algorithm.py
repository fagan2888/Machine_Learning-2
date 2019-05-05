import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pylab as plb

np.random.seed(0)
n = 200
K = 2
e = np.array([0.7, 0.3])
w = np.array([-2, 1])
b = np.array([0.5, -0.5])
v = np.array([0.2, 0.1])
x = np.zeros([n])
y = np.zeros([n])
for i in range(0, n):
    x[i] = np.random.rand(1)
    if np.random.rand(1) < e[0]:
        y[i] = w[0] * x[i] + b[0] + np.random.randn(1) * np.sqrt(v[0])
    else:
        y[i] = w[1] * x[i] + b[1] + np.random.randn(1) * np.sqrt(v[1])

plt.plot(x, y, 'bo')
t = np.linspace(0, 1, num= 100)
plt.plot(t, w[0] * t + b[0], 'k')
plt.plot(t, w[1] * t + b[1], 'k')


def log_likelihood(epsl, w_new, b_new, sigm_sqr):
    phy = np.zeros((n, K))
    for i in range(0,n):
        for k in range(0,K):
            phy[i][k] = norm.pdf(y[i], w_new[k] * x[i] + b_new[k], np.sqrt(sigm_sqr[k]))
    summ = np.sum(np.log(np.matmul(phy, epsl.reshape((2, 1)))))
    return summ


gamma = np.zeros((n, K))
epsl = np.array([0.5, 0.5])
w_new = np.array([1., -1.])
b_new = np.array([0., 0.])
sigm_sqr = np.array([np.var(y), np.var(y)])
iter_num = 0
tolrance = 0.0001
likelihood_list = []

log0 = 0
log1 = log_likelihood(epsl, w_new, b_new, sigm_sqr)

while abs(log1 - log0) > tolrance:
    iter_num = iter_num + 1
    for i in range(0, n):
        denominator = epsl[0] * norm.pdf(y[i], w_new[0] * x[i] + b_new[0], np.sqrt(sigm_sqr[0])) \
                      + epsl[1] * norm.pdf(y[i], w_new[1] * x[i] + b_new[1], np.sqrt(sigm_sqr[1]))
        for k in range(0,K):
            numerator = epsl[k] * norm.pdf(y[i], w_new[k] * x[i] + b_new[k], np.sqrt(sigm_sqr[k]))
            gamma[i][k] = numerator/denominator
    epsl = np.sum(gamma, axis = 0)/n
    C0 = np.diag(gamma[:, 0])
    C1 = np.diag(gamma[:, 1])
    x_til = np.hstack((np.ones((n, 1)), x.reshape((n, 1))))
    matr0 = np.linalg.inv(np.matmul(np.matmul(np.transpose(x_til), C0), x_til))
    b_new[0], w_new[0] = np.matmul(np.matmul(matr0, np.transpose(x_til)), np.matmul(C0, y))
    matr1 = np.linalg.inv(np.matmul(np.matmul(np.transpose(x_til), C1), x_til))
    b_new[1], w_new[1] = np.matmul(np.matmul(matr1, np.transpose(x_til)), np.matmul(C1, y))
    sigm_sqr[0] = abs(np.inner(gamma[:, 0], (y - (w_new[0] * x + b_new[0])) ** 2)/np.sum(gamma[:, 0]))
    sigm_sqr[1] = abs(np.inner(gamma[:, 1], (y - (w_new[1] * x + b_new[1])) ** 2)/np.sum(gamma[:, 1]))
    log0 = log1
    log1 = log_likelihood(epsl, w_new, b_new, sigm_sqr)
    likelihood_list.append(log0)

print("The number of iterations is: ", iter_num)
print("mixing weights: ", epsl)
print("slopes of lines: ", w_new)
print("offsets of lines: ", b_new)
print("variances: ", sigm_sqr)
plt.plot(t, w_new[0] * t + b_new[0], ':')
plt.plot(t, w_new[1] * t + b_new[1], ':')

plt.figure()
iter = plb.frange(1,iter_num)
plt.plot(iter, likelihood_list[:iter_num], label = 'log-likelihood')
plt.show()






