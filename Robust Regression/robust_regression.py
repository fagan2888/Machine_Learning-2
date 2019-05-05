import numpy as np
import matplotlib.pyplot as plt
import random
import pylab as pl

n = 200
np.random.seed(0)
x = np.random.rand(n, 1)
z = np.zeros([n,1])
k = n * 0.4
rp = np.random.permutation(n)
outlier_subset = rp[1:int(k)]
z[outlier_subset] = 1
y = (1-z)*(10*x + 5 + np.random.randn(n,1)) + z*(20- 20*x + 10*np.random.randn(n,1))

# plot data and true line

plt.scatter(x, y, label = 'data')

# t = pl.frange(0,1,0.01)
t = np.arange(0,1,0.01)

plt.plot(t, 10*t+5, 'k-', label = 'true line')

# Add your code for ordinary least squares below

xtil = np.concatenate((np.ones([len(y), 1]), x), axis=1)
mat1 = np.linalg.pinv(np.matmul(np.transpose(xtil), xtil))
mat2 = np.matmul(np.transpose(xtil), y)
b_ols = np.matmul(mat1, mat2)[0][0]
w_ols = np.matmul(mat1, mat2)[1][0]
print("The parameters in robust regression are w_ols = " + str(w_ols) + ", b_ols = " + str(b_ols))
plt.plot(t, w_ols * t + b_ols, 'g--', label = 'least squares')

# add your code for robust regression MM algorithm below

def wls(xtil,y,c):
    # helper function to solve weighted least squares
    # add code here
    ctil = np.diag(c)
    matr1 = np.linalg.pinv(np.matmul(np.transpose(xtil), np.matmul(ctil, xtil)))
    matr2 = np.matmul(np.transpose(xtil), np.matmul(ctil,y))
    b = np.matmul(matr1, matr2)[0][0]
    w = np.matmul(matr1, matr2)[1][0]
    return w, b


def psi(r):
    return r/np.sqrt(1+r**2)


w0 = 0; b0 = 0; w1 = 1; b1 = 1
while (b1-b0)**2 + (w1-w0)**2 > 1e-5:
    b0 = b1
    w0 = w1
    c = []
    for i in range(len(y)):
        r = y[i][0] - w1*x[i][0] - b1
        c.append(psi(r)/r)
    w1, b1 = wls(xtil, y, c)

w_rob = w1
b_rob = b1
print("The parameters in robust regression are w_rob = " + str(w_rob) + ", b_rob = " + str(b_rob))
plt.plot(t, w_rob * t + b_rob, 'r:', label = 'robust')

legend = plt.legend(loc = 'upper right', shadow = True)

plt.show()



