import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

nuclear_data = sio.loadmat('nuclear.mat')
x = nuclear_data['x']
y = nuclear_data['y']
d, n = x.shape
lamb = 0.001


# calculate sub gradient
def subgradient(xi, yi, w, b):
    x = yi * (np.matmul(np.transpose(w), xi)[0][0] + b)
    if x > 1:
        return np.array([0,lamb/n*w[0][0], lamb/n*w[1][0]]).reshape(3,1)
    elif x < 1:
        grad = np.concatenate((np.array(-yi/n).reshape(1,1), -yi/n*xi + lamb/n*w), axis=0)
        return grad
    else:
        grad = 1/2*np.concatenate((np.array(-yi/n).reshape(1,1), -yi/n*xi + 2*lamb/n*w), axis=0)
        return grad


# calculate objective function
def obj_fun(theta):
    sum1 = 0
    for i in range(n):
        sum1 = sum1 + max(0, 1-y[0][i] * (np.matmul(np.transpose(theta[1:,:]), x[:,i].reshape(2,1)) + theta[0][0]))
    sum2 = sum1[0][0]
    return 1/n*sum2+lamb/2*np.matmul(np.transpose(theta[1:,:]), theta[1:,:])[0][0]


theta1 = np.array([0,0,0]).reshape(3,1)
iter_num = 0
obj_fun_val = []
a = obj_fun(theta1)
obj_fun_val.append(a)
while obj_fun_val[-1] > 0.3 and iter_num < 40:
    iter_num = iter_num + 1
    sum_subgrad = 0
    for i in range(n):
        sum_subgrad = sum_subgrad + subgradient(x[:,i].reshape(2,1), y[0][i], theta1[1:,:], theta1[0][0])
    theta1 = theta1 - 100 / iter_num * sum_subgrad
    obj_fun_val.append(obj_fun(theta1))

print("The estimated hyperplane parameters are b = "+str(theta1[0][0])+" w = "+str(theta1[1][0])+" and "+str(theta1[2][0]))
print("The achieved value of objective function is " + str(obj_fun_val[-1]))

# plot of data and learned line
t = np.arange(0, iter_num+1, 1)
plt.plot(t, obj_fun_val)
plt.show()

# plot showing J as a function of iteration number
plt.figure()
x1, x2 = x[0,:], x[1,:]
x1_min, x1_max = np.min(x1)*.7, np.max(x1)*1.3
gridPoints = 2000
x1s = np.linspace(x1_min, x1_max, gridPoints)
y_line = (-theta1[1][0]*x1s-theta1[0][0])/theta1[2][0]
plt.scatter(x[0,:], x[1,:], c=y[0], cmap=plt.cm.Paired)
plt.plot(x1s, y_line)
plt.show()