import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

np.random.seed(0)
nuclear_data = sio.loadmat('nuclear.mat')
x = nuclear_data['x']
y = nuclear_data['y']
d, n = x.shape
lamb = 0.001
xtil = np.concatenate((np.ones([20000, 1]), np.transpose(x)), axis=1)


# function to calculate the sub gradient of xi and yi
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


# calculate the objective function
def obj_fun(theta):
    arr = 1 - np.multiply(np.transpose(y), np.dot(xtil, theta))
    arr_posi = np.maximum(arr, 0)
    return 1/n*np.sum(arr_posi)+lamb/2*(theta[1][0]**2+theta[2][0]**2)


theta1 = np.array([0,0,0]).reshape(3,1)
iter_num = 0
obj_fun_val = []

# the iteration to calculate theta
while obj_fun(theta1) > 0.3 and iter_num < 40:
    iter_num = iter_num + 1
    per_n = np.random.permutation(n)
    for i in range(n):
        ind = per_n[i]
        sub_grad = subgradient(x[:,ind].reshape(2,1), y[0][ind], theta1[1:,:], theta1[0][0])
        theta1 = theta1 - 100 / iter_num * sub_grad
        obj_fun_val.append(obj_fun(theta1))

print("The estimated hyperplane parameters are b = "+str(theta1[0][0])+" w = "+str(theta1[1][0])+" and "+str(theta1[2][0]))
print("The achieved value of objective function is " + str(obj_fun_val[-1]))

# plot of data and learned line
t = np.arange(1, len(obj_fun_val)+1, 1)
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