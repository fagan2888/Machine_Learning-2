import numpy as np
import scipy.io as sio

bodyfat = sio.loadmat('bodyfat_data.mat')
x = np.transpose(bodyfat['X'])
y = bodyfat['y']
training_x = x[:, 0:150]
test_x = x[:, 150: ]
training_y = y[0:150]
test_y = y[150: ]
sigma = 15
lamb = 0.003
n = 150
m = 98


# calculate the gaussian kernel of two column vectors
def gaussian_ker(u, v):
    uv_trans = np.transpose(u-v)
    return np.exp(-1/2/sigma**2*np.dot(uv_trans, u-v))


# matrix K is the inner product matrix of kernel
K = np.zeros((n,n))
for i in range(n):
    for j in range(i, n):
        K[i][j] = gaussian_ker(training_x[:,i], training_x[:,j])
K = np.transpose(K) + K - np.eye(n)
On = 1/n*np.ones((n,n))
Om = 1/m*np.ones((m,m))
Onm = 1/n*np.ones((n,m))
# K_til is the matrix with centralized kernel
K_til = K - np.dot(K, On) - np.dot(On, K) + np.dot(np.dot(On, K), On)
# kernel with test data
K_pri = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        K_pri[i][j] = gaussian_ker(training_x[:,i], test_x[:,j])
# K_til_pri is the matrix centralized with test data
K_til_pri = K_pri - np.dot(On, K_pri) - np.dot(K, Onm) + np.dot(np.dot(On, K), Onm)
mean_y = np.mean(training_y)
y_til = training_y - mean_y
inv = np.linalg.pinv(K_til + n*lamb*np.eye(n))
test_prediction = np.transpose(mean_y + np.dot(np.dot(np.transpose(y_til), inv), K_til_pri))
# calculate test error
test_error = 1/m*np.dot(np.transpose(test_y - test_prediction), test_y - test_prediction)[0][0]
print("the test error with offset is " + str(test_error))
# calculate training error
train_prediction = np.transpose(mean_y + np.dot(np.dot(np.transpose(y_til), inv), K_til))
training_error = 1 / n * np.dot(np.transpose(training_y - train_prediction), training_y - train_prediction)[0][0]
print("the training error with offset is " + str(training_error))

# based on the formula in (a), the offset expression is shown below
N = 1/n * np.ones((150,1))
Xx = np.dot(K, N) - np.dot(np.transpose(N), np.dot(K,N))
invx = np.linalg.pinv(n*lamb*np.eye(n) + K_til)
off_set = mean_y - np.dot(np.dot(np.transpose(y_til), invx), Xx)[0][0]
print("the offset is " + str(off_set))