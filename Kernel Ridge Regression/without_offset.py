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


# calculate gaussian kernel
def gaussian_ker(u, v):
    uv_trans = np.transpose(u-v)
    return np.exp(-1/2/sigma**2*np.dot(uv_trans, u-v))


# kernel matrix with training data
K = np.zeros((n,n))
for i in range(n):
    for j in range(i, n):
        K[i][j] = gaussian_ker(training_x[:,i], training_x[:,j])
K = np.transpose(K) + K - np.eye(n)
# kernel matrix with test data
K_pri = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        K_pri[i][j] = gaussian_ker(training_x[:,i], test_x[:,j])
# test error
inv = np.linalg.pinv(K + n* lamb*np.eye(n))
test_prediction = np.transpose(np.dot(np.dot(np.transpose(training_y), inv), K_pri))
test_error = 1/m*np.dot(np.transpose(test_y - test_prediction), test_y - test_prediction)[0][0]
print("the test error without offset is " + str(test_error))
# training error
train_prediction = np.transpose(np.dot(np.dot(np.transpose(training_y), inv), K))
training_error = 1 / n * np.dot(np.transpose(training_y - train_prediction), training_y - train_prediction)[0][0]
print("the training error without offset is " + str(training_error))
