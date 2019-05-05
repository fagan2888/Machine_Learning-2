import numpy as np
import scipy.io as sio

np.random.seed(0)

body_fat = sio.loadmat('bodyfat_data.mat')
X = np.transpose(body_fat['X'])
y = body_fat['y']
X_row = X.shape[0]
X_col = X.shape[1]
ones = np.ones((1, X_col))
x_til = np.concatenate((X, ones), axis = 0)
train_num = 150
training_x = x_til[:, 0: train_num]
test_x = x_til[:, train_num:]
training_y = y[0: train_num]
test_y = y[train_num:]

frst_ly_node_num = 64  # number of nodes in first layer
sec_ly_node_num = 16  # number of nodes in second layer
learning_rate = 1e-7  # learning rate
W1 = np.random.randn(X_row, frst_ly_node_num)  # weights in first layer
b1 = np.zeros((frst_ly_node_num, 1))  # offset in first layer
theta1 = np.concatenate((np.transpose(W1), b1), axis=1)  # combination of w and b
W2 = np.random.randn(frst_ly_node_num, sec_ly_node_num)
b2 = np.zeros((sec_ly_node_num, 1))
theta2 = np.concatenate((np.transpose(W2), b2), axis=1)
W3 = np.random.randn(sec_ly_node_num, 1)
b3 = np.zeros((1,1))
theta3 = np.concatenate((np.transpose(W3), b3), axis=1)


# ReLU active function
def relu(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1]
    zero = np.zeros((row, col))
    return np.maximum(zero, matrix)


# forward process
def forward_process(x, theta1, theta2, theta3):
    a1 = np.matmul(theta1, x)
    z1 = relu(a1)
    z1_til = np.concatenate((z1, np.ones((1, z1.shape[1]))), axis=0)
    a2 = np.matmul(theta2, z1_til)
    z2 = relu(a2)
    z2_til = np.concatenate((z2, np.ones((1, z2.shape[1]))), axis=0)
    pred_y = np.matmul(theta3, z2_til)
    return pred_y, a1, z1, a2, z2


# squared error loss
def sqr_error_loss(y, pred_y):
    row = y.shape[0]
    col = y.shape[1]
    pred_y = pred_y.reshape((row, col))
    R = np.mean((y - pred_y) ** 2)
    return R


# derivative of ReLU function
def de_relu(a):
    return (a > 0) * np.sign(a)


# backward process
def backward_process(a3, a1, z1, a2, z2, W1, W2, W3, b1, b2, b3):
    delta3 = - 2 * (training_y.reshape((1, -1)) - a3)
    dR_dw3 = np.matmul(z2, np.transpose(delta3))
    dR_db3 = delta3
    W3 = W3 - learning_rate / train_num * dR_dw3
    # W3 in the above line is calculated by gradient decent
    b3 = b3 - learning_rate * np.mean(dR_db3)
    # updata offset term in third layer
    delta2 = np.multiply(np.matmul(W3, delta3), de_relu(a2))
    dR_dw2 = np.matmul(z1, np.transpose(delta2))
    W2 = W2 - learning_rate / train_num * dR_dw2
    # W2 is calculated by gradient decent
    dR_db2 = delta2
    b2 = b2 - learning_rate * np.mean(dR_db2, axis=1).reshape(sec_ly_node_num,1)
    delta1 = np.multiply(np.matmul(W2, delta2), de_relu(a1))
    dR_dw1 = np.matmul(X[:, 0: train_num], np.transpose(delta1))
    W1 = W1 - learning_rate / train_num * dR_dw1
    # W1 is calculated in gradient decent method
    dR_db1 = delta1
    b1 = b1 - learning_rate * np.mean(dR_db1, axis=1).reshape(frst_ly_node_num, 1)

    return W1, b1, W2, b2, W3, b3


error = 0
iter_num = 0
while iter_num < 5000:
    a3, a1, z1, a2, z2 = forward_process(training_x, theta1, theta2, theta3)
    W1, b1, W2, b2, W3, b3 = backward_process(a3, a1, z1, a2, z2, W1, W2, W3, b1, b2, b3)
    theta1 = np.concatenate((np.transpose(W1), b1), axis=1)
    theta2 = np.concatenate((np.transpose(W2), b2), axis=1)
    theta3 = np.concatenate((np.transpose(W3), b3), axis=1)
    error = sqr_error_loss(training_y, a3)
    iter_num = iter_num + 1

print("the number of iteration is ", iter_num)
print("the training error is ", error)
test_pred, test_a1, test_z1, test_a2, test_z2 = forward_process(test_x, theta1, theta2, theta3)
test_error = sqr_error_loss(test_y, test_pred)
print("the test error is ", test_error)