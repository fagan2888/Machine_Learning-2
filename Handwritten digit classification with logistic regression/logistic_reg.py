import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
d, n = x.shape
lamb = 10

add_train_x = np.ones((1, 2000))
add_test_x = np.ones((1, 1000))
training_x = np.vstack([add_train_x, x[:, 0:2000]])
training_y = y[:, 0:2000]
test_x = np.vstack([add_test_x, x[:, 2000: ]])
test_y = y[:, 2000: ]

# initialize the value of theta
theta_zero = np.zeros((d + 1, 1))


# compute the value of conditional probability
def ita(theta, x):
    theta = theta.reshape(1, d+1)
    x = x.reshape(d+1,1)
    prod = np.matmul(theta, x)[0][0]
    product = np.float128(prod)
    a = 1/(1 + np.exp(-product))
    return float(a)


# compute the first order derivative of J
def first_order_der(theta):
    ita_vec = []
    for i in range(2000):
        ita_vec.append(ita(theta, training_x[:,i]))
    ita_vec = np.array(ita_vec).reshape(2000,1)
    itab = ita_vec - ((training_y + 1) / 2).transpose()
    vec = np.matmul(training_x, itab) + 2 * lamb * theta
    return vec


# compute the Hessian of J
def second_order_der(theta):
    matri = np.copy(training_x)
    for i in range(2000):
        itaa = ita(theta, training_x[:,i])
        matri[:, i] = matri[:, i] * itaa * (1 - itaa)
    matr_prod = np.matmul(matri, training_x.transpose())
    mat = matr_prod + 2 * lamb * np.identity(785)
    return mat

theta_one = theta_zero - np.matmul(np.linalg.inv(second_order_der(theta_zero)), first_order_der(theta_zero))


# compute the likelyhood of function
def likehood_fun(theta):
    summ = 0
    theta = theta.reshape((1, d+1))
    prod = np.matmul(theta, training_x)
    for i in range(2000):
        summ = summ + np.log(1 + np.exp(np.float128(-training_y[0][i] * prod[0][i])))
    return summ


# compute the objective function
def obj_fun(theta):
    objective = likehood_fun(theta) + lamb * np.sum(theta[:] * theta[:])
    return objective


# iteration to find theta
while np.abs(likehood_fun(theta_one) - likehood_fun(theta_zero)) > 0.01:
    theta_two = theta_one - np.matmul(np.linalg.inv(second_order_der(theta_one)), first_order_der(theta_one))
    theta_zero = theta_one
    theta_one = theta_two


# compute the prediction
def prediction(x):
    x = x.reshape(1, d+1)
    ita_value = 1/(1+np.exp(-np.float128(np.matmul(x, theta_one))))
    return np.float(ita_value)


log_prediction = []
for i in range(1000):
    if prediction(test_x[:,i]) >= 0.5:
        log_prediction.append(1)
    else: log_prediction.append(-1)

count = 0
# compute the wrong prediction index in terms of test data
wrong_pred_index = []
for i in range(1000):
    if test_y[0][i] != log_prediction[i]:
        count = count+1
        wrong_pred_index.append(i)


# the difference between probability and label predicted
wrong_value_diff = np.zeros((1, len(wrong_pred_index)))
for i in range(len(wrong_pred_index)):
     wrong_value = prediction(test_x[:, wrong_pred_index[i]])
     if wrong_value >= 0.5:
        wrong_value_diff[0][i] = 1 - wrong_value
     else: wrong_value_diff[0][i] = wrong_value


# list of index of 20 most confident wrong prediction
max_20_wrong = np.zeros((1, 20))
for i in range(20):
    ind = np.argmin(wrong_value_diff)
    max_20_wrong[0][i] = wrong_pred_index[ind]
    wrong_value_diff[0][ind] = 1

# compute the error
error = count/1000
print("The error of the prediction is " + str(error))
value_obj = obj_fun(theta_one)
print("The value of objective function is " + str(value_obj))


# draw the plots
for i in range(20):
    plt.subplot(4,5,i+1)
    col = 2000 + np.int(max_20_wrong[0][i])
    plt.imshow(np.reshape(x[:, col], (int(np.sqrt(d)), int(np.sqrt(d)))))
    if y[0][col] == 1:
        plt.title("true: 9")
    else: plt.title("true: 4")

plt.show()

