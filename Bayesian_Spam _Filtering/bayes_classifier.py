import time
start = time.clock()
import numpy as np

z = np.genfromtxt('spambase.data', dtype = float, delimiter = ',')
np.random.seed(0)
rp = np.random.permutation(z.shape[0])
z = z[rp,:]
x = z[:,:-1]
y = z[:,-1]
original_training_x = x[0:2000,:]
training_y = y[0:2000]
original_test_x = x[2000:4601,:]
test_y = y[2000:4601]

# median_x is a list that contain the median of column of x
median_x = np.median(original_training_x, axis=0)

# training_x is matrix produced from original_training_x that contains 1 and 2
training_x = np.copy(original_training_x)
for i in range(original_training_x.shape[0]):
    for j in range(original_training_x.shape[1]):
        if training_x[i,j] > median_x[j]:
            training_x[i,j] = 2
        else:
            training_x[i,j] = 1

# test_x is matrix produced from original_test_x that contains 1 and 2
test_x = np.copy(original_test_x)
for i in range(original_test_x.shape[0]):
    for j in range(original_test_x.shape[1]):
        if test_x[i,j] > median_x[j]:
            test_x[i,j] = 2
        else:
            test_x[i,j] = 1

indices_y0 = [i for i, x in enumerate(training_y) if x == 0]
indices_y1 = [i for i, x in enumerate(training_y) if x == 1]
num_y1 = len(indices_y1)
num_y0 = len(indices_y0)

# training_x_0/1 is the rows of training_x whose label is 0 or 1
training_x_0 = training_x[indices_y0,:]
training_x_1 = training_x[indices_y1,:]
pie0 = num_y0/len(training_y)
pie1 = 1 - pie0

# conditional probability of x equals 1 given y is 0 or 1
conditional_prob_x1_y0 = []
conditional_prob_x1_y1 = []
for i in range(training_x_0.shape[1]):
    ind1 = np.where(training_x_0[:,i] == 1)
    conditional_prob_x1_y0.append(len(ind1[0])/num_y0)
    ind2 = np.where(training_x_1[:, i] == 1)
    conditional_prob_x1_y1.append(len(ind2[0])/num_y1)


# calculate the conditional probability
def conditional_prob(test, value_y):
    product = 1
    if value_y == 0:
        for i in range(training_x.shape[1]):
            if test[i] == 1:
                product = product * conditional_prob_x1_y0[i]
            else:
                product = product * (1 - conditional_prob_x1_y0[i])
    if value_y == 1:
        for i in range(training_x.shape[1]):
            if test[i] == 1:
                product = product * conditional_prob_x1_y1[i]
            else:
                product = product * (1 - conditional_prob_x1_y1[i])
    return product


# calculate the prediction from test data
def prediction(test):
    p0 = pie0 * conditional_prob(test, 0)
    p1 = pie1 * conditional_prob(test, 1)
    if p0>= p1: return 0
    else: return 1

# prediction of the test data
prediction_y = np.zeros(len(test_y))
for i in range(len(test_y)):
    prediction_y[i] = prediction(test_x[i,:])

# the wrong prediction position
error_ind = [i for i in range(len(test_y)) if prediction_y[i] != test_y[i]]
test_error = len(error_ind) / len(test_y)
print("test error of prediction is " + str(test_error))

#sanity check
print("the number of mails with label y=0 is " + str(num_y0))
print("the number of mails with label y=1 is " + str(num_y1))
print("Since y0 > y1, we predict the 0 for all test data.")
test_1_ind = [i for i in range(len(test_y)) if test_y[i] == 1]
err = len(test_1_ind)/len(test_y)
print("test error will be " + str(err))

# calculate used time
elapsed = (time.clock() - start)
print("Time used:",elapsed)