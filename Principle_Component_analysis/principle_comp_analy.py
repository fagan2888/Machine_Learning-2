import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

yale = sio.loadmat('yalefaces.mat')
yalefaces = yale['yalefaces']

fig, ax = plt.subplots()
for i in range(0, yalefaces.shape[2]):
    x = yalefaces[:, :, i]
    ax.imshow(x, cmap=plt.get_cmap('gray'))
    # time.sleep(0.1)
    plt.show()

num, row, col = yalefaces.shape
row_matr = np.reshape(yalefaces[:, :, 0], (1, num * row))
for i in range(1, col):
    row_vec = np.reshape(yalefaces[:, :, i], (1, num * row))
    row_matr = np.concatenate((row_matr, row_vec), axis=0)

col_matr = np.transpose(row_matr)  # each column represent an image
cov_matr = np.cov(col_matr)  # covariance matrix
w, v = np.linalg.eig(cov_matr)  # w is eigenvalue
sorted_eigval = np.flip(np.reshape(np.sort(w), (1, -1)), axis=1)[0]  # sort descending
xarr = np.arange(0, len(sorted_eigval)) + 1  # x axis
plt.semilogy(xarr, sorted_eigval)  # plot
plt.show()  # show image
total_num = len(sorted_eigval)
eigval_sum = 0
index = 0
while eigval_sum < 0.95 * np.sum(sorted_eigval):
    eigval_sum = eigval_sum + sorted_eigval[index]
    index = index + 1
comp_num = index + 1  # 95% accuracy
perc_red = (total_num - comp_num) / total_num
print("the number of eigenvalues is " + str(total_num))
print("the components needed for 95% total variance is " + str(comp_num))
print("the precentage reduction is " + str(perc_red))
eigval_sum1 = 0
index1 = 0
while eigval_sum1 < 0.99 * np.sum(sorted_eigval):
    eigval_sum1 = eigval_sum1 + sorted_eigval[index]
    index1 = index1 + 1
comp_num1 = index1 + 1  # 99% accuracy
perc_red1 = (total_num - comp_num1) / total_num
print("the components needed for 99% total variance is " + str(comp_num1))
print("the precentage reduction is " + str(perc_red1))


# show images in problem (b)

f, axarr = plt.subplots(4, 5)

eigenvec_ind = np.argsort(-w)[0:19]
sample_mean = np.transpose(col_matr.mean(1))
mean_image = np.reshape(sample_mean, (num, row))
axarr[0, 0].imshow(mean_image, cmap=plt.get_cmap('gray'))
for i in range(1,20):
    xx = np.reshape(np.transpose(v[:, eigenvec_ind[i - 1]]), (num, row))
    row_ind = int(i/5)
    col_ind = i%5
    axarr[row_ind, col_ind].imshow(xx, cmap=plt.get_cmap('gray'))

plt.show()