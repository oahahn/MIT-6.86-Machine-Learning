from utils import *
import pandas as pd

train_x, train_y, test_x, test_y = get_MNIST_data()
# X = train_x[0:20, :]
# print(X.shape[1])
# print(train_x[0:5])
# print(train_y[0:5])
# print(test_x[0:5])
# print(test_y[0:5])
print(train_x[[0:5],[0:5]])