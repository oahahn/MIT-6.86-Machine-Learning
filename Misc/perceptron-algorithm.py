import numpy as np

def perceptron_origin(x, y, t):
    """
    :param x: training set
    :param y: labels
    :param t: number of trials
    :return: list of necessary updates + how many mistakes were made
    """
    theta = np.array([0, 0])
    updates = []
    mistakes = 0
    for e in range(t):
        temp = 0
        for i in range(np.size(x, 0)):
            if y[i] * (np.dot(x[i], theta)) <= 0:
                mistakes += 1
                temp += 1
                theta = theta + y[i] * x[i]
                updates.append(np.ndarray.tolist(theta))
        if temp == 0:
            break
    return updates, mistakes

def perceptron(x, y, t):
    """
    :param x: training set
    :param y: labels
    :param t: number of trials
    :return: list of necessary updates + how many mistakes were made
    """
    theta = np.array([-1, 2])
    theta0 = -3
    theta_updates = []
    theta0_updates = []
    mistakes = 0
    for e in range(t):
        temp = 0
        for i in range(np.size(x, 0)):
            if y[i] * (np.dot(x[i], theta) + theta0) <= 0:
                mistakes += 1
                temp += 1
                theta = theta + y[i] * x[i]
                theta0 = theta0 + y[i]
                theta_updates.append(np.ndarray.tolist(theta))
                theta0_updates.append(theta0)
        if temp == 0:
            break
    return theta_updates, theta0_updates, mistakes, theta, theta0


'''
x1 = np.array([[-1, -1], [1, 0], [-1, 10]])
x2 = np.array([[1, 0], [-1, 10], [-1, -1]])
y1 = np.array([1, -1, 1])
y2 = np.array([-1, 1, 1])
'''
x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y = np.array([1, 1, -1, -1, -1])
index1 = np.array([1, 2, 3, 4, 0])
index2 = np.array([2, 3, 4, 0, 1])
index3 = np.array([3, 4, 0, 1, 2])
index4 = np.array([4, 0, 1, 2, 3])

theta_updates, theta0_updates, mistakes, theta, theta0 = perceptron(x, y, 1)
print("A list of the updated versions of theta ", theta_updates)
print("A list of the different values of theta0 ", theta0_updates)
print("The total mistakes were ", mistakes)
print("The final value for theta was ", theta)
print("The final value for theta0 was ", theta0)
