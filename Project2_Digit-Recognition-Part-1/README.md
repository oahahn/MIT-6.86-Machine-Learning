## Project Objectives

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The goal of this project is to experiment with the task of classifying these images into the correct digit.

Part of the code for this project was written by the course staff. My tasks were the following: 

* Implement a linear regression and understand understand how it is inadequate for this task. This was implemented in linear\_regression.py and tested in main.py. 

* Use scikit-learn's SVM for binary classification and multiclass classification. This was implemented in svm.py and tested in main.py.

* Implement your own softmax regression using gradient descent. This was implemented in softmax.py and tested in main.py.

* Experiment with different hyperparameters and labels. This can again be seen in softmax.py and main.py.

* Perform dimensionality reduction using PCA by projecting the data onto the principal components and explore the effects on performance. This was implemented in features.py and tested in main.py.

* Implement a direct mapping to the high-dimensional features using a cubic feature mapping and understand understand how it is inefficient for this task. This can also be seen in features.py.

* Write a polynomial kernel function and a Gaussian RBF kernel function to map the features into d dimensional polynomial space and calculate the change in the test error. This was implemented in kernel.py and tested in main.py.

In the next project, neural networks are applied to this task.