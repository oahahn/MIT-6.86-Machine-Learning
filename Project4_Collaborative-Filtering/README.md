## Project Objectives

The aim of this project was to build a Gaussian mixture model for collaborative filtering using a portion of the Netflix database. The users have only rated a small fraction of the movies so the Expectation Maximization (EM) algorithm was utilised to predict values for the missing entries. 

Part of the code for this project was written by the course staff. Most of my work can be seen in:

1. **naive_em.py** where I implemented an E-step, M-step and run function for a basic EM algorithm
2. **common.py** where I implemented the Bayesian Information Criterion to tune the hyperparameter K
3. **em.py** where I built upon versions of the E-step and M-step function to accomodate partially observed vectors and made the computations more numerically stable
4. **main.py** where I ran my algorithms and compared results as I progressed through the project



