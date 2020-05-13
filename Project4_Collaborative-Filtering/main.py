import numpy as np
import kmeans
import common
import naive_em
#import em

X = np.loadtxt("toy_data.txt")

def run_kmeans():
    for K in range(1, 5):
        min_cost = None
        best_seed = None
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = "K-means for K = {}, seed = {}, cost = {}".format(K, best_seed, min_cost)
        print(title)
        common.plot(X, mixture, post, title)

# run_kmeans()

# Output
# K-means for K = 1, seed = 0, cost = 5462.297452340002
# K-means for K = 2, seed = 0, cost = 1684.9079502962372
# K-means for K = 3, seed = 3, cost = 1329.59486715443
# K-means for K = 4, seed = 4, cost = 1035.499826539466


def test_step():
    mixture, post = common.init(X, 3, 0)
    mixture, soft_counts, ll = naive_em.run(X, mixture, post)
    print("Log-likelihood: {}".format(ll))


def run_naive_em():
    for K in range(1, 5):
        max_ll = None
        best_seed = None
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        title = "EM for K = {}, seed = {}, ll = {}".format(K, best_seed, max_ll)
        print(title)
        common.plot(X, mixture, post, title)


# run_naive_em()

# Output
# EM for K = 1, seed = 0, ll = -1307.2234317600937
# EM for K = 2, seed = 2, ll = -1175.7146293666792
# EM for K = 3, seed = 0, ll = -1138.8908996872672
# EM for K = 4, seed = 4, ll = -1138.601175699485
