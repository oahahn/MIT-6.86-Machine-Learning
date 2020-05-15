import numpy as np
import kmeans
import common
import naive_em
import em

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


def run_bic():
    max_bic = None
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
        bic = common.bic(X, mixture, ll)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        title = "EM for K = {}, seed = {}, ll = {}, bic = {}".format(K, best_seed, ll, bic)
        print(title)
        common.plot(X, mixture, post, title)

# run_bic()

# Output
# EM for K = 1, seed = 0, ll = -1307.2234317600937, bic = -1315.5056231368872
# EM for K = 2, seed = 2, ll = -1175.7146293666792, bic = -1195.039742579197
# EM for K = 3, seed = 0, ll = -1138.8908996872672, bic = -1169.2589347355095
# EM for K = 4, seed = 4, ll = -1138.601175699485, bic = -1180.012132583452

X = np.loadtxt("netflix_incomplete.txt")

def run_EM_Netflix():
    """Runs the EM algorithm on the incomplete data matrix from Netflix ratings
    """
    for K in [1, 12]:
        max_ll = None
        best_seed = None
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        title = "EM for K = {}, seed = {}, ll = {}".format(K, best_seed, max_ll)
        print(title)


# run_EM_Netflix()

# Output
# EM for K = 1, seed = 0, ll = -1521060.9539852454
# EM for K = 12, seed = 1, ll = -1390234.422346942

def run_matrix_completion():
    mixture, post = common.init(X, 12, 1)
    mixture, post, ll = em.run(X, mixture, post)
    X_pred = em.fill_matrix(X, mixture)
    X_gold = np.loadtxt('netflix_complete.txt')
    print("root mean squared error:", common.rmse(X_gold, X_pred))

run_matrix_completion()

# Output
# root mean squared error: 0.4804908505400684