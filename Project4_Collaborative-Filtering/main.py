import numpy as np
import kmeans
import common
#import naive_em
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

run_kmeans()