import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from pylab import *
from random import sample
from sklearn.decomposition import PCA
from ggplot import *
np.random.seed(123)

def getDatapoints(estimator, label):
    """
    Returns data points belonging to cluster "label".
    estimator: k-means estimator
    label: the cluster we are looking at.
    """
    indices = np.where(estimator.labels_ == label)[0]
    return indices

def computeHopkins(fea):
    """
    This code is from below with minor modifications:
    http://datascience.stackexchange.com/questions/14142/cluster-tendency-using-hopkins-statistic-implementation-in-python
    """
    d = fea.shape[1]
    n = fea.shape[0]
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(fea)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.normal(size=d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(fea[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    return H

def pre_visualize(fea, fea_new, config):
    Hidx = computeHopkins(fea)
    print("Hopkins statistic for clustering = %.4f" % Hidx)
    fea_vis = pd.DataFrame(fea_new, columns=["PC1", "PC2"])
    fea_vis["clusterIDX"] = 0
    g = ggplot(fea_vis, aes(x="PC1", y="PC2", color="factor(clusterIDX)")) + \
        geom_point(size=50) + \
        ggtitle("Data points, Hopkins statistic = %.4f" % Hidx)
    if config.F_vis:
        g.save(filename=config.TPRvis_dir + "/clustering_preVis_F.png")
    elif config.R_vis:
        g.save(filename=config.TPRvis_dir + "/clustering_preVis_R.png")

def post_visualize(fea, fea_new, f, c_new, centers, config):
    num = len(centers)
    # Silhouette score:
    # best value=1, worst value=-1. If close to zero, means overlapping clusters.
    # Negative value means a sample is assigned to wrong cluster.
    ss = silhouette_score(fea, f.labels_)
    fea_vis = pd.DataFrame(fea_new, columns=["PC1", "PC2"])
    fea_vis["clusterIDX"] = f.labels_
    g = ggplot(fea_vis, aes(x="PC1", y="PC2", color="factor(clusterIDX)")) + \
        geom_point(size=50) + \
        ggtitle("Clustered data points, Silhouette score = %.4f" % ss)
    if config.F_vis:
        g.save(filename=config.TPRvis_dir + "/clustering_postVis_F.png")
    elif config.R_vis:
        g.save(filename=config.TPRvis_dir + "/clustering_postVis_R.png")
    # Number of data points in each cluster
    for i in range(num):
        indices = getDatapoints(estimator=f, label=i)
        print("Number of data points in cluster %d is %d" % (i, len(indices)))
        # centers[i]

def do_cluster(num, fea, config):
    """
    num: list of number of clusters.
    fea: feature vectors to cluster [nTr, nFea]
         where nTr is the # of feature vectors and
         nFea is the # of features in each feature vector.
    """
    fea = scale(fea) # subtract mean, divide by std.
    pca = PCA(n_components=2)
    pca.fit(fea)
    fea_new = pca.transform(fea)
    pre_visualize(fea, fea_new, config)
    # %%%%%% select number of clusters.
    n = len(num)
    ss = [0.0]*n
    for i in range(n):
        f = KMeans(init='k-means++', n_clusters=num[i], n_init=10)
        f.fit(fea)
        ss[i] = silhouette_score(fea, f.labels_)
        print("with %d clusters Silhouette score is: %.4f" % (num[i], ss[i]))
    val = max(ss)
    idx = ss.index(val)
    print("Performing clustering with %d clusters" % num[idx])
    f = KMeans(init='k-means++', n_clusters=num[idx], n_init=10)
    f.fit(fea)
    centers = f.cluster_centers_
    c_new = pca.transform(centers)
    # %%%%%%
    post_visualize(fea, fea_new, f, c_new, centers, config)
    print("Clustering Done!")