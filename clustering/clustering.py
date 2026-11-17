import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#TASK 1 of fitting cluster centers

def plot_clusters(X, cluster_assignments, centroid_locations, title):
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    plt.viridis() # Set colour map

    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        s=20,
        c=cluster_assignments,
        alpha=0.8,
        cmap='viridis'
    )

    ax.scatter(
        centroid_locations[:,0],
        centroid_locations[:, 1],
        centroid_locations[:, 2],
        s=200,
        marker='o',
        c=range(len(centroid_locations)),
        edgecolors='k'
    )


    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")


    plt.show()





training_data = np.loadtxt("clustering_train.txt")
X = training_data[:, : 3]


kmeans = KMeans(
    n_clusters = 3,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

centres = kmeans.cluster_centers_

print("Cluster centres:")
print(centres)

#GMM

training_data = np.loadtxt("clustering_train.txt")
X = training_data[:, : 3]


gmm = GaussianMixture(
    n_components =3,
    random_state=42,
    n_init=10
)
gmm.fit(X)
print("Gaussian means:", gmm.means_)
plot_clusters(X, kmeans.labels_, kmeans.cluster_centers_, "Kmeans Clustering")
plot_clusters(X, gmm.predict(X), gmm.means_, "gmm Clustering")



"""
Fun notes
The k means will basically produce
2 diagonals 
plus a weird diagonal and center


Whereas the gmm produces 2 overlapping clusters in diagonals 
and then one grouped by itself

this probs means gmm . is better it has cov
"""