import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn appartient au paquet scikit-learn 
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# quelques globules
X, labels_true = make_blobs(n_samples=15, centers=3, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X) 

x, y = zip(*X)
Nobs=len(x)

plt.scatter(x,y, marker='+',c="black")
plt.show()

agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
agglo.fit(X)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
plot_dendrogram(agglo)#,truncate_mode="level", p=3)

#decoupage selon un nombre de clusters souhaites.

aggloK3 = AgglomerativeClustering(n_clusters=3, metric = 'euclidean', linkage = 'complete')
HClabels=aggloK3.fit_predict(X)

plt.scatter(x, y, marker='+', c=HClabels)
plt.title("HC - Linkage method - Force K=3")
