import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import data_operations as data_ops

class PositionalCluster:

    def __init__(self, data, position, aggfunc='mean', subset='all', panelvar='name', method='kmeans', n_clust=5):

        train = data_ops.prep_data_clustering(data, position, aggfunc, subset, panelvar)
        self.train_data = train
        
        if method != 'kmeans':
            raise ValueError(f"Clustering method {method} not implemented yet")
        else:
            self.clusters = KMeans(n_clust)
        
    def fit_clusters(self, normalizer=None):
        if normalizer is not None:
            self.clusters.fit(normalizer.fit_transform(self.train_data))
        else:
            self.clusters.fit(self.train_data)

    def plot_groups(self, normalizer=None):
        n_clust = self.clusters.cluster_centers_.shape[0]
        x = np.arange(0,1,1/n_clust)

        cluster_centers = self.clusters.cluster_centers_
        if normalizer is not None:
            cluster_centers = normalizer.fit_transform(cluster_centers.T).T

        fig, ax = plt.subplots()

        for i in range(len(self.train_data.columns)):
            print(cluster_centers[:,i])
            ax.bar(x + i*(n_clust+1)*(1/n_clust), cluster_centers[:,i], (1/n_clust), label=self.train_data.columns[i])
        ax.legend()

        return fig