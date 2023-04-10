import numpy as np
from sklearn.cluster import KMeans

class ClusterClients:
    '''
        Clusters the label inclination vector of all 
        clients using clustering algorithms
    '''
    def __init__(self):
        self.clusters = []
        self.cluster_count = 0

    def clusterClients(self, vec_lbl_incl, client_set, no_of_cluster,
                       clustering_algo = 'kmeans'):
        if clustering_algo != 'kmeans':
            raise Exception("Clustering algo not supported...")
        if len(vec_lbl_incl) != len(client_set):
            raise Exception("Invalid inputs...Client count not matching")
        self.X = np.array()
        self.cluster_count = no_of_cluster
        self.kmeans_obj = KMeans(n_clusters=no_of_cluster, init='k-means++', tol=1e-6)
        self.kmeans_obj.fit(self.X)
        self.labels = self.kmeans_obj.labels_

        self.client_dict = {}
        for i in range(range(len(client_set))):
            if self.labels[i] not in self.client_dict.keys():
                self.client_dict[self.labels[i]] = []
            self.client_dict[self.labels[i]].append(client_set[i])
        
        # Reassign self.clusters to ensure multiple execution to be consistent
        self.clusters = []
        for key in self.client_dict.keys():
            self.clusters.append(self.client_dict[key])

        return self.clusters