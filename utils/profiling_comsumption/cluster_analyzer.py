import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


class ClusterAnalyzer:
    """Performs clustering analysis on daily profiles."""
    
    def __init__(self, max_clusters=10, random_state=42):
        """
        Initialize the cluster analyzer.
        
        Parameters:
        max_clusters: maximum number of clusters to consider
        random_state: random seed for reproducibility
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.labels = None
        self.centroids = None
    
    @staticmethod
    def cosine_similarity_matrix(X):
        """
        Calculates cosine similarity matrix.
        (as in the article, equation 2)
        """
        return 1 - pairwise_distances(X, metric='cosine')
    
    def find_optimal_n_clusters(self, X):
        """
        Finds the optimal number of clusters.
        Based on inertia (elbow method) and silhouette score.
        """
        inertias = []
        silhouette_scores = []
        K_range = range(2, self.max_clusters + 1)
        
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                silhouette_scores.append(silhouette_score(X, kmeans.labels_, metric='cosine'))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow method')
        
        ax2.plot(range(2, self.max_clusters+1), silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        
        plt.tight_layout()
        plt.show()
        
        optimal_k = np.argmax(silhouette_scores) + 2  
        print(f"Recommended number of clusters: {optimal_k}")
        
        return optimal_k
    
    def cluster_daily_profiles(self, daily_profiles, n_clusters=None, method='kmeans'):
        """
        Applies clustering to daily profiles.
        
        Parameters:
        daily_profiles: DataFrame with profiles
        n_clusters: number of clusters (if None, determined automatically)
        method: 'kmeans' or 'hierarchical'
        
        Returns:
        labels: cluster labels for each day
        centroids: typical profiles (centroids)
        """
        X = daily_profiles.values
        
        # X_norm = normalize(X, norm='l2')
        X_norm = X
        
        if n_clusters is None:
            n_clusters = self.find_optimal_n_clusters(X_norm)
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            kmeans.fit(X_norm)
            self.labels = kmeans.labels_
            self.centroids = kmeans.cluster_centers_
            
            for i in range(n_clusters):
                cluster_mask = self.labels == i
                if np.any(cluster_mask):
                    cluster_norms = np.linalg.norm(X[cluster_mask], axis=1)
                    self.centroids[i] = self.centroids[i] * np.mean(cluster_norms)
        else:
            
            hc = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            self.labels = hc.fit_predict(X_norm)
            
            self.centroids = np.zeros((n_clusters, X.shape[1]))
            for i in range(n_clusters):
                self.centroids[i] = np.median(X[self.labels == i], axis=0)
        
        return self.labels, self.centroids