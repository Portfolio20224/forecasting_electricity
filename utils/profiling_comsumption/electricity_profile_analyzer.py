import pandas as pd
import numpy as np
from .cluster_analyzer import ClusterAnalyzer
from .data_preparator import DataPreparator
from .vizualizer import Visualizer

class ElectricityProfileAnalyzer:
    """Main class for complete electricity profile analysis."""
    
    def __init__(self, resolution='h', max_clusters=10, random_state=42):
        """
        Initialize the analyzer.
        
        Parameters:
        resolution: 'h' for hourly, '15min' for quarter-hour
        max_clusters: maximum number of clusters to consider
        random_state: random seed for reproducibility
        """
        self.resolution = resolution
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.data_preparator = DataPreparator(resolution)
        self.cluster_analyzer = ClusterAnalyzer(max_clusters, random_state)
        self.visualizer = Visualizer()
        self.daily_profiles = None
        self.labels = None
        self.centroids = None
    
    def analyze(self, df, n_clusters=None, method='kmeans', year=None):
        """
        Complete analysis of electrical profiles.
        
        Parameters:
        df: DataFrame with consumption data
        n_clusters: number of clusters (if None, determined automatically)
        method: 'kmeans' or 'hierarchical'
        year: specific year to focus on for visualizations
        
        Returns:
        daily_profiles, labels, centroids
        """

        
        print("\n1. Preparing daily profiles...")
        self.daily_profiles = self.data_preparator.prepare_daily_profiles(df)
        print(f"   - {len(self.daily_profiles)} days analyzed")
        print(f"   - {len(self.daily_profiles.columns)} points per day")
        
        print("\n2. Applying clustering...")
        self.labels, self.centroids = self.cluster_analyzer.cluster_daily_profiles(
            self.daily_profiles, 
            n_clusters=n_clusters, 
            method=method
        )
        n_clusters_actual = len(self.centroids)
        print(f"   - {n_clusters_actual} clusters identified")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        for i, count in zip(unique, counts):
            print(f"   - Class {i+1}: {count} days ({count/len(self.labels)*100:.1f}%)")
        
        print("\n3. Generating visualizations...")
        
        self.visualizer.plot_calendar_view(self.daily_profiles, self.labels, self.centroids, year)
        
        self.visualizer.plot_seasonal_analysis(self.daily_profiles, self.labels, self.centroids)
        
        self.visualizer.plot_typical_days_comparison(self.centroids, self.labels, self.daily_profiles)
        
        print("\n4. Exporting results...")
        self._export_results()
        
        
        return self.daily_profiles, self.labels, self.centroids
    
    def _export_results(self):
        """Export clustering results and typical profiles to CSV."""
        results = pd.DataFrame({
            'date': self.daily_profiles.index,
            'cluster': self.labels
        })
        results.to_csv('clustering_results.csv', index=False)
        print("   - Results saved to 'clustering_results.csv'")
        
        centroids_df = pd.DataFrame(self.centroids,
                                   columns=[f'H{i}' for i in range(self.centroids.shape[1])])
        centroids_df.to_csv('typical_profiles.csv', index=False)
        print("   - Typical profiles saved to 'typical_profiles.csv'")