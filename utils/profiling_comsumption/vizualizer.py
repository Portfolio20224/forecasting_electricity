import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import seaborn as sns
from datetime import datetime, timedelta
import calendar

class Visualizer:
    """Handles all visualization functions."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_calendar_view(self, daily_profiles, labels, centroids, year=None):
        """
        Creates a calendar visualization as in the article figures.
        
        Parameters:
        daily_profiles: DataFrame with index = dates
        labels: cluster labels
        centroids: typical profiles
        year: year to visualize (if None, uses all data)
        """
        dates = pd.to_datetime(daily_profiles.index)
        df_calendar = pd.DataFrame({
            'date': dates,
            'cluster': labels,
            'month': dates.month,
            'day': dates.day,
            'year': dates.year
        })
        
        if year:
            df_calendar = df_calendar[df_calendar['year'] == year]
        
        n_clusters = len(centroids)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        cluster_colors = {i: colors[i] for i in range(n_clusters)}
        
        fig = plt.figure(figsize=(20, 10))
        
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[:, 0])
        
        months = sorted(df_calendar['month'].unique())
        
        y_pos = 0
        cell_size = 12
        
        for month in months:
            month_data = df_calendar[df_calendar['month'] == month]
            month_data = month_data.sort_values('day')
            
            month_name = calendar.month_name[month][:3]
            ax1.text(-3, y_pos + cell_size/2, month_name,
                    ha='right', va='center', fontweight='bold', fontsize=10)
            
            for _, row in month_data.iterrows():
                color = cluster_colors[row['cluster']]
                rect = plt.Rectangle((row['day'] * cell_size, y_pos),
                                    cell_size-1, cell_size-1,
                                    facecolor=color, edgecolor='white', linewidth=0.5)
                ax1.add_patch(rect)
                
                if row['day'] <= 5:
                    ax1.text(row['day'] * cell_size + 2, y_pos + 2, str(row['day']),
                            fontsize=6, color='black', alpha=0.5)
            
            y_pos += cell_size * 1.5
        
        ax1.set_xlim(0, 32 * cell_size)
        ax1.set_ylim(0, y_pos)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Calendar of typical profiles', fontsize=14, fontweight='bold')
        
        legend_elements = []
        for i in range(n_clusters):
            legend_elements.append(plt.Rectangle((0,0), 1, 1,
                                               facecolor=cluster_colors[i],
                                               label=f'Class {i+1}'))
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        for i in range(n_clusters):
            row = i // 2
            col = i % 2
            
            ax = fig.add_subplot(gs[row, col + 1])
            
            cluster_dates = df_calendar[df_calendar['cluster'] == i]['date']
            cluster_profiles = daily_profiles.loc[cluster_dates]
            
            hours = range(len(centroids[i]))
            for _, profile in cluster_profiles.iterrows():
                ax.plot(hours, profile.values, color='lightgray', alpha=0.15, linewidth=0.5)
            
            ax.plot(hours, centroids[i], color=cluster_colors[i], linewidth=2.5,
                   label=f'Typical profile {i+1}')
            
            n_days = len(cluster_profiles)
            ax.text(0.95, 0.95, f'n = {n_days}', transform=ax.transAxes,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Hour')
            ax.set_ylabel('Power (kW)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            if len(cluster_profiles) > 0:
                y_max = cluster_profiles.values.max() * 1.1
                ax.set_ylim(0, y_max)
        
        plt.suptitle('Analysis of daily load profiles by clustering',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def plot_seasonal_analysis(self, daily_profiles, labels, centroids):
        """
        Seasonal analysis as in figure 2 of the article.
        """
        dates = pd.to_datetime(daily_profiles.index)
        n_clusters = len(centroids)
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        df_season = pd.DataFrame({
            'date': dates,
            'cluster': labels,
            'season': [get_season(d.month) for d in dates],
            'month': dates.month
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        for idx, season in enumerate(seasons):
            ax = axes[idx]
            season_data = df_season[df_season['season'] == season]
            
            if len(season_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{season}')
                continue
            
            cluster_counts = season_data['cluster'].value_counts().sort_index()
            
            full_counts = np.zeros(n_clusters)
            for i in range(n_clusters):
                if i in cluster_counts.index:
                    full_counts[i] = cluster_counts[i]
            
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            bars = ax.bar(range(n_clusters), full_counts, color=colors)
            
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of days')
            ax.set_title(f'{season} ({len(season_data)} days)')
            ax.set_xticks(range(n_clusters))
            ax.set_xticklabels([f'{i+1}' for i in range(n_clusters)])
            
            total = len(season_data)
            for i, (bar, count) in enumerate(zip(bars, full_counts)):
                if count > 0:
                    percentage = count / total * 100
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{percentage:.0f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Seasonal distribution of typical profiles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_typical_days_comparison(self, centroids, labels, daily_profiles):
        """
        Compares typical profiles for different day types
        (weekday vs weekend, as in figure 4).
        """
        dates = pd.to_datetime(daily_profiles.index)
        n_clusters = len(centroids)
        
        df_days = pd.DataFrame({
            'date': dates,
            'cluster': labels,
            'weekday': dates.dayofweek,
            'is_weekend': dates.dayofweek >= 5
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax = axes[0]
        week_mask = ~df_days['is_weekend']
        weekend_mask = df_days['is_weekend']
        
        week_profiles = daily_profiles.loc[df_days[week_mask]['date']]
        weekend_profiles = daily_profiles.loc[df_days[weekend_mask]['date']]
        
        hours = range(len(centroids[0]))
        
        if len(week_profiles) > 0:
            week_mean = week_profiles.mean()
            week_std = week_profiles.std()
            ax.plot(hours, week_mean, 'b-', linewidth=2, label='Weekday')
            ax.fill_between(hours, week_mean - week_std, week_mean + week_std, alpha=0.2, color='blue')
        
        if len(weekend_profiles) > 0:
            weekend_mean = weekend_profiles.mean()
            weekend_std = weekend_profiles.std()
            ax.plot(hours, weekend_mean, 'r-', linewidth=2, label='Weekend')
            ax.fill_between(hours, weekend_mean - weekend_std, weekend_mean + weekend_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Hour')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Weekday vs Weekend comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        
        cluster_by_day = pd.crosstab(
            df_days['is_weekend'].map({True: 'Weekend', False: 'Weekday'}),
            df_days['cluster']
        )
        
        for i in range(n_clusters):
            if i not in cluster_by_day.columns:
                cluster_by_day[i] = 0
        
        cluster_by_day = cluster_by_day[sorted(cluster_by_day.columns)]
        
        cluster_by_day_pct = cluster_by_day.div(cluster_by_day.sum(axis=1), axis=0) * 100
        
        cluster_by_day_pct.T.plot(kind='bar', ax=ax)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Cluster distribution')
        ax.legend(title='Day type')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[2]
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, centroid in enumerate(centroids):
            ax.plot(hours, centroid, color=colors[i], linewidth=2, label=f'Class {i+1}')
        
        ax.set_xlabel('Hour')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Typical profiles (centroids)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comparative analysis of load profiles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()