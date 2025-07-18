import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For plotting
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  # For hierarchical clustering
from sklearn.preprocessing import StandardScaler         # For data normalization
from sklearn.cluster import KMeans

# Correct file path based on the output
file_path = "london_houses.csv"

# Load the dataset 
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(df.head())

# Print total number of rows in the whole dataset
print(f"Total number of rows in the whole dataset: {len(df)}")

# Filter for the 'Chelsea' neighborhood only
df_chelsea = df[df['Neighborhood'] == 'Chelsea']
print(f"Total number of rows in Chelsea: {len(df_chelsea)}")
print()

# Check for missing values in 'Price (£)' and 'Square Meters' columns for Chelsea
print("Missing values in 'Square Meters':", df_chelsea['Square Meters'].isnull().sum())
print("Missing values in 'Price (£)':", df_chelsea['Price (£)'].isnull().sum())
print()

# Scatter plot of actual values for Chelsea
plt.figure(figsize=(8, 6))
plt.scatter(df_chelsea['Square Meters'], df_chelsea['Price (£)'], alpha=0.6)
plt.xlabel('Square Meters')
plt.ylabel('Price (£)')
plt.title('Price (£) vs Square Meters (Chelsea)')
plt.grid(True)
plt.show()

# Normalize the data using z-score normalization for Chelsea
features = ['Square Meters', 'Price (£)']
scaler = StandardScaler()
X_zscore = scaler.fit_transform(df_chelsea[features])
print("Means:", scaler.mean_)
print("Standard Deviations:", scaler.scale_)

# Scatter plot of standardized values for Chelsea
plt.figure(figsize=(8, 6))
plt.scatter(X_zscore[:, 0], X_zscore[:, 1], alpha=0.6)
plt.xlabel('Standardized Square Meters (z-score)')
plt.ylabel('Standardized Price (£) (z-score)')
plt.title('Standardized Price (£) vs Standardized Square Meters (Chelsea)')
plt.grid(True)
plt.show()

# Perform hierarchical clustering with Euclidean distance and single linkage for Chelsea
linked = linkage(X_zscore, method='single', metric='euclidean')

# Plot the dendrogram for Chelsea
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Chelsea, Single Linkage, Euclidean Distance)')
plt.xlabel('Sample index')
plt.ylabel('Distance')

# Add a horizontal dashed line at y=0.34
plt.axhline(y=0.34, color='r', linestyle='--', linewidth=2, label='Cut at y=0.34')
plt.legend()

plt.show()

# Assign cluster labels based on the cut at y=0.34
cluster_labels = fcluster(linked, t=0.34, criterion='distance')

# Scatter plot of standardized values for Chelsea, colored by cluster
plt.figure(figsize=(8, 6))
for cluster in np.unique(cluster_labels):
    plt.scatter(
        X_zscore[cluster_labels == cluster, 0],
        X_zscore[cluster_labels == cluster, 1],
        label=f'Cluster {cluster}',
        alpha=0.7
    )
plt.xlabel('Standardized Square Meters (z-score)')
plt.ylabel('Standardized Price (£) (z-score)')
plt.title('Chelsea: Clusters from Dendrogram Cut at 0.34')
plt.legend()
plt.grid(True)
plt.show()

# Set number of clusters (e.g., 7 to match dendrogram)
k = 7
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_zscore)

# Scatter plot of standardized values for Chelsea, colored by k-means cluster
plt.figure(figsize=(8, 6))
for cluster in np.unique(kmeans_labels):
    plt.scatter(
        X_zscore[kmeans_labels == cluster, 0],
        X_zscore[kmeans_labels == cluster, 1],
        label=f'Cluster {cluster+1}',
        alpha=0.7
    )
plt.xlabel('Standardized Square Meters (z-score)')
plt.ylabel('Standardized Price (£) (z-score)')
plt.title('Chelsea: K-Means Clusters (k=7)')
plt.legend()
plt.grid(True)
plt.show()
