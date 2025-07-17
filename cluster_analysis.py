import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For plotting
from scipy.cluster.hierarchy import dendrogram, linkage  # For hierarchical clustering
from sklearn.preprocessing import StandardScaler         # For data normalization
from sklearn.cluster import KMeans                       # For k-means clustering

# Correct file path based on the output
file_path = "london_houses.csv"

# Load the dataset 
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(df.head())

# Inspect column names to confirm the correct one
print(df.columns)
print()  # Print a blank line for spacing
space_size = len(df)
print(f"Total number of rows in the file: {space_size}")
print()  # Print a blank line for spacing

# Check for missing values in 'Price (£)' and 'Square Meters' columns
print("Missing values in 'Square Meters':", df['Square Meters'].isnull().sum())
print("Missing values in 'Price (£)':", df['Price (£)'].isnull().sum())
print()  # Print a blank line for spacing

# Plotting the scatter plot for 'Square Meters' vs 'Price (£)'
#plt.figure(figsize=(8, 6))
#plt.scatter(df['Square Meters'], df['Price (£)'], alpha=0.6)
#plt.xlabel('Square Meters')
#plt.ylabel('Price (£)')
#plt.title('Price (£) vs Square Meters')
#plt.grid(True)
#plt.show()

# Normalize the data using z-score normalization
features = ['Square Meters', 'Price (£)']
scaler = StandardScaler()
X_zscore = scaler.fit_transform(df[features])
print("Means:", scaler.mean_)
print("Standard Deviations:", scaler.scale_)

# Optional: print the first 5 rows to check
#print("First 5 rows of standardized data (z-scores):\n", X_zscore[:5])

# Perform hierarchical clustering with Euclidean distance and single linkage
linked = linkage(X_zscore, method='single', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage, Euclidean Distance)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
