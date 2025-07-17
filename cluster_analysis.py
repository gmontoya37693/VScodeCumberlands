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

# Check for missing values in 'Price' and 'square meters' columns
print("Missing values in 'Price':", df['Price (£)'].isnull().sum())
print("Missing values in 'square meters':", df['Square Meters'].isnull().sum())