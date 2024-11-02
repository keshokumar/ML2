import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load your dataset
data = pd.read_csv(r"C:\Users\kesho\Downloads\Mall_Customers.csv")
print(data.columns)

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Plot the data
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Customer Data")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=4, max_iter=10, random_state=42)

# Initial clusters
initial_labels_kmeans = kmeans.fit_predict(X)

# Plotting initial clusters for K-Means
plt.scatter(X[:, 0], X[:, 1], c=initial_labels_kmeans, cmap='viridis', s=30)
plt.title("K-Means Initial Clusters (Epoch Size: 10)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Fit with more iterations to converge
kmeans = KMeans(n_clusters=4, max_iter=100, random_state=42)
final_labels_kmeans = kmeans.fit_predict(X)

# Final clusters for K-Means
plt.scatter(X[:, 0], X[:, 1], c=final_labels_kmeans, cmap='viridis', s=30)
plt.title("K-Means Final Clusters (Epoch Size: 100)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Error rate (inertia) for K-Means
error_rate_kmeans = kmeans.inertia_
print(f"K-Means Final Error Rate (Inertia): {error_rate_kmeans}")

# Agglomerative Clustering - Final Clusters
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_clustering.fit_predict(X)

# Plotting final clusters for Agglomerative Clustering
plt.scatter(X[:, 0], X[:, 1], c=agg_labels, cmap='viridis', s=30)
plt.title("Agglomerative Clustering Final Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Calculate Silhouette Score for Agglomerative Clustering
silhouette_avg = silhouette_score(X, agg_labels)
print(f"Agglomerative Clustering Silhouette Score: {silhouette_avg}")

# Compare the results of both algorithms
print("\nComparison of Clustering Algorithms:")
print(f"K-Means Final Inertia: {error_rate_kmeans}")
print(f"Agglomerative Clustering Silhouette Score: {silhouette_avg}")

# Visualization of Inertia and Silhouette Score
plt.figure(figsize=(8, 5))
labels = ['K-Means Inertia', 'Agglomerative Silhouette Score']
values = [error_rate_kmeans, silhouette_avg]

plt.bar(labels, values, color=['blue', 'green'])
plt.title('Comparison of Clustering Algorithms')
plt.ylabel('Score')
plt.ylim(0, max(values) + 10)  # Set y-axis limits for better visualization
plt.grid(axis='y')
plt.show()
