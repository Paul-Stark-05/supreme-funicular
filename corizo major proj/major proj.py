

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Credit Card Customer Data.csv')

# Select the relevant columns for clustering
features = data[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# According to the elbow plot the optimum number of clusters is 3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters

# Save the clustered data to a new CSV file
data.to_csv('clustered_data.csv', index=False)

print("Clustering completed and saved to 'clustered_data.csv'")

# Plot the clustered data
cluster_colors=['Red','Green','Blue']
plt.figure(figsize=(10, 6))

'''Scatter plot for the first two features,'Avg Credit Limit' vs 'Total Credit Cards'.
 Note: The K-means considered all the relevant features provided during clustering,
 even though we visually represent it using just two features.'''
for cluster_id in range(3):
    plt.scatter(data.loc[data['Cluster'] == cluster_id, 'Avg_Credit_Limit'],
                data.loc[data['Cluster'] == cluster_id, 'Total_Credit_Cards'],
                color=cluster_colors[cluster_id], label=f'Cluster {cluster_id + 1}', marker='o')

plt.title('K-means Clustering')
plt.xlabel('Avg Credit Limit')
plt.ylabel('Total Credit Cards')
plt.legend()
plt.show()
