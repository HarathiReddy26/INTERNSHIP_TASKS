import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Step 1: Load data
train_df = pd.read_csv('/kaggle/input/input-csv/Mall_Customers.csv')

# Step 2: Check for missing values
print("Missing values:\n", train_df.isnull().sum())

# Step 3: Drop 'CustomerID'
data = train_df.drop(['CustomerID'], axis=1)

# Step 4: Encode Gender
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Step 5: Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 6: Elbow Method to find optimal K
sse = []
with threadpool_limits(limits=1, user_api='blas'):
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

# Step 7: Plot Elbow curve
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 8: Choose K (for example, K=5 from elbow method)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 9: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='Set2',
    s=100
)
plt.title('Customer Segments based on Income and Spending Score')
plt.show()
