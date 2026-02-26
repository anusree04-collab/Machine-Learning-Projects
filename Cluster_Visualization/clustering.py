import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

df = pd.read_csv(r"C:\Users\K anusree\Downloads\Mall_Customers.csv")

df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Spending'
}, inplace=True)

X = df[['Age', 'Income', 'Spending']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)

def run_clustering(algo):
    if algo == "kmeans":
        labels = KMeans(n_clusters=5, random_state=42).fit_predict(x_pca)
    elif algo == "hierarchical":
        labels = AgglomerativeClustering(n_clusters=5).fit_predict(x_pca)
    elif algo == "dbscan":
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(x_pca)

    return x_pca, labels


