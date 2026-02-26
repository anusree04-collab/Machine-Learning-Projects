import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\K anusree\Downloads\Mall_Customers.csv")

x = df[['Age','Annual Income (k$)','Spending Score (1-100)']]

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_scaled = scaler.fit_transform(x)

#PCA
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)

#K-Means
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=5,random_state=42)
kmeans_labels=kmeans.fit_predict(x_pca)

#Hierarchical
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5)
hc_labels=hc.fit_predict(x_pca)

#DB SCAN
from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=0.5,min_samples=5)
dbscan_labels=dbscan.fit_predict(x_pca)

plt.scatter(x_pca[:,0],x_pca[:,1],c=kmeans_labels)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title("K-Means CLustering")
plt.show()

