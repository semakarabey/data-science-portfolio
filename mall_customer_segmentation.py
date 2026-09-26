import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("30-mall_customers.csv")
print(df.head())
print(df.info())

df.drop("CustomerID", axis=1, inplace=True)

# encoding
label_encod = LabelEncoder()
df["Gender"] = label_encod.fit_transform(df["Gender"])
print(df.head())

# scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

df = pd.DataFrame(df, columns=df.columns)
print(df.head())

# dendrogram
plt.figure(1, figsize = (10,8))
dendrogram = sch.dendrogram(sch.linkage(df, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# we can clearly select 4 to 6 clusters from this dendrogram, let's go for 4
cluster = AgglomerativeClustering(n_clusters=4)
y_cluster = cluster.fit_predict(df)
print(y_cluster)

df["Cluster"] = pd.DataFrame(y_cluster)
print(df.head())

print("SCORE:", silhouette_score(df,y_cluster))

# decrease the amount of columns
X = df[["Annual Income (k$)","Spending Score (1-100)"]].copy()

cluster2 = AgglomerativeClustering(n_clusters=4)
y_cluster2 = cluster2.fit_predict(X)
X["cluster"] = y_cluster2
print(X)

sns.scatterplot(data=X, x = "Annual Income (k$)", y="Spending Score (1-100)", hue="cluster", palette = "Set2")
plt.title("Customer Clusters")
plt.show()

print(silhouette_score(X,y_cluster2))


# comparing the algorithms and results with hierarchical cluster
df = pd.read_csv("30-mall_customers.csv")
df = df.drop("CustomerID", axis = 1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

features_2d = ["Annual Income (k$)","Spending Score (1-100)"]
features_3d = ["Age","Annual Income (k$)","Spending Score (1-100)"]
features_4d = ["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]

for feats in [features_2d,features_3d,features_4d]:
    X = df[feats]
    X_scaled = MinMaxScaler().fit_transform(X)

    hc = AgglomerativeClustering(n_clusters=5)
    y_hc = hc.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_hc)
    db = davies_bouldin_score(X_scaled, y_hc)
    ch = calinski_harabasz_score(X_scaled, y_hc)

    print("-----------hierarchical cluster results------------")
    print(f"\n features :{feats}")
    print("Silhoutte score: ", sil)
    print("Davies Bouldin score: ", db)
    print("Calinski Harabasz score: ", ch)
    print("-------------")


# comparing the algorithms and results with KMeans cluster
df = pd.read_csv("30-mall_customers.csv")
df = df.drop("CustomerID", axis = 1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

features_2d = ["Annual Income (k$)","Spending Score (1-100)"]
features_3d = ["Age","Annual Income (k$)","Spending Score (1-100)"]
features_4d = ["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]

for feats in [features_2d,features_3d,features_4d]:
    X = df[feats]
    X_scaled = MinMaxScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=5)
    y_hc = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_hc)
    db = davies_bouldin_score(X_scaled, y_hc)
    ch = calinski_harabasz_score(X_scaled, y_hc)

    print("-----------KMeans cluster results------------")
    print(f"\n features :{feats}")
    print("Silhoutte score: ", sil)
    print("Davies Bouldin score: ", db)
    print("Calinski Harabasz score: ", ch)
    print("-------------")