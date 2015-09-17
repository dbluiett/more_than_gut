import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

#importing the raw data
data = pd.read_csv("/Users/datascientist/dbluiett/bank-additional/bank-additional-full.csv", sep =";")
#Transforming data to categorical and continous features
data["default"] = data.default == "yes"
data["housing"] = data.housing == "yes"
data["loan"] = data.loan == "yes"
data["cell_phone"]= data.contact == "cellular"
data.drop("contact", axis=1, inplace=True)

#subsetting the customer specific data
cust_data = data[["age","cell_phone","default","education","housing","job","loan","marital","poutcome"]]

#Transforming data to categorical and continous feature

cust_data_transform = pd.get_dummies(cust_data)

#Creating Customer Clusters
def make_cluster(df):
    cluster_df = pd.DataFrame()
    clusters = KMeans(n_clusters=4)
    distance_matrix = clusters.fit_transform(cust_data_transform)
    cluster_df["cluster"]=clusters.labels_
    #Finding the euclidean distance from the point to its cluster center
    cluster_df["dist"]=[min(x) for x in distance_matrix]
    return cluster_df, clusters.cluster_centers_

cust_cluster, cluster_centers = make_cluster(cust_data_transform)

#Creating plots
sns.violinplot(x="cluster", y="dist", data=cust_cluster, order = np.arange(4))
cluster_summary = pd.DataFrame(data=cluster_centers, columns=cust_data_transform.columns)
#cluster_summary[1:]=cluster_summary[1:].apply(lambda x: x>=0.5, axis=0)
max2 = [np.argmax(cluster_summary[x]) for x in cluster_summary.columns]
cluster_summary = cluster_summary.T
cluster_summary["max"] =max2


#Indexing out the positive outcomes
cust_data["target"]= data.y.copy()
target_customers = cust_data[cust_data.target =="yes"]
target_customers.drop("target", axis=1, inplace=True)
target_cust_transform = pd.get_dummies(target_customers)

target_cust_cluster, target_cust_cluster_center = make_cluster(target_cust_transform)
target_cluster_summary = pd.DataFrame(data=target_cust_cluster_center, columns=target_cust_transform.columns)
max1= [np.argmax(target_cluster_summary[x]) for x in target_cluster_summary.columns]
target_cluster_summary= target_cluster_summary.T
target_cluster_summary["max"]=max1








