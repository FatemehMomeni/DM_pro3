from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, normalize, KBinsDiscretizer, MinMaxScaler
import pandas as pd


df = pd.read_csv('Stars.csv')
data = pd.DataFrame(df, columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class'])
convert_numeric = data.apply(LabelEncoder().fit_transform)


# without preprocessing
kmeans = KMeans(n_clusters=6).fit(convert_numeric)
k_true = 0
for label in range(len(kmeans.labels_)):
    if kmeans.labels_[label] == df.loc[label, 'Type']:
        k_true += 1
k_accuracy = k_true/len(kmeans.labels_) * 100
print('Accuracy of partitional algorithm without preprocessing: {:.2f}'.format(k_accuracy))

agglomerative = AgglomerativeClustering().fit(convert_numeric)
a_true = 0
for label in range(len(agglomerative.labels_)):
    if agglomerative.labels_[label] == df.loc[label, 'Type']:
        a_true += 1
a_accuracy = a_true/len(agglomerative.labels_) * 100
print('Accuracy of hierarchical algorithm without preprocessing: {:.2f}'.format(a_accuracy))


# with preprocessing
#norm_df = normalize(convert_numeric, axis=0)
norm_df = MinMaxScaler().fit_transform(convert_numeric)
discretize_df = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
discretize_df.fit(norm_df)
preprocessed_data = discretize_df.transform(norm_df)

kmeans_pre = KMeans(n_clusters=6).fit(preprocessed_data)
k_true_pre = 0
for label in range(len(kmeans_pre.labels_)):
    if kmeans_pre.labels_[label] == df.loc[label, 'Type']:
        k_true_pre += 1
k_accuracy_pre = k_true_pre/len(kmeans_pre.labels_) * 100
print('Accuracy of partitional algorithm with preprocessing: {:.2f}'.format(k_accuracy_pre))

agglomerative_pre = AgglomerativeClustering().fit(preprocessed_data)
a_true_pre = 0
for label in range(len(agglomerative_pre.labels_)):
    if agglomerative_pre.labels_[label] == df.loc[label, 'Type']:
        a_true_pre += 1
a_accuracy_pre = a_true_pre/len(agglomerative_pre.labels_) * 100
print('Accuracy of hierarchical algorithm with preprocessing: {:.2f}'.format(a_accuracy_pre))

