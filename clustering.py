from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
import pandas as pd
import numpy as np


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)


df = pd.read_csv('Stars.csv')


numeric_features = ['Temperature', 'L', 'R', 'A_M']
no_outlier_df = pd.DataFrame(columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class', 'Type'])
empty_df = pd.DataFrame(columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class', 'Type'])
# find noisy data
for i in numeric_features:
    df_sort = df.sort_values(by=i, ascending=True)
    q1 = np.quantile(df[i], 0.25)
    q3 = np.quantile(df[i], 0.75)
    IQR = q3 - q1
    lower_fence = q1 - (1.5 * IQR)
    upper_fence = q3 + (1.5 * IQR)

    no_outlier_df = empty_df.copy()
    for row in range(len(df[i])):
        if not(df.loc[row, i] < lower_fence or df.loc[row, i] > upper_fence):
            no_outlier_df = no_outlier_df.append(df.loc[row, ['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class',
                                                              'Type']], ignore_index=True)
    df = no_outlier_df.copy()

convert_numeric = df.apply(LabelEncoder().fit_transform)
#correlation = convert_numeric.corr(method='pearson', min_periods=1)
data = pd.DataFrame(convert_numeric, columns=['Temperature', 'L', 'R', 'Color', 'Spectral_Class'])


# without preprocessing
kmeans = KMeans(n_clusters=6).fit(convert_numeric)
k_true = 0
for label in range(len(kmeans.labels_)):
    if kmeans.labels_[label] == convert_numeric.loc[label, 'Type']:
        k_true += 1
k_accuracy = k_true/len(kmeans.labels_) * 100
print('Accuracy of partitional algorithm without preprocessing: {:.2f}'.format(k_accuracy))

agglomerative = AgglomerativeClustering().fit(convert_numeric)
a_true = 0
for label in range(len(agglomerative.labels_)):
    if agglomerative.labels_[label] == convert_numeric.loc[label, 'Type']:
        a_true += 1
a_accuracy = a_true/len(agglomerative.labels_) * 100
print('Accuracy of hierarchical algorithm without preprocessing: {:.2f}'.format(a_accuracy))


# with preprocessing
norm_df = MinMaxScaler().fit_transform(convert_numeric)
discretize_df = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
discretize_df.fit(norm_df)
preprocessed_data = discretize_df.transform(norm_df)

kmeans_pre = KMeans(n_clusters=6).fit(preprocessed_data)
k_true_pre = 0
for label in range(len(kmeans_pre.labels_)):
    if kmeans_pre.labels_[label] == convert_numeric.loc[label, 'Type']:
        k_true_pre += 1
k_accuracy_pre = k_true_pre/len(kmeans_pre.labels_) * 100
print('Accuracy of partitional algorithm with preprocessing: {:.2f}'.format(k_accuracy_pre))

agglomerative_pre = AgglomerativeClustering().fit(preprocessed_data)
a_true_pre = 0
for label in range(len(agglomerative_pre.labels_)):
    if agglomerative_pre.labels_[label] == convert_numeric.loc[label, 'Type']:
        a_true_pre += 1
a_accuracy_pre = a_true_pre/len(agglomerative_pre.labels_) * 100
print('Accuracy of hierarchical algorithm with preprocessing: {:.2f}'.format(a_accuracy_pre))
