import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.svm import OneClassSVM
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

print("read file")
FILE_NAME = "data.csv"
original_data = pd.read_csv(FILE_NAME)
data = original_data.drop(columns=['dAge', 'dHispanic', 'iYearwrk', 'iSex'])

# print("dAge")
dAge = original_data['dAge']
# print("dHispanic")
dHispanic = original_data['dHispanic']
# print("iYearwrk")
iYearwrk = original_data['iYearwrk']
# print("iSex")
iSex = original_data['iSex']

partial_data = data.sample(frac=0.02).reset_index(drop=True)
partial_data = minmax_scale(partial_data)  # comes back as numpy
partial_data = np.nan_to_num(partial_data)

data = minmax_scale(data)  # comes back as numpy
data = np.nan_to_num(data)


# finding anomaly #######################


# ONE CLASS SVM ############################################
def oneclasssvm_outlier(partial_data, data):
    print("starting OneClassSVM().fit")
    oneclassSVM = OneClassSVM().fit(partial_data)

    print("starting oneclassSVM.predict")
    #    prediction = oneclassSVM.predict(data)  # predict 1 part of data. -1 outlier
    prediction = oneclassSVM.predict(data)
    score_sample = oneclassSVM.score_samples(data)

    outlier_index = np.where(prediction == -1)  # returns tuple
    outlier_dict = {}  # key is the index in the data. value is the array of features of the index
    prediction_dict = {}
    print("starting outlier_dict")
    # outlier dict
    for index in outlier_index[0]:
        outlier_dict[index] = data[index]

    for index, predict in enumerate(prediction):
        prediction_dict[index] = [predict]

    return outlier_dict, prediction_dict, score_sample


def one_class_svm_percentage(outlier_dict, data, partial_data):
    outlier_size = len(outlier_dict)
    percentage = outlier_percentage(outlier_size, data)
    print("one class svm percentage data")
    print(percentage)
    print("fit part of data (frac=0.02). predict on whole data")


def one_class_svm_algo_outlier():

    outlier_svm_dict, svm_predict_dict, score_sample_svm = oneclasssvm_outlier(partial_data, data)
    print("dict to data frame")
    svm_predict_df = pd.DataFrame(svm_predict_dict)
    outlier_svm_df = pd.DataFrame(outlier_svm_dict)
    score_sample_svm_df = pd.DataFrame(score_sample_svm)

    one_class_svm_percentage(outlier_svm_dict, data, partial_data)
    print("storing to excel")

    # storing prediction to excel
    svm_predict_df.to_csv('svm_predict_df.csv')

    # storing outliers to excel
    outlier_svm_df.to_csv('outlier_svm_df.csv')

    # storing score sample to excel
    score_sample_svm_df.to_csv('whole_score_sample_svm_df.csv')

    sampled_score_svm_df = score_sample_svm_df.sample(frac=0.001).reset_index(drop=True)
    sampled_score_svm_df.to_csv('sampled_score_svm')

    svm_mutual_information(dAge, dHispanic, iYearwrk, iSex, svm_predict_df)


def svm_mutual_information(dAge, dHispanic, iYearwrk, iSex, svm_predict_df):
    svm_outlier_label = svm_predict_df.values
    svm_mi_dAge = mutual_info_score(svm_outlier_label[0], dAge)
    print("svm_mi_dAge = ", svm_mi_dAge)

    svm_mi_dHispanic = mutual_info_score(svm_outlier_label[0], dHispanic)
    print("svm_mi_dHispanic = ", svm_mi_dHispanic)

    svm_mi_iYearwrk = mutual_info_score(svm_outlier_label[0], iYearwrk)
    print("svm_mi_iYearwrk = ", svm_mi_iYearwrk)

    svm_mi_iSex = mutual_info_score(svm_outlier_label[0], iSex)
    print("svm_mi_iSex = ", svm_mi_iSex)


# CLUSTER - KMEANS ########################################

def kmeans_distances(partial_data, data, labels, centers):
    print("kmeans distance dict")
    distances_dict = {}  # dictionary to know the distance for the data

    for cluster, center in enumerate(centers):  # go over each centroid
        for index, label in enumerate(labels):  # index and label of each instance
            if label == cluster:  # the label belongs to the cluster
                point = data[index]
                point = point.reshape(1, -1)  # reshape to have 2 arguments of size
                center = center.reshape(1, -1)
                distance = distance_matrix(center, point, p=len(data[index]))
                distances_dict[index] = [distance[0][0]]
    #                distances_dict[index] = (cluster, distance[0][0])
    return distances_dict


def kmeans_outliers_threshold(partial_data, data, labels, centers, kmeans_distances_dict):
    threshold = [t for t in np.arange(0.1, 1.05, 0.05)]
    outlier_percent = []

    for t in tqdm(threshold):
        print("t", t)
        kmeans_outliers = dict((k, v) for k, v in kmeans_distances_dict.items() if v[0] > t)
        #        kmeans_outliers = dict((k, v) for k, v in kmeans_distances_dict.items() if v[1] > t)
        outlier_size = len(kmeans_outliers)
        print("kmeans_outlier size", outlier_size)
        prc = outlier_percentage(outlier_size, data)
        outlier_percent.append(prc)
        print("kmeans_percentage", prc)
        kmeans_outliers_df = pd.DataFrame(kmeans_outliers)
        kmeans_outliers_df.to_csv('kmeans_outliers_df_threshold 'f'{t}''.csv')

    return threshold, outlier_percent


def kmeans_algo_outlier():
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)  # ask if right or should be partial data

    labels = kmeans.labels_
    # storing labels to excel
    kmeans_label_df = pd.DataFrame(labels)
    kmeans_label_df.to_csv('kmeans_label_df.csv')
    print("kmeans_label_df")

    centers = kmeans.cluster_centers_
    # storing centers to excel
    kmeans_centers_df = pd.DataFrame(centers)
    kmeans_centers_df.to_csv('kmeans_centers_df.csv')
    print("kmeans_centers_df")

    kmeans_distances_dict = kmeans_distances(partial_data, data, labels, centers)

    print("sampling")
    kmeans_sample_distances = random.sample(kmeans_distances_dict.items(), 2400)
    kmeans_sample_distances_df = pd.DataFrame(kmeans_sample_distances)
    kmeans_sample_distances_df.to_csv('kmeans_sample_distances_df.csv')

    # storing distance from centers to excel
    kmeans_distances_df = pd.DataFrame(kmeans_distances_dict)
    kmeans_distances_df.to_csv('kmeans_distances_from centers_df.csv')

    kmeans_threshold, kmeans_outlier_percent = kmeans_outliers_threshold(partial_data, data, labels, centers,
                                                                         kmeans_distances_dict)
    kmeans_outlier_percent_df = pd.DataFrame(kmeans_outlier_percent)
    kmeans_outlier_percent_df.to_csv('kmeans_outlier_percent_df.csv')

    kmeans_mutual_information(dAge, dHispanic, iYearwrk, iSex)


def kmeans_mutual_information(dAge, dHispanic, iYearwrk, iSex):
    print("read kmeans")
    kmeans_outliers_df_threshold_0_7 = pd.read_csv('kmeans_outliers_df_threshold 0.7.csv', index_col=0)
    print("done")

    kmeans_outlier_label = []

    kmeans_outlier_columns = kmeans_outliers_df_threshold_0_7.columns.tolist()

    for i, index in enumerate(kmeans_outlier_columns):
        kmeans_outlier_columns[i] = int(index)
    print("done kmeans columns")

    for index in range(data.shape[0]):
        if index in kmeans_outlier_columns:  # index is an outlier
            print("outlier index = ", index)
            kmeans_outlier_label.append(-1)
        else:
            kmeans_outlier_label.append(1)
    print("done kmeans outlier label")

    kmeans_outlier_label_df = pd.DataFrame(data=kmeans_outlier_label)
    kmeans_outlier_label_df.to_csv('kmeans_outlier_label_df.csv')

    # kmeans mi
    kmeans_mi_dAge = mutual_info_score(kmeans_outlier_label, dAge)
    print("kmeans_mi_dAge = ", kmeans_mi_dAge)

    kmeans_mi_dHispanic = mutual_info_score(kmeans_outlier_label, dHispanic)
    print("kmeans_mi_dHispanic = ", kmeans_mi_dHispanic)

    kmeans_mi_iYearwrk = mutual_info_score(kmeans_outlier_label, iYearwrk)
    print("kmeans_mi_iYearwrk = ", kmeans_mi_iYearwrk)

    kmeans_mi_iSex = mutual_info_score(kmeans_outlier_label, iSex)
    print("kmeans_mi_iSex = ", kmeans_mi_iSex)
# <------------------------------------------------------------------------------------------------------> #


def outlier_percentage(outlier_size, data):
    percentage = outlier_size / data.shape[0]

    return percentage


# GMM ############################################################

def gmm_outlier(partial_data, data):
    score_sample_dict = {}
    print("gmm fit")
    gmm = GaussianMixture(n_components=5).fit(partial_data)
    print("gmm predict")
    prediction = gmm.predict(data)
    print("score samples")
    score_sample = gmm.score_samples(data)

    print("gmm_score dict")
    for index, score in enumerate(score_sample):
        score_sample_dict[index] = [score]

    return prediction, score_sample_dict


def gmm_outlier_threshold(partial_data, data, gmm_prediction, gmm_score_sample_dict):
    threshold = [t for t in range(-10, 110, 10)]
    outlier_percent = []

    for t in tqdm(threshold):
        print("t", t)

        gmm_outliers = dict((k, v) for k, v in gmm_score_sample_dict.items() if v[0] < t)
        outlier_size = len(gmm_outliers)
        print("outlier size", outlier_size)
        prc = outlier_percentage(outlier_size, data)
        outlier_percent.append(prc)
        print("percentage", prc)
        gmm_outliers_df = pd.DataFrame(gmm_outliers)
        gmm_outliers_df.to_csv('gmm_outliers_df_threshold 'f'{t}''.csv')

    return threshold, outlier_percent


def gmm_algo_outlier():
    gmm_prediction, gmm_score_sample_dict = gmm_outlier(partial_data, data)
    print("storing")
    # storing prediction to excel
    gmm_prediction_df = pd.DataFrame(gmm_prediction)
    gmm_prediction_df.to_csv('gmm_prediction_df.csv')

    # storing sample score to excel
    gmm_score_sample_df = pd.DataFrame(gmm_score_sample_dict)
    gmm_score_sample_df.to_csv('gmm_score_sample_df.csv')

    gmm_threshold, gmm_outlier_percent = gmm_outlier_threshold(partial_data, data, gmm_prediction,
                                                               gmm_score_sample_dict)

    # storing outliers percentage to excel
    gmm_outlier_percent_df = pd.DataFrame(gmm_outlier_percent)
    gmm_outlier_percent_df.to_csv('gmm_outlier_percent_df.csv')

    gmm_score_sample_values = [v[0] for k, v in gmm_score_sample_dict.items()]
    gmm_score_sample_values_random = random.sample(gmm_score_sample_values, 2400)
    gmm_score_sample_values_random_df = pd.DataFrame(gmm_score_sample_values_random)
    gmm_score_sample_values_random_df.to_csv('sampled_gmm_score_sample.csv')

    gmm_mutual_information(dAge, dHispanic, iYearwrk, iSex)


def gmm_mutual_information(dAge, dHispanic, iYearwrk, iSex):
    gmm_outliers_df_threshold_0 = pd.read_csv('gmm_outliers_df_threshold 0.csv', index_col=0)

    gmm_outlier_label = []
    gmm_outlier_columns = gmm_outliers_df_threshold_0.columns.tolist()

    for i, index in enumerate(gmm_outlier_columns):
        gmm_outlier_columns[i] = int(index)
    print("done gmm columns")

    for index in range(data.shape[0]):
        if index in gmm_outlier_columns:  # index is an outlier
            print("outlier index = ", index)
            gmm_outlier_label.append(-1)
        else:
            gmm_outlier_label.append(1)
    print("done gmm outlier label")

    gmm_outlier_label_df = pd.DataFrame(data=gmm_outlier_label)
    gmm_outlier_label_df.to_csv('gmm_outlier_label_df.csv')

    # gmm mi
    gmm_mi_dAge = mutual_info_score(gmm_outlier_label, dAge)
    print("gmm_mi_dAge = ", gmm_mi_dAge)

    gmm_mi_dHispanic = mutual_info_score(gmm_outlier_label, dHispanic)
    print("gmm_mi_dHispanic = ", gmm_mi_dHispanic)

    gmm_mi_iYearwrk = mutual_info_score(gmm_outlier_label, iYearwrk)
    print("gmm_mi_iYearwrk = ", gmm_mi_iYearwrk)

    gmm_mi_iSex = mutual_info_score(gmm_outlier_label, iSex)
    print("gmm_mi_iSex = ", gmm_mi_iSex)

# <------------------------------------------------------------------------------------------------------------------> #



def fig_four_subplot(gmm_x1, gmm_y1, gmm_x2, gmm_y2, kmeans_x1, kmeans_y1, kmeans_x2, kmeans_y2):
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.xticks(np.arange(min(gmm_x1), max(gmm_x1), 10))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('threshold', fontsize=18)
    plt.ylabel('percent', fontsize=18)
    plt.title('GMM outlier percentage', fontsize=20)
    plt.plot(gmm_x1, gmm_y1)
    plt.plot([0, 0], [0, gmm_y1[1]], linestyle='-', color='r', linewidth=4.0)
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('instance', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.title('score in GMM', fontsize=20)
    plt.bar(gmm_x2, gmm_y2, width=1)
    plt.plot([0, 2400], [0, 0], linestyle='--', color='r', linewidth=4.0)
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.xticks(np.arange(min(kmeans_x1), max(kmeans_x1), 0.1))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('threshold', fontsize=18)
    plt.ylabel('percent', fontsize=18)
    plt.title('kmeans outlier percentage', fontsize=20)
    plt.plot(kmeans_x1, kmeans_y1)
    plt.plot([0.7, 0.7], [0, kmeans_y1[12]], linestyle='-', color='r', linewidth=4.0)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('instance', fontsize=18)
    plt.ylabel('distance', fontsize=18)
    plt.title('distance from center in kmeans', fontsize=20)
    plt.bar(kmeans_x2, kmeans_y2, width=1)
    plt.plot([0, 2400], [0.7, 0.7], linestyle='--', color='r', linewidth=4.0)
    plt.grid()

    plt.savefig('subplot kmeans & gmm')


def fix_distance_dataframe():
    kmeans_sample_distances_df = pd.read_csv('kmeans_sample_distances_df.csv', index_col=0)
    v = []
    instance_index = []
    for inx, row in kmeans_sample_distances_df.iterrows():
        p = row[1]
        instance_index.append(row[0])
        #        p = p.split(", ")
        p = p.replace("[", '')
        p = p.replace("]", '')
        p = float(p)
        v.append(p)
    #    new_r = {instance_index: p}
    #    df = df.append({instance_index: p}, ignore_index=True)
    #    df[instance_index] = p
    df = pd.DataFrame(data=v, index=instance_index)
    df.to_csv('new_kmeans_sample_distances_df.csv')


print("run kmeans algo outlier")
kmeans_algo_outlier()
print("done kmeans")
print("run one class svm algo outlier()")
one_class_svm_algo_outlier()
print("done one class svm")
print("run gmm algo outlier")
gmm_algo_outlier()
print("done gmm")

# plotting ###################################################################################
print("starting to read csv")

sampled_gmm_score_sample_df = pd.read_csv('sampled_gmm_score_sample.csv', index_col=0)
gmm_outlier_percent_df = pd.read_csv('gmm_outlier_percent_df.csv', index_col=0)
kmeans_sample_distances_df = pd.read_csv('kmeans_sample_distances_df.csv', index_col=0)
kmeans_outlier_percent_df = pd.read_csv('kmeans_outlier_percent_df.csv', index_col=0)
print("done reading csv")

print("creating axes")

gmm_x1 = [t for t in range(-10, 110, 10)]
gmm_y1 = [v[0] for v in gmm_outlier_percent_df.values]
gmm_y2 = [v[0] for v in sampled_gmm_score_sample_df.values]

print("sorting gmm y2")
gmm_y2 = sorted(gmm_y2)
gmm_x2 = [i for i in range(len(gmm_y2))]

kmeans_x1 = [t for t in np.arange(0.1, 1.05, 0.05)]
kmeans_y1 = [v[0] for v in kmeans_outlier_percent_df.values]
kmeans_y2 = [v[0] for v in kmeans_sample_distances_df.values]

print("sorting kmeans y2")
kmeans_y2 = sorted(kmeans_y2)
kmeans_x2 = [i for i in range(len(kmeans_y2))]


print("figure")
fig_four_subplot(gmm_x1, gmm_y1, gmm_x2, gmm_y2, kmeans_x1, kmeans_y1, kmeans_x2, kmeans_y2)
print("done")
