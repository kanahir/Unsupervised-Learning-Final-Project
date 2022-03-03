from clustering import *
from main_clustering import get_hyper_params

params = get_hyper_params()
MAX_CLUSTERS = params["max_clusters"]
SAMPLE_FRAC = params["sample_fraction"]
N_RUNS = params["anova_runs"]


def calc_MI(data, exteranl_variables, my_dict, num_clusters=9):
    dbscan_labels = my_DBSCAN(data, num_clusters, epsilon=0.7)
    kmeans_labels = KMeans(n_clusters=num_clusters).fit(data).labels_
    ac_labels = AgglomerativeClustering(n_clusters=num_clusters).fit(data).labels_
    gmm_labels = GaussianMixture(n_components=num_clusters).fit(data.sample(frac=0.5)).predict(data)

    for external_name, exteranl_var in exteranl_variables.items():
        exteranl_var = exteranl_var.iloc[data.index]
        exteranl_var = exteranl_var.to_frame()
        lb_make = LabelEncoder()
        exteranl_labels = lb_make.fit_transform(exteranl_var[exteranl_var.columns[0]])
        kmeans_MI = mutual_info_score(kmeans_labels, exteranl_labels)
        ac_MI = mutual_info_score(ac_labels, exteranl_labels)
        gmm_MI = mutual_info_score(gmm_labels, exteranl_labels)
        if dbscan_labels is not None:
            dbscan_MI = mutual_info_score(dbscan_labels, exteranl_labels)
        else:
            dbscan_MI = None
        my_dict[external_name]['kmeans'].append(kmeans_MI)
        my_dict[external_name]['ac'].append(ac_MI)
        my_dict[external_name]['gmm'].append(gmm_MI)
        my_dict[external_name]['dbscan'].append(dbscan_MI)


def calc_MI_for_t_test(data, exteranl_variables):
    file = open("MI_after_mca", "rb")
    results = {}
    for exteranl_var_name in exteranl_variables.keys():
        results[exteranl_var_name] = {}
        results[exteranl_var_name]['kmeans']= []
        results[exteranl_var_name]['ac']= []
        results[exteranl_var_name]['gmm']= []
        results[exteranl_var_name]['dbscan']= []

    for i in range(N_RUNS):
        print(i)
        sampled_data = data.sample(frac=SAMPLE_FRAC)
        calc_MI(sampled_data, exteranl_variables, results)
    pickle.dump([results], file)
    for exteranl_var_name in exteranl_variables.keys():
        kmeans_MI = results[exteranl_var_name]['kmeans']
        ac_MI = results[exteranl_var_name]['ac']
        gmm_MI = results[exteranl_var_name]['gmm']
        dbscan_MI = results[exteranl_var_name]['dbscan']
        print(exteranl_var_name)
        t_test_to_MI(kmeans_MI, ac_MI, dbscan_MI, gmm_MI)
    return results

def t_test_to_MI(kmeans_MI, ac_MI, dbscan_MI, gmm_MI):
    dbscan_MI = [x for x in dbscan_MI if x is not None]

    anova_results = scipy.stats.f_oneway(kmeans_MI, ac_MI, dbscan_MI, gmm_MI)
    p = anova_results[1]
    print("p is: {}".format(p))
    # t test
    algo = ['K-Means']*len(kmeans_MI) +['Agglomerative']*len(ac_MI) + ['Dbscan']*len(dbscan_MI) + ['GMM']*len(gmm_MI)
    result = kmeans_MI + ac_MI + dbscan_MI + gmm_MI
    tukey = pairwise_tukeyhsd(endog=result, groups=algo, alpha=0.05)
    print(tukey)

def find_silhouette(my_dict, data):
    silhouette = {}
    for n_clustres in my_dict:
        epsilon, labels = my_dict[n_clustres]
        silhouette_avg = silhouette_score(data, labels)
        silhouette[n_clustres] = silhouette_avg
    return silhouette


def find_silhouette_for_t_test(original_data, initi_epsilon, num_clusters, runs=N_RUNS, frac=SAMPLE_FRAC):
    kmeans_silhouette = []
    ac_silhouette = []
    dbscan_silhouette = []
    gmm_silhouette = []
    predictions = {'kmeans':[], 'ac': [], 'gmm':[], 'dbscan':[]}
    for r in range(runs):
        data = original_data.sample(frac=frac)
        kmean_labels = KMeans(n_clusters=num_clusters).fit(data).labels_
        ac_labels = AgglomerativeClustering(n_clusters=num_clusters).fit(data).labels_
        gmm_labels = GaussianMixture(n_components=num_clusters).fit(data).predict(original_data)
        dbscan_labels = my_DBSCAN(data, num_clusters, initi_epsilon)

        predictions['kmeans'].append(kmean_labels)
        predictions['ac'].append(ac_labels)
        predictions['gmm'].append(gmm_labels)

        ac_silhouette.append(silhouette_score(data, ac_labels))
        kmeans_silhouette.append(silhouette_score(data, kmean_labels))
        gmm_silhouette.append(silhouette_score(original_data, gmm_labels))

        if dbscan_labels is not None:
            data = data.to_numpy()
            ind_of_not_anomaly = [i for i in range(len(dbscan_labels)) if dbscan_labels[i] != -1]
            dbscan_silhouette.append(silhouette_score(data[ind_of_not_anomaly, :], dbscan_labels[ind_of_not_anomaly]))
            predictions['dbscan'].append(dbscan_labels)

        file = open("results_silhouette", "wb")
        pickle.dump([predictions], file)
        file.close()
    return kmeans_silhouette, ac_silhouette, dbscan_silhouette, gmm_silhouette


