from reduce_dimentions import reduce_dimension_and_plot
from visualization import *
from metrics import *

FILE_NAME = "USCensus1990.data.csv"
MAX_CLUSTERS = 20
SAMPLE_FRAC = 0.01
N_RUNS = 30


def get_hyper_params():
    return {"max_clusters": MAX_CLUSTERS, "sample_fraction": SAMPLE_FRAC, "anova_runs": N_RUNS}


def find_elbow_kmeans(data):
    data = data.sample(frac=SAMPLE_FRAC)
    score = []
    for k in range(2, MAX_CLUSTERS):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        score.append(kmeans.inertia_)

    plt.figure()
    plt.plot(list(range(2, MAX_CLUSTERS)), score)
    plt.xticks(list(range(2, MAX_CLUSTERS)))
    plt.title("K-MEANS SSE VS N Clusters")
    plt.xlabel("N Clusters")
    plt.ylabel("SSE")
    plt.grid()
    plt.axvline(x=8, ymin=0, ymax=0.27, color="r", ls="--", lw=2)
    plt.savefig("plots/kmeans_elbow")
    plt.show()
    return score


def find_n_clusters_by_silhouette(original_data):
    kmeans_silhouette = []
    ac_silhouette = []
    dbscan_silhouette = []
    data = original_data.sample(frac=SAMPLE_FRAC)
    data = minmax_scale(data)
    data = np.nan_to_num(data)
    for n in range(2, MAX_CLUSTERS):
        kmean_labels = KMeans(n_clusters=n).fit(data).labels_
        ac_labels = AgglomerativeClustering(n_clusters=n).fit(data).labels_
        dbsacen_labels = my_DBSCAN(data, n, epsilon=1.65)
        kmeans_silhouette.append(silhouette_score(data, kmean_labels))
        ac_silhouette.append(silhouette_score(data, ac_labels))
        if dbsacen_labels is not None:
            dbscan_silhouette.append(silhouette_score(data, dbsacen_labels))
        else:
            dbscan_silhouette.append(None)
    plot_bar(kmeans_silhouette, ac_silhouette, dbscan_silhouette, list(range(2, MAX_CLUSTERS)))
    return


def find_best_method(data, n_clusters=5):
    kmeans_silhouette, ac_silhouette, dbscan_silhouette, gmm_silhouette = find_silhouette_for_t_test(data, 0.19, num_clusters=n_clusters)

    anova_results = scipy.stats.f_oneway(kmeans_silhouette, ac_silhouette, dbscan_silhouette, gmm_silhouette)
    p = anova_results[1]
    print("p is: {}".format(p))
    # t test
    algo = ['K-Means']*len(kmeans_silhouette) +['Agglomerative']*len(ac_silhouette) + ['Dbscan']*len(dbscan_silhouette) + ['GMM']*len(gmm_silhouette)
    result = kmeans_silhouette + ac_silhouette + dbscan_silhouette + gmm_silhouette
    tukey = pairwise_tukeyhsd(endog=result, groups=algo, alpha=0.05)
    print(tukey)
    # plot:
    d = {'algorithm': algo, 'silhouette': result}
    df = pd.DataFrame(data=d)
    df = df.groupby("algorithm").agg([np.mean, np.std])
    silhouette = df['silhouette']
    silhouette.plot(kind = "barh", y = "mean", legend = False,
                    title = "Average Silhouette", xerr = "std")
    plt.tight_layout()
    plt.savefig("plots/silhouette_with_gmm")

    plt.show()


def pre_processing(data):
    # to onehot and normalization
    age = data['dAge']
    hispanic = data['dHispanic']
    yhearwrk = data['iYearwrk']
    sex = data['iSex']

    data = data.drop(columns=['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'])
    sampled_data = data.sample(frac=SAMPLE_FRAC*3)
    mca = prince.CA(n_components=40,
                    n_iter=3,
                    copy=True,
                    check_input=True,
                    engine='auto',
                    random_state=42)
    mca = mca.fit(sampled_data)
    plot_mca_dim(mca.eigenvalues_)

    mca = prince.CA(n_components=19,
                    n_iter=3,
                    copy=True,
                    check_input=True,
                    engine='auto',
                    random_state=42)
    mca = mca.fit(sampled_data)
    data_after_mca = mca.transform(data)
    return data, data_after_mca, {'dAge': age, 'dHispanic':hispanic, 'iYearwrk': yhearwrk, 'iSex':sex}

if __name__ == '__main__':

    data = pd.read_csv(FILE_NAME)

    original_data, data, external_variables = pre_processing(data)

    # find n clusters:
    find_elbow_kmeans(data)

    # find best method to cluster the data by its silhouette
    find_best_method(data, n_clusters=9)


    # calc mutual information
    calc_MI_for_t_test(data, external_variables)

    # reduce dimensions and visualization
    sampled_data = data.sample(frac=SAMPLE_FRAC)
    my_labels = KMeans(n_clusters=8).fit(sampled_data).labels_
    reduce_dimension_and_plot(original_data.sample(frac=SAMPLE_FRAC), sampled_data, my_labels)

    # plot T-SNE
    yhearwrk = external_variables['iYearwrk']
    plot_algo_vs_real(sampled_data, yhearwrk.array[sampled_data.index], my_labels)
