from modules import *
from scipy.cluster.hierarchy import dendrogram


def plot_bar(kmeans, ac, dbscan, ticks):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(kmeans))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, kmeans, color='r', width=barWidth,
            edgecolor='grey', label='K-MEANS')
    plt.bar(br2, ac, color='g', width=barWidth,
            edgecolor='grey', label='Agglomerative')
    plt.bar(br3, dbscan, color='b', width=barWidth,
            edgecolor='grey', label='DBSCAN')

    # Adding Xticks
    plt.xlabel('N Clusters', fontweight='bold', fontsize=15)
    plt.ylabel('Average Silhouette', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(kmeans))],
               ticks)

    plt.legend()
    # plt.savefig("plots/n_clusters_by_silhouette")
    plt.show()


def plot_MI_bar(my_dict):
    ticks = my_dict.keys()
    fig, ax = plt.subplots()
    # set width of bar
    barWidth = 0.2

    for ind, (variable_name, value) in enumerate(my_dict.items()):
        kmeans_MI = value['kmeans']
        ac_MI = value['ac']
        dbscan_MI = value['dbscan']
        gmm_MI = value['gmm']

        dbscan_MI = [x for x in dbscan_MI if x != None]

        kmeans_mean = np.mean(kmeans_MI)
        ac_mean = np.mean(ac_MI)
        dbscan_mean = np.mean(dbscan_MI)
        gmm_mean = np.mean(gmm_MI)

        # Calculate the standard deviation
        kmeans_std = np.std(kmeans_MI)
        ac_std = np.std(ac_MI)
        dbscan_std = np.std(dbscan_MI)
        gmm_std = np.std(gmm_MI)


        # Build the plot

        # Set position of bar on X axis
        br1 = np.arange(1)
        br1[0] = ind
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        # Make the plot
        kmean = plt.bar(br1, kmeans_mean, width=barWidth, color='lightcoral', label='K-MEANS')
        agg = plt.bar(br2, ac_mean, width=barWidth, color='palegreen', label='Agglomerative')
        dbscan = plt.bar(br3, dbscan_mean, width=barWidth, color='powderblue',  label='DBSCAN')
        gmm = plt.bar(br4, gmm_mean, width=barWidth, color='slateblue', label='GMM')

        plt.errorbar(br1, kmeans_mean, yerr=kmeans_std, color='r',capsize=10,ecolor='k', elinewidth=3)
        plt.errorbar(br2, ac_mean, yerr=ac_std, color='g',capsize=10, ecolor='k',elinewidth=3)
        plt.errorbar(br3, dbscan_mean, yerr=dbscan_std, color='b',capsize=10,ecolor='k', elinewidth=3)
        plt.errorbar(br4, gmm_mean, yerr=gmm_std, color='b',capsize=10,ecolor='k', elinewidth=3)

    # Adding Xticks
    plt.ylabel('Mutual Information')
    plt.xticks([r + 1.5*barWidth for r in range(len(ticks))],
               ticks)

    # ax.set_ylabel('Mutual Information')
    ax.set_title('MI VS External Variable')
    ax.yaxis.grid(True)
    plt.tight_layout()
    # ax.legend()
    # plt.legend(["kmean", "AC", "DBSCAN"])
    ax.legend(handles=[kmean, agg, dbscan, gmm])
    # plt.savefig('plots/bar_plot.png')
    plt.show()

def plot_clustering(dataset, labels, title, i):
    plt.subplot(2, 4, i)
    plot = plt.scatter(dataset[:, 0], dataset[:, 1], c=labels)
    plt.title(title)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    return plot

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.title("Hierarchical Clustering Dendrogram")
    # plt.savefig("plots/dendogram")
    plt.show()

def plot_algo_vs_real(data, real_labels, pred_labels, method="T-SNE"):
    if method == "T-SNE":
        dataset_2_dim = TSNE(n_components=2, learning_rate='auto',
                             init='random', method='exact').fit_transform(data)
    # f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    plt.subplot(1, 2, 1)
    plt.scatter(dataset_2_dim[:, 0], dataset_2_dim[:, 1], c=pred_labels)
    plt.title("Predicted Clustering")

    plt.subplot(1, 2, 2)
    plt.scatter(dataset_2_dim[:, 0], dataset_2_dim[:, 1], c=real_labels, label=np.unique(real_labels))
    plt.title("Real Clustering")

    plt.suptitle("{}".format(method))
    plt.savefig("plots/visualization_dAge")
    plt.show()

def plot_mca_dim(eigen_vec):
    cumsum = np.cumsum(eigen_vec)
    cumsum_normalized = [x/sum(eigen_vec) for x in cumsum]
    eigen_vec_normalized = [x/sum(eigen_vec) for x in eigen_vec]
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigen_vec)), eigen_vec_normalized, width=0.7)
    plt.axvline(19, color='r', ls="--")
    plt.xlabel("N dimensions")
    plt.ylabel("Relative Variance")
    plt.title("Eigen Values")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.bar(range(len(cumsum)), cumsum_normalized, width=0.7)
    plt.axhline(0.9, color='r', ls="--")
    plt.title("Cumulative Sum")
    plt.xlabel("N dimensions")
    plt.ylabel("Relative Variance")
    plt.grid()

    plt.tight_layout()
    plt.suptitle("MCA")
    # plt.savefig("plots/eigen")
    plt.show()





