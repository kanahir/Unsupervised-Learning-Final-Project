
from visualization import *
from modules import *


def reduce_dimension_and_plot(original_data, X, labels):
    X = minmax_scale(X)
    X = np.nan_to_num(X)

    #PCA:
    dataset_after_pca = PCA(n_components=2).fit_transform(X)
    plot_clustering(dataset_after_pca, labels, "PCA", 1)

    #mds
    dataset_after_mds = MDS(n_components=2).fit_transform(X)
    plot_clustering(dataset_after_mds, labels, "MDS", 2)
    #LLE
    dataset_after_LLE = LocallyLinearEmbedding(n_neighbors=30, n_components=2).fit_transform(X)
    plot_clustering(dataset_after_LLE, labels, "LLE", 3)

    #isomap
    dataset_after_ISOMAP = Isomap(n_components=2).fit_transform(X)
    plot_clustering(dataset_after_ISOMAP, labels, "ISOMAP", 4)

    #Laplasian
    dataset_after_laplacian = SpectralEmbedding(n_components=2, n_neighbors=50).fit_transform(X)
    plot_clustering(dataset_after_laplacian, labels, "Laplacian", 5)

    # T - SNE
    dataset_after_tsne = TSNE(n_components=2, learning_rate='auto',
                              init='random', method='exact').fit_transform(X)
    plot_clustering(dataset_after_tsne, labels, "T-SNE", 6)
    # ICA
    dataset_after_ICA = FastICA(n_components=2, random_state=0).fit_transform(X)
    plot_clustering(dataset_after_ICA, labels, "ICA", 7)

    # MCA
    mca = prince.CA(n_components=2,
                    n_iter=3,
                    copy=True,
                    check_input=True,
                    engine='auto',
                    random_state=42)

    mca = mca.fit(original_data.sample(frac=0.5))
    dataset_after_MCA = mca.transform(original_data)
    dataset_after_MCA = dataset_after_MCA.to_numpy()
    plot_clustering(dataset_after_MCA, labels, "MCA", 8)

    plt.suptitle("Visualization")
    plt.savefig("plots/visualization")
    plt.show()

if __name__ == '__main__':
    print("main")

