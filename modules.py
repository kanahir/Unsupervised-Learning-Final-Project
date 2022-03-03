from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score, mutual_info_score
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import math
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import pickle
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import prince
