from ml import datasets
from ml.cluster import KMeans
from ml import viz

data = datasets.load_oldfaithful()

X = datasets.load_oldfaithful()
kmeans = KMeans(n_components=2, figs_dir='out')
kmeans.fit(X)
preds = kmeans.predict(X)

viz.plot_kmeans(X, kmeans.centroids, preds, 'kmeans_final.png')

