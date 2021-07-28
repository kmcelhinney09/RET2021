import pickle
import umap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

celebA_data_path = 'CelebA_prediction_numpy'
celebA_labels_path = 'CelebA_predictions_numpy_labels'

with open(celebA_data_path, 'rb') as data:
    celebA_data = pickle.load(data)

with open(celebA_labels_path, 'rb') as labels:
    celebA_labels = pickle.load(labels)


# Initialize umap and tsne
umap_reducer = umap.UMAP()
tsne_reducer = TSNE()

umap_embedding = umap_reducer.fit_transform(celebA_data)

plt.scatter(
    umap_embedding[:, 0],
    umap_embedding[:, 1],
    c=celebA_labels)

plt.title('UMAP for 500 samples of CelebA predictions', fontsize=24)
plt.show()

tsne_embedding = tsne_reducer.fit_transform(celebA_data)

plt.scatter(
    tsne_embedding[:, 0],
    tsne_embedding[:, 1],
    c=celebA_labels)


plt.title('TSNE for 500 samples of CelebA predictions', fontsize=24)
plt.show()
