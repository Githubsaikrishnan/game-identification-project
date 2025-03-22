import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load embeddings for all games
games = ["minecraft", "fortnite", "fifa22", "fifa23"]
all_embeddings = []
labels = []
for idx, game in enumerate(games):
    embeddings = np.load(f"{game}_embeddings.npy")
    all_embeddings.extend(embeddings)
    labels.extend([idx] * len(embeddings))  # Assign label for each game

all_embeddings = np.array(all_embeddings)
labels = np.array(labels)

# Cluster with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusters = clusterer.fit_predict(all_embeddings)

# Visualize clusters with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=clusters,
    cmap="viridis",
    alpha=0.6
)
plt.title("Game Clusters (t-SNE Visualization)")
plt.colorbar(label="Cluster ID")
plt.show()