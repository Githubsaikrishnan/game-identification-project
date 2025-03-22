import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ClusterManager:
    def __init__(self):
        self.clusters = {}  # Format: {game_name: [embeddings]}

    def add_to_cluster(self, new_embedding, similarity_threshold=0.75):
        best_similarity = -1
        best_cluster = None
        
        for game, embeddings in self.clusters.items():
            similarities = cosine_similarity([new_embedding], embeddings)
            max_similarity = np.max(similarities)
            print(f"Max similarity with {game}: {max_similarity:.3f}")
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_cluster = game
        
        print(f"Best similarity: {best_similarity:.3f}")
        if best_similarity > similarity_threshold:
            self.clusters[best_cluster].append(new_embedding)
            return best_cluster
        else:
            return None

# Example usage
cluster_manager = ClusterManager()
games = ["minecraft", "fortnite", "fifa22", "fifa23"]
for game in games:
    cluster_manager.clusters[game] = np.load(f"{game}_embeddings.npy").tolist()

new_embedding = np.load("fortnite_embeddings.npy")[0]
detected_game = cluster_manager.add_to_cluster(new_embedding)
print(f"Detected game: {detected_game}")