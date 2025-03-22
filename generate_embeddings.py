import clip
import torch
import numpy as np
from PIL import Image
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(image_path):
    image = Image.open(image_path)
    preprocessed = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(preprocessed)
    return embedding.cpu().numpy().flatten()  # Shape: (512,)

# Generate embeddings for all games
games = ["minecraft", "fortnite", "fifa22", "fifa23"]
for game in games:
    embeddings = []
    frame_folder = f"{game}_frames"
    for frame_file in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame_file)
        embedding = get_embedding(frame_path)
        embeddings.append(embedding)
    np.save(f"{game}_embeddings.npy", embeddings)