from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tempfile
import os
from generate_embeddings import get_embedding
from dynamic_expansion import ClusterManager

app = FastAPI()
cluster_manager = ClusterManager()

# Load initial clusters (for demo)
games = ["minecraft", "fortnite", "fifa22", "fifa23"]
for game in games:
    cluster_manager.clusters[game] = np.load(f"{game}_embeddings.npy").tolist()

@app.post("/detect_game")
async def detect_game(video: UploadFile = File(...)):
    # Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name
    
    # Extract a test frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cv2.imwrite("temp_frame.jpg", frame)
    cap.release()
    
    # Generate embedding
    embedding = get_embedding("temp_frame.jpg")
    
    # Detect game
    game = cluster_manager.add_to_cluster(embedding)
    return {"game": game}