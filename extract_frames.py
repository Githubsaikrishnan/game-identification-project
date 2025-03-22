import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:  # Extract every 30th frame
            cv2.imwrite(f"{output_folder}/frame_{frame_id}.jpg", frame)
            frame_id += 1
        count += 1
    cap.release()

# Extract frames for all games
games = ["minecraft", "fortnite", "fifa22", "fifa23"]
for game in games:
    extract_frames(f"{game}.mp4", f"{game}_frames")