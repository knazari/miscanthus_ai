import cv2
import os

def extract_frames(video_path, output_folder, interval=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frames at specified interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from the video.")

root_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF"
extract_frames(root_path + "/IMG_8793.MOV", root_path + "/images", interval=10)
