import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil
from sklearn.model_selection import train_test_split

def extract_frames(video_path, frames_dir, num_frames=30, resize=(128, 128)):
    """
    Extract frames from a video file and save to directory
    """
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video has fewer frames than num_frames, extract all frames
    if frame_count <= num_frames:
        sample_indices = range(frame_count)
    else:
        # Sample frames uniformly
        sample_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    frame_paths = []
    for i in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            frame = cv2.resize(frame, resize)
            frame_path = os.path.join(frames_dir, f"{os.path.basename(video_path)}_frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    video.release()
    return frame_paths

def process_dataset(real_videos_dir, fake_videos_dir, output_dir, num_frames_per_video=30):
    """
    Process all videos in dataset directories
    """
    # Create output directories
    train_real_dir = os.path.join(output_dir, 'train', 'real')
    train_fake_dir = os.path.join(output_dir, 'train', 'fake')
    val_real_dir = os.path.join(output_dir, 'val', 'real')
    val_fake_dir = os.path.join(output_dir, 'val', 'fake')
    test_real_dir = os.path.join(output_dir, 'test', 'real')
    test_fake_dir = os.path.join(output_dir, 'test', 'fake')
    
    for directory in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir, test_real_dir, test_fake_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process real videos
    real_videos = [os.path.join(real_videos_dir, f) for f in os.listdir(real_videos_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    # Process fake videos
    fake_videos = [os.path.join(fake_videos_dir, f) for f in os.listdir(fake_videos_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    # Split videos into train, validation, test sets (70%, 15%, 15%)
    real_train, real_temp = train_test_split(real_videos, test_size=0.3, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    
    fake_train, fake_temp = train_test_split(fake_videos, test_size=0.3, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)
    
    # Process videos and extract frames
    print("Processing real training videos...")
    for video in tqdm(real_train):
        extract_frames(video, train_real_dir, num_frames_per_video)
    
    print("Processing fake training videos...")
    for video in tqdm(fake_train):
        extract_frames(video, train_fake_dir, num_frames_per_video)
    
    print("Processing real validation videos...")
    for video in tqdm(real_val):
        extract_frames(video, val_real_dir, num_frames_per_video)
    
    print("Processing fake validation videos...")
    for video in tqdm(fake_val):
        extract_frames(video, val_fake_dir, num_frames_per_video)
    
    print("Processing real testing videos...")
    for video in tqdm(real_test):
        extract_frames(video, test_real_dir, num_frames_per_video)
    
    print("Processing fake testing videos...")
    for video in tqdm(fake_test):
        extract_frames(video, test_fake_dir, num_frames_per_video)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    # Example usage
    process_dataset(
        real_videos_dir="data/real",
        fake_videos_dir="data/fake",
        output_dir="data/processed",
        num_frames_per_video=30
    )