import cv2
import numpy as np
from tqdm import tqdm
import os

def detect_faces(image):
    """
    Detect faces in an image using OpenCV's Haar Cascade
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return faces

def extract_face_features(image, face_coords, target_size=(128, 128)):
    """
    Extract and preprocess face region
    """
    x, y, w, h = face_coords
    
    # Add some margin around the face
    margin = int(0.2 * w)
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(image.shape[1], x + w + margin)
    y_end = min(image.shape[0], y + h + margin)
    
    # Extract face region
    face = image[y_start:y_end, x_start:x_end]
    
    # Resize to target size
    face = cv2.resize(face, target_size)
    
    return face

def compute_noise_features(image):
    """
    Extract noise features using ELA (Error Level Analysis)
    """
    # Save the image with a specific quality
    temp_filename = 'temp_ela.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Read the saved image
    saved_image = cv2.imread(temp_filename)
    
    # Calculate the difference
    ela_image = cv2.absdiff(image, saved_image) * 10
    
    # Remove the temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return ela_image

def extract_features_from_frame(frame_path, save_features=False, output_dir=None):
    """
    Extract relevant features from a frame
    """
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Warning: Could not read image at {frame_path}")
        return None
    
    # Detect faces
    faces = detect_faces(image)
    
    if len(faces) == 0:
        return None
    
    # Process the largest face (assuming it's the main subject)
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    
    # Extract face region
    face_img = extract_face_features(image, largest_face)
    
    # Compute noise features
    ela_features = compute_noise_features(face_img)
    
    # If save_features is True, save the processed images
    if save_features and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(frame_path)
        face_path = os.path.join(output_dir, f"face_{filename}")
        ela_path = os.path.join(output_dir, f"ela_{filename}")
        cv2.imwrite(face_path, face_img)
        cv2.imwrite(ela_path, ela_features)
    
    return {
        'face': face_img,
        'ela': ela_features
    }

def process_dataset_features(input_dir, output_dir=None, save_features=False):
    """
    Process all frames in a dataset and extract features
    """
    features = []
    labels = []
    
    # Process real images
    real_dir = os.path.join(input_dir, 'real')
    print("Processing real images...")
    for img_file in tqdm(os.listdir(real_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(real_dir, img_file)
            feature_dict = extract_features_from_frame(
                img_path, 
                save_features=save_features,
                output_dir=os.path.join(output_dir, 'real') if output_dir else None
            )
            if feature_dict:
                features.append(feature_dict)
                labels.append(0)  # 0 for real
    
    # Process fake images
    fake_dir = os.path.join(input_dir, 'fake')
    print("Processing fake images...")
    for img_file in tqdm(os.listdir(fake_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(fake_dir, img_file)
            feature_dict = extract_features_from_frame(
                img_path, 
                save_features=save_features,
                output_dir=os.path.join(output_dir, 'fake') if output_dir else None
            )
            if feature_dict:
                features.append(feature_dict)
                labels.append(1)  # 1 for fake
    
    return features, labels

if __name__ == "__main__":
    # Example usage
    features, labels = process_dataset_features(
        input_dir="data/processed/train",
        output_dir="data/features/train",
        save_features=True
    )