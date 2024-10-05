import os
from scipy.spatial.distance import cosine
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import extract_middle_frame

# Define paths to training and test datasets
train_data_path = 'traindata'
test_data_path = 'test'
results_file = 'Results.csv'

# Get the instance of HandShapeFeatureExtractor (Singleton)
extractor = HandShapeFeatureExtractor.get_instance()

# Function to extract features from all videos in a given folder
def extract_features_from_videos(folder_path):
    features = []
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]  # Filter only .mp4 files
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        
        # Extract the middle frame of the video
        middle_frame = extract_middle_frame(video_path)
        if middle_frame is None:
            print(f"Warning: Could not extract frame from {video_file}. Skipping.")
            continue
        
        # Extract features using the HandShapeFeatureExtractor
        feature_vector = extractor.extract_feature(middle_frame)
        
        # Store video label (assumed to be filename without extension) and the feature vector
        video_label = int(os.path.splitext(video_file)[0])  # Assuming the filename is the label, e.g., "0.mp4"
        features.append((video_label, feature_vector))
    
    return features

# Extract features from training data
print("Extracting features from training data...")
training_features = extract_features_from_videos(train_data_path)

# Extract features from test data
print("Extracting features from test data...")
test_features = extract_features_from_videos(test_data_path)

# Classify test gestures using cosine similarity
print("Classifying test gestures...")
results = []

for test_label, test_feature in test_features:
    min_distance = float('inf')
    recognized_label = -1

    for train_label, train_feature in training_features:
        # Calculate cosine similarity between test feature and training feature
        distance = cosine(test_feature, train_feature)
        if distance < min_distance:
            min_distance = distance
            recognized_label = train_label

    # Append the recognized label to the results list
    results.append(recognized_label)

# Save the results to Results.csv
print(f"Saving results to {results_file}...")
with open(results_file, 'w') as f:
    for result in results:
        f.write(f"{result}\n")

print("Classification complete. Results saved.")
