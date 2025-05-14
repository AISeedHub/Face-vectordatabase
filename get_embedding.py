import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# Initialize the FaceAnalysis app
MODEL_NAME = "buffalo_l"  # Pre-trained model for face recognition
DETECTOR_BACKEND = "insightface"  # Using insightface for detection and embedding

# Initialize the FaceAnalysis app
app = FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU (ctx_id=0) or CPU (ctx_id=-1)

# Load the face recognition model directly
model = model_zoo.get_model(MODEL_NAME)
model.prepare(ctx_id=0)  # Use GPU (ctx_id=0) or CPU (ctx_id=-1)

def get_face_embedding(image_path):
    """
    Detects a face in an image and returns its embedding.
    Returns None if no face is detected or multiple faces are found (for simplicity).
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return None

        # Detect faces in the image
        faces = app.get(img)

        # Ensure only one face is detected
        if len(faces) == 1:
            face = faces[0]
            return face.embedding  # Return the embedding vector
        elif len(faces) == 0:
            print(f"No face detected in {image_path}")
            return None
        else:
            print(f"Multiple faces detected in {image_path}. Please use an image with a single face.")
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_face_embedding_direct(image_path, bbox=None):
    """
    Extracts face embedding directly using the model, bypassing the app interface.
    Requires a bounding box for the face.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return None

        # Crop the face using the bounding box
        if bbox is None:
            print("Error: No bounding box provided. Use full image detection instead.")
            face = img  # Use the whole image if no bbox is provided
        elif len(bbox) != 4:
            print(f"Error: Invalid bounding box {bbox}. Expected format: [x1, y1, x2, y2]")
            return None
        else:
            x1, y1, x2, y2 = bbox
            face = img[int(y1):int(y2), int(x1):int(x2)]  # Crop the face
            if face.size == 0:
                print(f"Error: Invalid bounding box {bbox}")
                return None

        # Preprocess the face
        face = cv2.resize(face, (112, 112))  # Resize to 112x112 as required by the model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = np.transpose(face, (2, 0, 1))  # Change to CHW format
        face = np.expand_dims(face, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Get the face embedding using the model
        embedding = model.forward(face)
        return embedding
    except Exception as e:
        print(f"Error processing {image_path} with bbox {bbox}: {e}")
        return None

# --- Test the functions (optional) ---
# Example usage of get_face_embedding_direct
# bbox = [50, 50, 200, 200]  # Example bounding box [x1, y1, x2, y2]
# embedding = get_face_embedding_direct("path/to/your/test_face.jpg", bbox)
# if embedding is not None:
#     print(f"Embedding vector length: {len(embedding)}")
# else:
#     print("Could not get embedding.")