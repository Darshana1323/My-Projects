from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# Load the saved model
model = load_model('bird_species_recognition_model.keras')

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

def predict_bird_species(img_path):
    """
    Function to predict bird species from an image.

    Args:
        img_path (str): Path to the test image.

    Returns:
        str: Predicted bird species.
    """
    # Load and preprocess the test image
    img = load_img(img_path, target_size=(224, 224))  # Resize image to match input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

# Example usage
if __name__ == "__main__":
    test_image_path = r"D:\My Project\project\New folder\New folder\uploaded_images\test.jpg"

    result = predict_bird_species(test_image_path)
    print(f"Predicted bird species: {result}")
