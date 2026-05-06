# predict.py
# The core prediction module. Given a path to a maize leaf image, this returns
# the disease class, confidence score, and a farmer-friendly description.
#
# Usage:
#   python predict.py images/test_leaf.jpg
#
# Or import the predict() function into your web backend (app.py).

import os
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

# Folder where the downloaded model files live
MODEL_DIR = "./model"

# The 4 disease categories the model was trained to recognise.
# Order matters — it must match the order the model was trained with.
CLASSES = ["Healthy", "Gray_Leaf_Spot", "Blight", "Common_Rust"]

# Image size the model expects as input (width x height in pixels)
IMAGE_SIZE = (300, 300)

# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    """Load the TensorFlow SavedModel from disk.

    tf.saved_model.load() reads the saved_model.pb file plus the variables/
    folder and returns a callable model object.
    """
    if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_DIR}'. "
            "Run `python download_model.py` first."
        )
    print("Loading model...")
    model = tf.saved_model.load(MODEL_DIR)
    print("Model loaded.")
    return model


def load_class_details():
    """Load the detailed disease descriptions from classes_detailed.json.

    The JSON has a nested structure: { "details": { "Healthy": {...}, ... } }
    We return just the inner "details" dict for easy class-name lookup.
    """
    path = os.path.join(MODEL_DIR, "classes_detailed.json")
    with open(path, "r") as f:
        data = json.load(f)
    # Return only the nested "details" section, keyed by class name
    return data.get("details", {})


# Load model and class details once when this module is imported.
# This avoids reloading on every prediction call (expensive operation).
model = load_model()
class_details = load_class_details()


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> tf.Tensor:
    """Open an image file and convert it into the format the model expects.

    Steps:
    1. Open the image using Pillow (PIL)
    2. Convert to RGB (removes any alpha channel from PNG files)
    3. Resize to 300x300 pixels (the model's required input size)
    4. Convert to a NumPy array of float32 numbers
    5. Add a batch dimension: shape goes from (300,300,3) to (1,300,300,3)
       because the model expects a batch (group) of images, even if it's just one
    """
    img = Image.open(image_path).convert("RGB")   # ensure 3 colour channels
    img = img.resize(IMAGE_SIZE)                  # resize to 300x300
    arr = np.array(img, dtype=np.float32)         # pixel values as floats
    arr = arr[None]                                # add batch dimension → (1,300,300,3)
    return tf.constant(arr)                        # wrap in a TensorFlow tensor


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(image_path: str) -> dict:
    """Run the model on one image and return a structured result.

    Returns a dict with:
    - class:        The predicted disease name (e.g. "Common_Rust")
    - confidence:   How sure the model is, as a percentage (e.g. 87.4)
    - description:  A plain-language explanation of the disease
    - all_scores:   Confidence % for all 4 classes (useful for debugging)
    """
    # Preprocess the image into the format the model needs
    inp = preprocess_image(image_path)

    # Run the model — returns an array of 4 raw scores (called logits or probabilities)
    scores = model(inp)[0].numpy()   # [0] removes the batch dimension

    # Find which class has the highest score
    class_index = int(scores.argmax())
    class_name = CLASSES[class_index]

    # Convert the winning score to a percentage
    confidence = round(float(scores[class_index]) * 100, 1)

    # Look up the farmer-friendly description for the predicted class.
    # The JSON structure differs per class, so we build a message from
    # whichever fields are available: description, causative_agent, symptoms.
    detail = class_details.get(class_name, {})
    if "description" in detail:
        # Healthy class has a plain "description" field
        description = detail["description"]
    elif "symptoms" in detail:
        # Disease classes list their symptoms as an array — join into one sentence
        agent = detail.get("causative_agent", "Unknown agent")
        symptoms_list = detail["symptoms"]
        description = f"Caused by {agent}. Symptoms: {symptoms_list[0]}"
    else:
        description = "No description available."

    # Build the full result dictionary
    return {
        "class": class_name,
        "confidence": confidence,
        "description": description,
        # All 4 class scores as percentages — helpful to see runner-up classes
        "all_scores": {
            CLASSES[i]: round(float(s) * 100, 1)
            for i, s in enumerate(scores)
        },
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running directly from the terminal:
    #   python predict.py images/test_leaf.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "model/image.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    result = predict(image_path)

    print("\n" + "=" * 40)
    print(f"  Disease   : {result['class']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Info      : {result['description']}")
    print("=" * 40)
    print(f"\nAll scores: {result['all_scores']}")
