# download_model.py
# Run this script ONCE to download the pre-trained model from Hugging Face.
# After it finishes, a ./model/ folder will appear with all model files.
# You do NOT need to run this again unless you delete the model folder.

import os
from huggingface_hub import snapshot_download

# Path where the model files will be saved on your machine
MODEL_DIR = "./model"

# Check if the model is already downloaded to avoid re-downloading
if os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
    print("Model already downloaded. Nothing to do.")
else:
    print("Downloading model from Hugging Face (this may take a minute)...")

    # snapshot_download fetches every file in the HuggingFace repo
    # repo_id is the "username/model-name" from the HuggingFace URL
    snapshot_download(
        repo_id="eligapris/maize-diseases-detection",
        local_dir=MODEL_DIR,
    )

    print(f"Done! Model saved to: {os.path.abspath(MODEL_DIR)}")
    print("\nFiles downloaded:")
    for f in os.listdir(MODEL_DIR):
        print(f"  {f}")
