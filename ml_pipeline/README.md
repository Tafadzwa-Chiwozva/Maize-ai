# ML Pipeline — MobileNetV2 Transfer Learning

This folder contains the full training pipeline for training a custom MobileNetV2
model on maize disease images. It is completely separate from the MVP backend —
running or modifying anything here will not affect the existing app.

---

## Folder structure

```
ml_pipeline/
├── data/
│   └── raw/                  ← YOU place your downloaded images here (see below)
│       ├── healthy/
│       ├── common_rust/
│       ├── leaf_blight/
│       └── gray_leaf_spot/
│
├── dataset/                  ← auto-created by prepare_data.py (gitignored)
│   ├── train/
│   ├── val/
│   └── test/
│
├── outputs/                  ← auto-created by evaluate.py (gitignored)
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── classification_report.txt
│
├── saved_model/              ← auto-created by train.py (gitignored)
│   ├── saved_model.pb
│   └── variables/
│
├── prepare_data.py           ← Step 1: split raw images into train/val/test
├── train.py                  ← Step 2: train MobileNetV2
├── evaluate.py               ← Step 3: evaluate on test set, generate plots
├── requirements_ml.txt       ← extra Python packages needed
└── README.md                 ← this file
```

---

## Step 0 — Download and place your dataset

1. Go to Kaggle and download the dataset:
   https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

2. After downloading, unzip it and copy images into the four class folders:

```
ml_pipeline/data/raw/healthy/          ← all healthy leaf images
ml_pipeline/data/raw/common_rust/      ← all common rust images
ml_pipeline/data/raw/leaf_blight/      ← all blight images
ml_pipeline/data/raw/gray_leaf_spot/   ← all gray leaf spot images
```

**Important naming rules:**
- Folder names must be exactly as shown above (lowercase, underscores)
- The class name comes from the folder name — any typo will create a wrong label
- Accepted image formats: `.jpg`, `.jpeg`, `.png`

Expected counts after placing images:

| Class | Expected images |
|---|---|
| healthy | ~1,162 |
| common_rust | ~1,306 |
| leaf_blight | ~1,146 |
| gray_leaf_spot | ~574 |

---

## Step 1 — Install extra dependencies

```bash
# From the project root (maize-ai/)
source venv/bin/activate
pip install -r ml_pipeline/requirements_ml.txt
```

---

## Step 2 — Split the dataset

```bash
python ml_pipeline/prepare_data.py
```

This script:
- Shuffles each class with a fixed random seed (reproducible)
- Copies images into `dataset/train/`, `dataset/val/`, `dataset/test/`
- Uses a 70% / 15% / 15% split per class
- Prints a summary table of image counts

---

## Step 3 — Train the model

**Recommended: Google Colab (free GPU, ~10–15 min)**

1. Upload this `ml_pipeline/` folder to Google Drive
2. Open a new Colab notebook at https://colab.research.google.com
3. Mount your Drive and run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Maize-ai/ml_pipeline
!python train.py
```

**Alternative: run locally (slow on CPU, ~3–6 hours)**

```bash
python ml_pipeline/train.py
```

The trained model is saved to `ml_pipeline/saved_model/`.

---

## Step 4 — Evaluate the model

```bash
python ml_pipeline/evaluate.py
```

This generates:
- A classification report (precision, recall, F1 per class) in the terminal
- A confusion matrix heatmap saved to `outputs/confusion_matrix.png`
- Training accuracy and loss curve plots saved to `outputs/`

---

## Step 5 — Plug the trained model into the backend

Once you are happy with the evaluation results:

1. Copy `ml_pipeline/saved_model/` to `model_v2/` in the project root
2. In `predict.py`, change `MODEL_DIR = "./model"` to `MODEL_DIR = "./model_v2"`
3. Restart the server — the frontend works unchanged

---

## Model architecture summary

| Component | Detail |
|---|---|
| Base model | MobileNetV2 (ImageNet pre-trained) |
| Input size | 224 × 224 pixels |
| Optimizer | Nadam |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Total epochs | 20 (5 frozen + 15 fine-tuned) |
| Classes | healthy, common_rust, leaf_blight, gray_leaf_spot |
