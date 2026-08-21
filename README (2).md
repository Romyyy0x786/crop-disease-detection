# 🌱 Crop Disease Detection & Prevention System

An AI-powered full-stack application that detects crop leaf diseases from images and provides grounded prevention/treatment recommendations using a Retrieval-Augmented Generation (RAG) pipeline.

**Live Demo:** *(add your Streamlit Cloud link here once deployed)*

## What it does

- Upload a photo of a crop leaf through the web interface
- A CNN (EfficientNetB0, transfer learning) classifies the leaf into one of 38 disease/healthy categories across 14 crop types
- The predicted disease is used to query a RAG pipeline (FAISS + Sentence Transformers) that retrieves accurate symptoms, prevention, and treatment information
- Results are displayed instantly with confidence scores

A live-camera variant (`app_live.py`) is also included, using `streamlit-webrtc` for real-time webcam-based detection.

## How it works

```
Leaf Image → CNN (EfficientNetB0) → Disease Prediction → RAG Retrieval (FAISS) → Prevention & Treatment Info
```

1. **Image preprocessing & CNN Classification** — Images are resized to 224x224 and passed through a fine-tuned EfficientNetB0 model (transfer learning from ImageNet), trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
2. **RAG-based Info Retrieval** — Disease-specific information (symptoms, prevention, treatment) is embedded using Sentence Transformers (`all-MiniLM-L6-v2`) and indexed with FAISS for fast similarity search.
3. **Streamlit UI** — Combines both components into an interactive, easy-to-use web app.

## Model Performance

- **Validation Accuracy: 96.88%**
- 38 classes covering Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- Trained using transfer learning (frozen EfficientNetB0 base + custom classification head) on Google Colab (GPU)

## Tech Stack

**Machine Learning**
- TensorFlow / Keras — EfficientNetB0 (transfer learning)
- Sentence Transformers — text embeddings for RAG
- FAISS — vector similarity search

**Application**
- Streamlit — web interface
- streamlit-webrtc — live camera feed (optional variant)
- OpenCV, Pillow — image processing

## Project Structure

```
crop-disease-detection/
├── app.py                    # Main Streamlit app (image upload)
├── app_live.py                # Live camera variant (streamlit-webrtc)
├── crop_disease_model.h5      # Trained CNN model
├── class_indices.json         # Class label mappings
├── disease_info.py            # Disease knowledge base (symptoms/prevention/treatment)
├── disease_index.faiss        # FAISS vector index
├── disease_names.pkl          # Disease name list (matches FAISS index order)
├── src/
│   └── train_model.py         # CNN training script (run on Colab with GPU)
└── requirements.txt
```

## Running locally

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/Romyyy0x786/crop-disease-detection.git
cd crop-disease-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Open the printed local URL (typically `http://localhost:8501`).

### Run the live camera version (optional)

```bash
streamlit run app_live.py
```

## Training your own model

The CNN was trained on Google Colab using a GPU runtime for speed. See
`src/train_model.py` for the full training pipeline (EfficientNetB0 transfer
learning, data augmentation, early stopping).

## Known Limitations

- The model is a closed-set classifier — it will always predict one of the 38 trained classes, even for images that aren't leaves at all (no out-of-distribution detection). A confidence threshold is used in the UI to flag low-confidence predictions.
- Live camera predictions update every ~10 frames for performance reasons, not every frame.

## Future Improvements

- Add an out-of-distribution / "not a leaf" detector
- Expand to more crops and disease classes
- Deploy with a managed vector database for the RAG component at scale
- Add multi-language support for treatment recommendations

---
Built by Sohail Khan
