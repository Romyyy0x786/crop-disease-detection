import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱", layout="centered")

# ---------------- LOAD MODELS (cached so it only loads once) ----------------
@st.cache_resource
def load_all_models():
    cnn_model = load_model("crop_disease_model.h5")

    with open("class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index("disease_index.faiss")

    with open("disease_names.pkl", "rb") as f:
        disease_names = pickle.load(f)

    sys.path.append(".")
    from disease_info import disease_info

    return cnn_model, idx_to_class, embed_model, faiss_index, disease_names, disease_info

cnn_model, idx_to_class, embed_model, faiss_index, disease_names, disease_info = load_all_models()

# ---------------- RAG RETRIEVAL FUNCTION ----------------
def retrieve_disease_info(predicted_class_name, k=1):
    query_text = predicted_class_name.replace("___", " ").replace("_", " ")
    query_embedding = embed_model.encode([query_text]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k)
    idx = indices[0][0]
    name = disease_names[idx]
    return disease_info[name]

# ---------------- UI ----------------
st.title("🌱 Crop Disease Detection & Prevention System")
st.write("Upload a photo of a crop leaf to detect disease and get prevention/treatment advice.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing leaf..."):
        # Preprocess (NO /255 - matches training fix, EfficientNetB0 handles rescaling internally)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = cnn_model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = idx_to_class[predicted_idx]

        # Clean up display name
        display_name = predicted_class.replace("___", " - ").replace("_", " ")

        # Retrieve disease info via RAG
        info = retrieve_disease_info(predicted_class)

    st.success(f"**Prediction:** {display_name}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    if "healthy" in predicted_class.lower():
        st.balloons()
        st.info("This leaf looks healthy! No treatment needed.")
    else:
        st.subheader("📋 Symptoms")
        st.write(info["symptoms"])

        st.subheader("🛡️ Prevention")
        st.write(info["prevention"])

        st.subheader("💊 Treatment")
        st.write(info["treatment"])

    with st.expander("See raw prediction confidence for top classes"):
        top5_idx = np.argsort(predictions[0])[-5:][::-1]
        for idx in top5_idx:
            cls_name = idx_to_class[idx].replace("___", " - ").replace("_", " ")
            st.write(f"{cls_name}: {predictions[0][idx]*100:.2f}%")

st.markdown("---")
st.caption("Built with EfficientNetB0 (CNN) + FAISS/Sentence-Transformers (RAG) | Sohail Khan")