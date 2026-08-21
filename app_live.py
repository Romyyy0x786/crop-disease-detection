import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import sys
import pickle
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Live Crop Disease Detector", page_icon="🎥", layout="centered")

# ---------------- LOAD MODELS (cached) ----------------
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

def retrieve_disease_info(predicted_class_name, k=1):
    query_text = predicted_class_name.replace("___", " ").replace("_", " ")
    query_embedding = embed_model.encode([query_text]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k)
    idx = indices[0][0]
    name = disease_names[idx]
    return disease_info[name]

# ---------------- SHARED STATE (to pass prediction from video thread to main UI) ----------------
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
    st.session_state.latest_confidence = 0.0

# ---------------- VIDEO PROCESSOR ----------------
class DiseaseDetector(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_frames = 10  # only run prediction every 10th frame for performance
        self.last_label = "Analyzing..."
        self.last_confidence = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.skip_frames == 0:
            # Preprocess for model
            resized = cv2.resize(img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_arr = np.expand_dims(rgb.astype(np.float32), axis=0)  # no /255 - matches training fix

            predictions = cnn_model.predict(input_arr, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_class = idx_to_class[predicted_idx]

            self.last_label = predicted_class.replace("___", " - ").replace("_", " ")
            self.last_confidence = confidence

            # Update session state so main UI can show RAG info
            st.session_state.latest_prediction = predicted_class
            st.session_state.latest_confidence = confidence

        # Overlay text on frame
        label_text = f"{self.last_label} ({self.last_confidence*100:.1f}%)"
        cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- UI ----------------
st.title("🎥 Live Crop Disease Detection")
st.write("Point your camera at a leaf. Predictions update every few frames.")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="disease-detection",
    video_processor_factory=DiseaseDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")

# Show RAG info for latest confident prediction
if st.session_state.latest_prediction and st.session_state.latest_confidence > 0.6:
    predicted_class = st.session_state.latest_prediction
    display_name = predicted_class.replace("___", " - ").replace("_", " ")

    st.subheader(f"Latest Detection: {display_name}")
    st.write(f"Confidence: {st.session_state.latest_confidence*100:.2f}%")

    if "healthy" not in predicted_class.lower():
        info = retrieve_disease_info(predicted_class)
        st.write(f"**Symptoms:** {info['symptoms']}")
        st.write(f"**Prevention:** {info['prevention']}")
        st.write(f"**Treatment:** {info['treatment']}")
    else:
        st.info("This looks healthy!")
else:
    st.write("Waiting for a confident detection (>60%)...")

st.caption("Note: predictions update every ~10 frames for performance. Hold the leaf steady for best results.")