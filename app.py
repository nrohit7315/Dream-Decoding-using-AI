import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import tempfile
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from scipy.signal import welch
import json
from pymongo import MongoClient
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from base64 import b64encode
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import pyedflib
import io
from  pyedflib import FILETYPE_EDF
import datetime

output_dir = "./results/edf_files"
os.makedirs(output_dir, exist_ok=True)

def csv_to_edf(csv_bytes, edf_filename):
    data = pd.read_csv(io.BytesIO(csv_bytes), header=None, dtype=str)
    data[0] = data[0].str.replace(r'[^0-9\.\-\+eE]', '', regex=True)
    signal = pd.to_numeric(data[0], errors='coerce').dropna().to_numpy()
    n_channels = 1
    channel_info = [{
        'label': 'EEG',
        'dimension': 'uV',
        'sample_frequency': 8,
        'physical_min': float(np.min(signal)),
        'physical_max': float(np.max(signal)),
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }]
    with pyedflib.EdfWriter(edf_filename, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as file:
        file.setSignalHeaders(channel_info)
        file.writeSamples([signal])
    print(" EDF file created successfully:", edf_filename)
    return edf_filename

BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
         'beta': (12, 30), 'gamma': (30, 45)}

MODEL_PATH_256 = './results/baseline_models_256.pkl'      
MODEL_PATH_8FEAT = './results/baseline_models_8feat.pkl' 
EMOTION_MODEL_PATH = './results/emotion_model_lstm.pth'

class EmotionLSTM(nn.Module):
    def __init__(self, input_size=1, num_classes=3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

emotion_model = None
try:
    if os.path.exists(EMOTION_MODEL_PATH):
        emotion_model = EmotionLSTM()
        emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location='cpu'))
        emotion_model.eval()
    else:
        emotion_model = None
except Exception as e:
    print("Could not load emotion model:", e)
    emotion_model = None

KEY = get_random_bytes(32)

def encrypt_data(plaintext):
    cipher = AES.new(KEY, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    return b64encode(cipher.nonce + tag + ciphertext).decode()

MONGO_URI = "mongodb://localhost:27017/"
try:
    client = MongoClient(MONGO_URI)
    db = client['eeg_analysis']
    users_collection = db['users']
except Exception as e:
    print("MongoDB connection failed:", e)
    client = None
    users_collection = None

def save_user_info_mongo(user_info, eeg_results):
    if users_collection is None:
        return
    record = {
        "timestamp": datetime.datetime.utcnow(),
        "user_info": {k: encrypt_data(str(v)) for k, v in user_info.items()},
        "eeg_results": eeg_results
    }
    users_collection.insert_one(record)

def load_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw

def preprocess_raw(raw):
    picks = mne.pick_types(raw.info, eeg=True)
    return raw.get_data(picks=picks)

def epoch_data(data, sfreq, epoch_s=30):
    n_samples = int(sfreq) * int(epoch_s)
    if n_samples == 0:
        return []
    n_epochs = data.shape[1] // n_samples
    return [data[:, i*n_samples:(i+1)*n_samples] for i in range(n_epochs)]

def extract_epoch_features(epoch, sfreq):
    """Return features vector for an epoch.
       Produces bandpower (delta..gamma) for each channel and Hjorth + entropy per channel.
    """
    feats = []
    for ch in range(epoch.shape[0]):
        f, Pxx = welch(epoch[ch], sfreq, nperseg=min(256, epoch.shape[1]))
        total = np.trapz(Pxx, f) + 1e-12
        for lo, hi in BANDS.values():
            idx = np.logical_and(f >= lo, f <= hi)
            feats.append(np.trapz(Pxx[idx], f[idx]) / total)
    for ch in range(epoch.shape[0]):
        sig = epoch[ch]
        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)
        var0 = np.var(sig) + 1e-12
        var1 = np.var(diff1) + 1e-12
        var2 = np.var(diff2) + 1e-12
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else 0.0
        f2, Pxx2 = welch(sig, sfreq, nperseg=min(256, len(sig)))
        P_norm = Pxx2 / (Pxx2.sum() + 1e-12)
        ent = -np.sum(P_norm * np.log2(P_norm + 1e-12))
        feats.extend([mobility, complexity, ent])
    return np.array(feats)

def emotion_predict(raw):
    if emotion_model is None:
        return "Model missing", "Emotion model not loaded.", np.array([0.0,0.0,0.0])
    try:
        raw_copy = raw.copy().filter(0.5, 45, fir_design='firwin', verbose=False)
    except Exception:
        raw_copy = raw
        try:
            raw_copy.filter(0.5, 45, fir_design='firwin', verbose=False)
        except Exception:
            pass
    data = raw_copy.get_data()[0]
    num_samples = 128
    eeg_segment = data[:num_samples] if len(data) >= num_samples else np.pad(data, (0, num_samples-len(data)))
    scaled = MinMaxScaler().fit_transform(eeg_segment.reshape(-1,1)).flatten()
    X_input = torch.tensor(scaled.reshape(1, num_samples, 1), dtype=torch.float32)
    with torch.no_grad():
        logits = emotion_model(X_input)
        probs = F.softmax(logits, dim=1).numpy().flatten()
    idx = int(np.argmax(probs))
    emotion_map = {0:'Happy 😊', 1:'Fear 😨', 2:'Neutral 😐'}
    description = {'Happy 😊': "Elevated mood and engagement.",
                   'Fear 😨': "Anxious or stressed activity detected.",
                   'Neutral 😐': "Baseline brain rhythm."}
    chosen = emotion_map.get(idx, "Unknown")
    return chosen, description.get(chosen, ""), probs

def dream_features(eeg, sf):
    mean, std = np.mean(eeg), np.std(eeg)
    bandpower_val = np.mean([np.trapz(welch(eeg[ch], sf, nperseg=min(len(eeg[ch]), max(256, sf*2)))[1]) for ch in range(eeg.shape[0])])
    return {"mean":mean, "std":std, "bandpower_1_30Hz":bandpower_val}


def load_ml_pack(path):
    """Load a joblib saved pack and return (rf, scaler, le, expected_feat_len)"""
    pack = joblib.load(path)
    if isinstance(pack, dict):
        rf = pack.get('rf') or pack.get('model') or pack.get('classifier')
        scaler = pack.get('scaler') or pack.get('sc') or None
        le = pack.get('le') or pack.get('label_encoder') or None
        expected_len = pack.get('feature_shape') if 'feature_shape' in pack else None
    else:
        rf = pack
        scaler = getattr(pack, 'scaler_', None) or None
        le = None
        expected_len = None
        if scaler is None and hasattr(pack, 'named_steps'):
            steps = pack.named_steps
            for s in steps.values():
                if hasattr(s, 'n_features_in_'):
                    expected_len = int(s.n_features_in_)
                    break
    if expected_len is None and scaler is not None and hasattr(scaler, 'n_features_in_'):
        expected_len = int(scaler.n_features_in_)
    return {'rf': rf, 'scaler': scaler, 'le': le, 'expected_len': expected_len}

def choose_and_load_model(n_features):
    """Decide whether to use 8-feat or 256-feat model based on n_features (or fallback heuristics)."""
    candidates = []
    if os.path.exists(MODEL_PATH_8FEAT):
        try:
            p = load_ml_pack(MODEL_PATH_8FEAT)
            candidates.append((p, MODEL_PATH_8FEAT))
        except Exception:
            pass
    if os.path.exists(MODEL_PATH_256):
        try:
            p = load_ml_pack(MODEL_PATH_256)
            candidates.append((p, MODEL_PATH_256))
        except Exception:
            pass
    for pack, path in candidates:
        if pack['expected_len'] == n_features:
            return pack, path
    best = None
    best_diff = 1e9
    for pack, path in candidates:
        if pack['expected_len'] is None:
            continue
        diff = abs(pack['expected_len'] - n_features)
        if diff < best_diff:
            best_diff = diff
            best = (pack, path)
    if best:
        return best
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No ML model files found. Please ensure model files exist at configured paths.")

def pad_or_trim_features(features_array, target_len):
    """features_array: (n_epochs, n_feats). Return array with second dim == target_len."""
    n_epochs, cur_len = features_array.shape
    if cur_len == target_len:
        return features_array
    elif cur_len < target_len:
        pad_width = target_len - cur_len
        return np.hstack([features_array, np.zeros((n_epochs, pad_width))])
    else:
        return features_array[:, :target_len]

# ===================== DREAM GENERATION USING EDENAI =====================
import datetime
import mne
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2
from gtts import gTTS

EDENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmQwMTY1ZjUtYmVlYy00YjA1LTg4Y2YtMWI2NGJkZWYyZWQyIiwidHlwZSI6ImFwaV90b2tlbiJ9.cNBgE2TAvJys-EPyJi2h2S05lbecdo075VTdWbEQ7r4"
EDENAI_URL = "https://api.edenai.run/v2/text/generation"

PEXELS_API_KEY = "Zi3EMpHvWuuMcOVIhogZFRM8DvRcbOdbGcr0Omr4tEDzjGGRgpeTuMZd"
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

def generate_dream_story(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw.get_data(return_times=True)

    mean_vals = np.mean(data, axis=1)
    std_vals = np.std(data, axis=1)
    power = np.mean(data**2, axis=1)

    summary = {
        "channels": raw.info["nchan"],
        "mean_activity": float(np.mean(mean_vals)),
        "std_dev": float(np.mean(std_vals)),
        "signal_power": float(np.mean(power))
    }

    prompt = (
        f"Based on these EEG features:\n"
        f"Channels: {summary['channels']}\n"
        f"Mean activity: {summary['mean_activity']:.4f}\n"
        f"Signal power: {summary['signal_power']:.4f}\n\n"
        f"Write a dreamy poetic story about the person’s dream. "
        f"Use emotional and vivid language."
    )

    headers = {
        "Authorization": f"Bearer {EDENAI_API_KEY}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    payload = {
        "providers": "openai",
        "text": prompt + "\n\nWrite 3 paragraphs (each 4–6 lines).",
        "temperature": 0.8,
        "max_tokens": 150
    }

    response = requests.post("https://api.edenai.run/v2/text/generation", headers=headers, json=payload)

    if response.status_code == 200:
        dream_text = response.json()["openai"]["generated_text"]
    else:
        print("Error:", response.text)
        dream_text = "A surreal dream of endless stars and floating thoughts."

    return dream_text


def create_dream_video(dream_text, output_dir):
    width, height = 1280, 720
    fps = 24
    duration_per_scene = 5
    frames_per_scene = duration_per_scene * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "dream_video.mp4")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    def fetch_image_for_scene(scene_text, save_path):
        params = {"query": scene_text, "per_page": 1}
        headers = {"Authorization": PEXELS_API_KEY}
        response = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params)
        if response.status_code != 200:
            return False
        data = response.json()
        if not data["photos"]:
            return False
        img_url = data["photos"][0]["src"]["landscape"]
        img_data = requests.get(img_url).content
        with open(save_path, "wb") as f:
            f.write(img_data)
        return True

    def text_to_speech(text, output_path):
        tts = gTTS(text)
        tts.save(output_path)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    scenes = [s.strip() for s in dream_text.split('.') if s.strip()]
    if not scenes:
        scenes = ["A peaceful void of light and color.", "Dreams weaving into reality."]

    for i, scene_text in enumerate(scenes):
        img_path = os.path.join(output_dir, f"scene_{i}.jpg")
        if not fetch_image_for_scene(scene_text, img_path):
            base_img = Image.new("RGB", (width, height), (0, 0, 0))
        else:
            base_img = Image.open(img_path).convert("RGB").resize((width, height))
        draw = ImageDraw.Draw(base_img)
        bbox = draw.textbbox((0, 0), scene_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(((width - text_w) // 2, height - text_h - 30), scene_text, font=font, fill="white")
        frame = np.array(base_img)[:, :, ::-1]
        for _ in range(frames_per_scene):
            video.write(frame)
        text_to_speech(scene_text, os.path.join(output_dir, f"audio_{i}.mp3"))

    video.release()
    return video_path













st.set_page_config(page_title="EEG Dream & Emotion Analysis", layout="wide")
st.title("🧠 EEG Dream & Emotion Analysis")

st.header("📝 Enter Your Info")
name = st.text_input("Full Name")
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male","Female","Other"])
user_info_complete = all([name, age, gender])

uploaded_file = st.file_uploader("Upload your EEG file (CSV or EDF)", type=["edf", "csv"])

if uploaded_file and user_info_complete:
    try:
        tmp_bytes = uploaded_file.read()
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.edf')]
        next_num = len(existing_files) + 1
        edf_save_path = os.path.join(output_dir, f"edf{next_num}.edf")
        if uploaded_file.name.endswith('.csv'):
            st.info("Converting CSV to EDF format...")
            csv_to_edf(tmp_bytes, edf_save_path)
        else:
            with open(edf_save_path, 'wb') as f:
                f.write(tmp_bytes)
        st.success(f"✅ EEG file saved successfully: {edf_save_path}")
        raw = load_eeg(edf_save_path)
        eeg_data = raw.get_data() * 1e6
        sfreq = int(raw.info['sfreq']) if 'sfreq' in raw.info else 256
        try:
            f = pyedflib.EdfReader(tmp_path)  # type: ignore
            signal_sample = f.readSignal(0)
            f.close()
            fs = sfreq if sfreq else 256
            freqs, psd = welch(signal_sample, fs, nperseg=min(len(signal_sample), 1024))
            def band_power(low, high):
                mask = (freqs >= low) & (freqs <= high)
                return np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
            delta = band_power(0.5, 4)
            theta = band_power(4, 8)
            alpha = band_power(8, 12)
            beta  = band_power(12, 30)
            gamma = band_power(30, 45)
            total_power = delta + theta + alpha + beta + gamma + 1e-12
            seed_value = int(total_power) % 10000
        except Exception as e:
            delta = theta = alpha = beta = gamma = 0.0
            seed_value = 0


        tabs = st.tabs(["Overview","Emotion Analysis","Sleep Stages","Dream & Video"])
        with tabs[0]:
            st.subheader("📊 EEG Summary & Heatmap")
            duration_s = raw.n_times / raw.info['sfreq'] if ('n_times' in raw.__dict__ and 'sfreq' in raw.info) or True else eeg_data.shape[1]/sfreq
            st.write(f"Channels: {raw.info['nchan']}, Sampling: {raw.info['sfreq']} Hz, Duration: {raw.n_times/raw.info['sfreq']:.2f}s")
            fig,ax = plt.subplots(figsize=(10,4))
            ax.imshow(eeg_data, aspect='auto', cmap='plasma', origin='lower')
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Channels")
            st.pyplot(fig)

        with tabs[1]:
            st.subheader("🎯 Emotion Prediction")
            emotion, desc, probs = emotion_predict(raw)
            st.markdown(f"**Predicted Emotion:** {emotion}")
            st.write(desc)
            prob_df = pd.DataFrame({"Emotion":['Happy 😊','Fear 😨','Neutral 😐'], "Probability":probs})
            st.dataframe(prob_df)
            fig,ax = plt.subplots(figsize=(6,2))
            bars = ax.barh(prob_df["Emotion"], prob_df["Probability"])
            for i,bar in enumerate(bars):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{probs[i]:.2f}", va='center', fontsize=9)
            ax.set_xlim(0,1)
            st.pyplot(fig)

        with tabs[2]:
            st.subheader("🛌 Sleep Stage Prediction")
            data = preprocess_raw(raw)
            epochs = epoch_data(data, sfreq)
            if len(epochs) == 0:
                st.warning("No full epochs found in the recording for the given epoch length; try a longer recording or adjust epoch length.")
            else:
                feature_list = [extract_epoch_features(ep, sfreq) for ep in epochs]
                features_array = np.vstack(feature_list)  

                n_feats = features_array.shape[1]

                try:
                    ml_pack, used_path = choose_and_load_model(n_feats)
                    rf_model = ml_pack['rf']
                    scaler = ml_pack['scaler']
                    le = ml_pack['le']
                    expected_len = ml_pack['expected_len']
                except Exception as e:
                    st.error(f"Could not load any ML model: {e}")
                    rf_model = None
                    scaler = None
                    le = None
                    expected_len = None

                if rf_model is not None:
                    if expected_len is None and scaler is not None and hasattr(scaler, 'n_features_in_'):
                        expected_len = int(scaler.n_features_in_)
                    target_len = expected_len if expected_len is not None else n_feats
                    if n_feats != target_len:
                        old = n_feats
                        features_array = pad_or_trim_features(features_array, target_len)
                        st.warning(f"Feature length adjusted from {old} -> {target_len} to match the model.")

                    try:
                        if scaler is not None:
                            features_scaled = scaler.transform(features_array)
                        else:
                            features_scaled = StandardScaler().fit_transform(features_array)
                    except Exception as e:
                        st.error("Scaler transform failed: " + str(e))
                        features_scaled = StandardScaler().fit_transform(features_array)

                    # Predict
                    try:
                        preds = rf_model.predict(features_scaled)
                        preds_proba = rf_model.predict_proba(features_scaled) if hasattr(rf_model, "predict_proba") else None
                    except Exception as e:
                        st.error("Model prediction failed: " + str(e))
                        preds = np.zeros((features_scaled.shape[0],), dtype=int)
                        preds_proba = None
                    if le is not None and hasattr(le, 'inverse_transform'):
                        try:
                            pred_labels = le.inverse_transform(preds)
                        except Exception:
                            pred_labels = [str(p) for p in preds]
                    else:
                        pred_labels = [str(p) for p in preds]

                    df_stages = pd.DataFrame({"Predicted Stage": pred_labels})

                    unique_labels = list(pd.unique(pred_labels))
                    stage_map = {lab: i for i, lab in enumerate(unique_labels)}

                    numeric_stages = df_stages["Predicted Stage"].map(stage_map).tolist()

                    fig, ax = plt.subplots(figsize=(10, 3))
                    x_vals = list(range(len(numeric_stages)))
                    plot_y = numeric_stages.copy()
                    if len(plot_y) == 1:
                        x_vals = [0, 1]
                        plot_y = [plot_y[0], plot_y[0]]

                    ax.plot(x_vals, plot_y, linewidth=2, marker='o', markersize=4)
                    ax.set_yticks(list(stage_map.values()))
                    ax.set_yticklabels(list(stage_map.keys()))
                    ax.set_xlabel("Epoch (Time)")
                    ax.set_ylabel("Sleep Stage")
                    ax.set_title("Predicted Sleep Stage Over Time")
                    ax.grid(True, linestyle='--', alpha=0.4)
                    st.pyplot(fig)

                    st.dataframe(df_stages)

                    if preds_proba is not None:
                        st.markdown("**Last Epoch Stage Probabilities:**")
                        proba_df = pd.DataFrame(preds_proba[-1:], columns=(le.classes_ if hasattr(le, 'classes_') else [str(i) for i in range(preds_proba.shape[1])]))
                        st.bar_chart(proba_df)
                else:
                    st.info("ML model not available. Place your model files in the ./results folder with names:\n"
                    " - baseline_models.pkl (256-feature) or\n"
                    " - baseline_models_8feat.pkl (8-feature)")

        with tabs[3]:
            st.subheader("🌙 Dream Story Generation with EdenAI")
            dream_text = generate_dream_story(edf_save_path)

            st.markdown("### 💤 **Generated Dream Story**")
            st.markdown(
                f"""
                <div style="
                    background-color: #111827;
                    color: #E5E7EB;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    line-height: 1.6;
                    font-size: 18px;">
                    {dream_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            os.makedirs("./results/Dream_Results", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dream_dir = os.path.join("./results/Dream_Results", f"dream_{timestamp}")
            os.makedirs(dream_dir, exist_ok=True)

            st.info("🎬 Creating dream video scenes...")
            video_path = create_dream_video(dream_text, dream_dir)

            st.success("✅ Dream video created successfully!")
            st.markdown("### 🎥 Dream Visualization")
            if os.path.exists(video_path):
                st.video(video_path, format="video/mp4", start_time=0)

                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="📥 Download Dream Video",
                    data=video_bytes,
                    file_name="dream_video.mp4",
                    mime="video/mp4"
                )

            else:
                st.error("❌ Video file not found. Please try regenerating the dream.")      


        eeg_results = {"emotion":emotion,"emotion_prob":probs.tolist() if isinstance(probs, (np.ndarray, list)) else [0,0,0],"dream_text":dream_text}
        save_user_info_mongo({"name":name,"age":age,"gender":gender}, eeg_results)
        st.success("✅ All info and EEG analysis securely saved to MongoDB.")

    except Exception as e:
        st.exception(e)
else:
    st.info("Fill all your info and upload an EEG EDF file to start analysis.")
