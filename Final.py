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

def csv_to_edf(csv_bytes, edf_filename):
    data = pd.read_csv(io.BytesIO(csv_bytes), header=None, dtype=str)

    # Clean unwanted characters and convert to numeric
    data[0] = data[0].str.replace(r'[^0-9\.\-\+eE]', '', regex=True)
    signal = pd.to_numeric(data[0], errors='coerce').dropna().to_numpy()

    # EDF channel metadata
    n_channels = 1
    channel_info = [{
        'label': 'EEG',
        'dimension': 'uV',
        'sample_frequency': 256,  # adjust if you know actual rate
        'physical_min': float(np.min(signal)),
        'physical_max': float(np.max(signal)),
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }]

    # Write EDF
    with pyedflib.EdfWriter(edf_filename, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as file:
        file.setSignalHeaders(channel_info)
        file.writeSamples([signal])

    print("✅ EDF file created successfully:", edf_filename)
    return edf_filename


# ------------------- Constants -------------------
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
         'beta': (12, 30), 'gamma': (30, 45)}

# Paths for ML models (modify if your filenames differ)
MODEL_PATH_256 = './results/baseline_models_256.pkl'        # full model (256-feat)
MODEL_PATH_8FEAT = './results/baseline_models_8feat.pkl' 
EMOTION_MODEL_PATH = './results/emotion_model_lstm.pth'

# ------------------- Helper: LSTM Emotion Model -------------------
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

# Try load emotion model if exists (non-fatal)
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

# ------------------- Encryption & MongoDB -------------------
# NOTE: KEY is generated per-run here. If you want decryptable DB fields across runs,
# use a persistent key instead of get_random_bytes on each startup.
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

# ------------------- EDF / Feature extraction helpers -------------------
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
    # bandpower per channel
    for ch in range(epoch.shape[0]):
        f, Pxx = welch(epoch[ch], sfreq, nperseg=min(256, epoch.shape[1]))
        total = np.trapz(Pxx, f) + 1e-12
        for lo, hi in BANDS.values():
            idx = np.logical_and(f >= lo, f <= hi)
            feats.append(np.trapz(Pxx[idx], f[idx]) / total)
    # hjorth + entropy per channel
    for ch in range(epoch.shape[0]):
        sig = epoch[ch]
        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)
        var0 = np.var(sig) + 1e-12
        var1 = np.var(diff1) + 1e-12
        var2 = np.var(diff2) + 1e-12
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else 0.0
        # entropy: reuse Pxx for last channel computation safely (recompute)
        f2, Pxx2 = welch(sig, sfreq, nperseg=min(256, len(sig)))
        P_norm = Pxx2 / (Pxx2.sum() + 1e-12)
        ent = -np.sum(P_norm * np.log2(P_norm + 1e-12))
        feats.extend([mobility, complexity, ent])
    return np.array(feats)

# ------------------- Emotion prediction helper -------------------
def emotion_predict(raw):
    if emotion_model is None:
        return "Model missing", "Emotion model not loaded.", np.array([0.0,0.0,0.0])
    # operate on a copy to avoid changing original raw elsewhere
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

# ------------------- Dream helpers -------------------
def dream_features(eeg, sf):
    mean, std = np.mean(eeg), np.std(eeg)
    # bandpower across channels 1-30Hz (average)
    bandpower_val = np.mean([np.trapz(welch(eeg[ch], sf, nperseg=min(len(eeg[ch]), max(256, sf*2)))[1]) for ch in range(eeg.shape[0])])
    return {"mean":mean, "std":std, "bandpower_1_30Hz":bandpower_val}

def dream_to_text(features, seed_value=0):
    np.random.seed(seed_value)
    adjectives = ["surreal", "luminous", "rustic", "vibrant", "silent"]
    adj = np.random.choice(adjectives)
    return (f"Once upon a dream... In a {adj} landscape where the air felt {features['mean']:.2f} "
            f"and the colors trembled with a variance of {features['std']:.2f}, a melody composed of "
            f"{features['bandpower_1_30Hz']:.2f} vibrations led the dreamer through shifting doors.")

# ------------------- Image / TTS helpers -------------------
PEXELS_API_KEY = "IwJRdJBvv9uKT7sBka8cbKIWAcA5y3n0T5pB224JpDDQnlfN37BpGR4O"  # set your key if needed
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
def fetch_image_for_scene(query, filename):
    if not PEXELS_API_KEY:
        return False
    headers = {'Authorization': PEXELS_API_KEY}
    params = {'query': query, 'per_page': 1}
    resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return False
    data = resp.json()
    if data.get('photos'):
        img_url = data['photos'][0]['src']['landscape']
        img_data = requests.get(img_url).content
        with open(filename, 'wb') as f:
            f.write(img_data)
        return True
    return False

def text_to_speech(text, filename):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
    except Exception as e:
        # if TTS fails, write a tiny silent mp3 to avoid crashing later
        open(filename, 'wb').close()

# ------------------- Model utilities -------------------
def load_ml_pack(path):
    """Load a joblib saved pack and return (rf, scaler, le, expected_feat_len)"""
    pack = joblib.load(path)
    # if pack is not a dict, try to interpret common wrappers
    if isinstance(pack, dict):
        rf = pack.get('rf') or pack.get('model') or pack.get('classifier')
        scaler = pack.get('scaler') or pack.get('sc') or None
        le = pack.get('le') or pack.get('label_encoder') or None
        expected_len = pack.get('feature_shape') if 'feature_shape' in pack else None
    else:
        # pack might be an sklearn Pipeline or estimator
        rf = pack
        scaler = getattr(pack, 'scaler_', None) or None
        le = None
        expected_len = None
        # If scaler exists as attribute, try to read n_features_in_
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
    # exact match based on feature count
    for pack, path in candidates:
        if pack['expected_len'] == n_features:
            return pack, path
    # pick closest expected_len
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

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="EEG Dream & Emotion Analysis", layout="wide")
st.title("🧠 EEG Dream & Emotion Analysis")

st.header("📝 Enter Your Info")
name = st.text_input("Full Name")
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male","Female","Other"])
user_info_complete = all([name, age, gender])

uploaded_file = st.file_uploader("Upload your EEG file (CSV or EDF)", type=["edf", "csv"])

if uploaded_file and user_info_complete:
    tmp_path = None
    try:
        tmp_bytes = uploaded_file.read()

        # Create proper temporary path based on file type
        if uploaded_file.name.lower().endswith('.csv'):
            st.info("Converting CSV to EDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                edf_filename = tmp_file.name
            csv_to_edf(tmp_bytes, edf_filename)
            tmp_path = edf_filename
        elif uploaded_file.name.lower().endswith('.edf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(tmp_bytes)
                tmp_path = tmp_file.name
        else:
            st.error("Unsupported file type. Please upload a .csv or .edf file.")
            st.stop()
            
        raw = load_eeg(tmp_path)
        # convert to microvolts like original code
        eeg_data = raw.get_data() * 1e6
        sfreq = int(raw.info['sfreq']) if 'sfreq' in raw.info else 256

        # quick single-channel PSD metrics via pyedflib for dream prompt (keeps original logic)
        try:
            f = pyedflib.EdfReader(tmp_path)
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
            # fallback if pyedflib fails
            delta = theta = alpha = beta = gamma = 0.0
            seed_value = 0

        # compute dream text BEFORE tabs so it's available later
        try:
            dream_feats = dream_features(eeg_data, sfreq)
            dream_text = dream_to_text(dream_feats, seed_value=seed_value)
        except Exception:
            dream_text = "Once upon a dream... (dream generation failed due to processing error.)"

        dream_prompt = (f"Based on the sleeper's brainwaves:\n"
                        f"- Delta={delta:.2f}, Theta={theta:.2f}, Alpha={alpha:.2f}, Beta={beta:.2f}, Gamma={gamma:.2f}\n"
                        f"With seed {seed_value}, write a dream as a short story. Begin with 'Once upon a dream...' and describe in detail:\n\n")

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
                features_array = np.vstack(feature_list)  # shape: (n_epochs, n_features)

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
                    # If scaler exists, try to get expected len from it if not provided
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

                    # Try to map labels if label encoder exists, otherwise show raw preds
                    if le is not None and hasattr(le, 'inverse_transform'):
                        try:
                            pred_labels = le.inverse_transform(preds)
                        except Exception:
                            pred_labels = [str(p) for p in preds]
                    else:
                        pred_labels = [str(p) for p in preds]

                    df_stages = pd.DataFrame({"Predicted Stage": pred_labels})

                    # Stage map for plotting
                    unique_labels = list(pd.unique(pred_labels))
                    stage_map = {lab: i for i, lab in enumerate(unique_labels)}

                    numeric_stages = df_stages["Predicted Stage"].map(stage_map).tolist()

                    # --- Clean single-line stage visualization ---
                    fig, ax = plt.subplots(figsize=(10, 3))
                    x_vals = list(range(len(numeric_stages)))
                    plot_y = numeric_stages.copy()
                    if len(plot_y) == 1:
                        # Duplicate the single point to make a visible line
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
            st.subheader("🎥 Dream Story & Video")
            st.write(dream_text)

            width, height = 1280, 720
            fps = 24
            duration_per_scene = 5
            frames_per_scene = duration_per_scene * fps
            video_dir = "dream_video_output"
            os.makedirs(video_dir, exist_ok=True)

            video_path = os.path.join(video_dir, "dream_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except Exception:
                font = ImageFont.load_default()

            scenes = [s.strip() for s in dream_text.split('.') if s.strip()]

            for i, scene_text in enumerate(scenes):
                st.write(f"Processing scene {i+1}/{len(scenes)}: {scene_text}")

                img_path = os.path.join(video_dir, f"scene_{i}.jpg")
                if not fetch_image_for_scene(scene_text, img_path):
                    base_img = Image.new("RGB", (width, height), (0, 0, 0))
                else:
                    try:
                        base_img = Image.open(img_path).convert("RGB").resize((width, height))
                    except Exception:
                        base_img = Image.new("RGB", (width, height), (0, 0, 0))

                draw = ImageDraw.Draw(base_img)
                bbox = draw.textbbox((0, 0), scene_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text(((width - text_w) // 2, height - text_h - 30),
                      scene_text, font=font, fill="white")

                frame = np.array(base_img)[:, :, ::-1]
                for _ in range(frames_per_scene):
                    video.write(frame)

                audio_file = os.path.join(video_dir, f"audio_{i}.mp3")
                text_to_speech(scene_text, audio_file)

            video.release()
            cv2.destroyAllWindows()

            if os.path.exists(video_path):
                st.success(f"✅ Dream video created and stored at: {video_path}")
                st.video(video_path)
            else:
                st.error("Video creation failed.")

        # Save to DB (probs may be numpy array)
        eeg_results = {"emotion":emotion,"emotion_prob":probs.tolist() if isinstance(probs, (np.ndarray, list)) else [0,0,0],"dream_text":dream_text}
        save_user_info_mongo({"name":name,"age":age,"gender":gender}, eeg_results)
        st.success("✅ All info and EEG analysis securely saved to MongoDB.")

    except Exception as e:
        st.exception(e)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
else:
    st.info("Fill all your info and upload an EEG EDF file to start analysis.")
