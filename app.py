from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import sqlite3
import hashlib
import pickle
import numpy as np
from scipy import stats as scipy_stats
import io
from PIL import Image
import cv2
import os

app = Flask(__name__)
CORS(app)
DB_NAME = "users.db"

with open("et_model.pkl", 'rb') as f:
    model_data = pickle.load(f)
et_model = model_data['model']
scaler   = model_data['scaler']

with open("all_epochs_final.pkl", 'rb') as f:
    all_epochs = pickle.load(f)

with open("stored_vgg_features.pkl", 'rb') as f:
    stored_data = pickle.load(f)
stored_vgg    = stored_data['vgg_features']
stored_labels = stored_data['labels']

vgg_model = VGG16(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)
print("VGG16 loaded!")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("Face detector loaded!")

scaler_vgg        = StandardScaler()
stored_vgg_scaled = scaler_vgg.fit_transform(stored_vgg)
knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn.fit(stored_vgg_scaled)
print("KNN ready!")

CH_NAMES = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8',
            'C4','F4','Fz','C3','F3','T7','F7','Oz','PO4','CP6',
            'CP2','CP1','CP5','PO3']

ALL_FEAT_NAMES = [
    'mean_all','var_all','std_all','skew_all','kurt_all',
    'mean_Pz','var_Pz','std_Pz','skew_Pz','kurt_Pz',
    'mean_Cz','var_Cz','std_Cz','skew_Cz','kurt_Cz',
    'mean_Fz','var_Fz','std_Fz','skew_Fz','kurt_Fz',
    'mean_P3','var_P3','std_P3','skew_P3','kurt_P3',
    'mean_P4','var_P4','std_P4','skew_P4','kurt_P4',
    'mean_Oz','var_Oz','std_Oz','skew_Oz','kurt_Oz',
    'delta_power_all','delta_power_Pz','delta_power_Cz',
    'delta_power_Fz','theta_power_all','theta_power_Pz',
    'theta_power_Cz','theta_power_Fz','alpha_power_all',
    'alpha_power_Pz','alpha_power_Cz','alpha_power_Fz',
    'beta_power_all','beta_power_Pz','beta_power_Cz',
    'beta_power_Fz','gamma_power_all','gamma_power_Pz',
    'gamma_power_Cz','gamma_power_Fz',
    'p300_amp_Pz','p300_mean_Pz','p300_latency_Pz',
    'p300_amp_Cz','p300_mean_Cz',
    'n200_amp_Fz','n200_mean_Fz',
    'corr_Pz_Cz','corr_Pz_P3','corr_Pz_P4',
    'corr_Fz_Cz','corr_Cz_Oz'
]

SIG_FEAT_NAMES = ['var_P3','skew_P3','mean_P3',
                  'var_all','std_P3','delta_power_all']
SIG_INDICES    = [ALL_FEAT_NAMES.index(f) for f in SIG_FEAT_NAMES]


def extract_all_eeg_features(epoch_data):
    times_local = np.linspace(-200, 800, 500)
    features    = []
    features.append(epoch_data.mean())
    features.append(epoch_data.var())
    features.append(epoch_data.std())
    features.append(scipy_stats.skew(epoch_data.flatten()))
    features.append(scipy_stats.kurtosis(epoch_data.flatten()))

    for ch in ['Pz','Cz','Fz','P3','P4','Oz']:
        if ch in CH_NAMES:
            idx = CH_NAMES.index(ch)
            sig = epoch_data[idx]
            features.append(sig.mean())
            features.append(sig.var())
            features.append(sig.std())
            features.append(scipy_stats.skew(sig))
            features.append(scipy_stats.kurtosis(sig))

    freqs   = np.fft.rfftfreq(epoch_data.shape[-1], d=1.0/500.0)
    fft_val = np.abs(np.fft.rfft(epoch_data, axis=-1))**2
    bands   = [(1,4),(4,8),(8,13),(13,30),(30,40)]

    for fmin, fmax in bands:
        idx = (freqs >= fmin) & (freqs <= fmax)
        features.append(fft_val[:, idx].mean())
        for ch in ['Pz','Cz','Fz']:
            if ch in CH_NAMES:
                ci = CH_NAMES.index(ch)
                features.append(fft_val[ci, idx].mean())

    p300_win = (times_local >= 250) & (times_local <= 500)
    n200_win = (times_local >= 150) & (times_local <= 250)

    for ch in ['Pz','Cz']:
        if ch in CH_NAMES:
            idx = CH_NAMES.index(ch)
            sig = epoch_data[idx]
            features.append(sig[p300_win].max())
            features.append(sig[p300_win].mean())
            if ch == 'Pz':
                pk = np.argmax(sig[p300_win])
                features.append(times_local[p300_win][pk])

    if 'Fz' in CH_NAMES:
        fz = epoch_data[CH_NAMES.index('Fz')]
        features.append(fz[n200_win].min())
        features.append(fz[n200_win].mean())

    pairs = [('Pz','Cz'),('Pz','P3'),('Pz','P4'),
             ('Fz','Cz'),('Cz','Oz')]
    for ch1, ch2 in pairs:
        if ch1 in CH_NAMES and ch2 in CH_NAMES:
            s1   = epoch_data[CH_NAMES.index(ch1)]
            s2   = epoch_data[CH_NAMES.index(ch2)]
            corr, _ = scipy_stats.pearsonr(s1, s2)
            features.append(corr)

    return np.array(features)


def get_db():
    return sqlite3.connect(DB_NAME)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_table():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    db.commit()
    db.close()

create_table()


@app.route("/register", methods=["POST"])
def register():
    data     = request.json
    name     = data["name"]
    email    = data["email"]
    password = hash_password(data["password"])
    db       = get_db()
    cursor   = db.cursor()
    cursor.execute("SELECT 1 FROM users WHERE email=?", (email,))
    if cursor.fetchone():
        return jsonify({"error": "Email already registered"}), 400
    cursor.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email, password))
    db.commit()
    db.close()
    return jsonify({"message": "Registration successful"})


@app.route("/login", methods=["POST"])
def login():
    data     = request.json
    email    = data["email"]
    password = hash_password(data["password"])
    db       = get_db()
    cursor   = db.cursor()
    cursor.execute("SELECT password FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    db.close()
    if not user:
        return jsonify({"error": "Invalid email"}), 400
    if user[0] != password:
        return jsonify({"error": "Invalid password"}), 400
    return jsonify({"message": "Login successful"})


@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files['image'].read()

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    # ===== Face Detection — STRICT mode =====
    face_found = False
    try:
        cv_img = np.array(pil_img)
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # Pehle strict try karo
        faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,   # was 1.1 — zyada sensitive
                    minNeighbors=4,     # was 3
                    minSize=(60, 60)    # choti faces ignore karo
                )
        
        if len(faces) > 0:
            # Sabse bara face lo
            faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            # Thoda padding do face ke around
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(pil_img.width,  x + w + pad)
            y2 = min(pil_img.height, y + h + pad)
            pil_img    = pil_img.crop((x1, y1, x2, y2))
            face_found = True
        else:
            # ===== Face nahi mili — reject karo =====
            return jsonify({"error": "No face detected"}), 200

    except Exception as e:
        print(f"Face detection error: {e}")
        return jsonify({"error": "No face detected"}), 200

    # ===== VGG16 features =====
    img     = pil_img.resize((224, 224))
    img_arr = keras_image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    new_vgg = vgg_model.predict(img_arr, verbose=0).flatten()

    # ===== KNN matching =====
    new_vgg_scaled     = scaler_vgg.transform(new_vgg.reshape(1, -1))
    distances, indices = knn.kneighbors(new_vgg_scaled)

    # ===== EEG + ET classification =====
    predictions = []
    probas      = []

    for idx in indices[0]:
        epoch_data = all_epochs[idx]['epoch_data']
        all_feats  = extract_all_eeg_features(epoch_data)
        feat_vec   = np.array([all_feats[i] for i in SIG_INDICES])
        X_scale    = scaler.transform(feat_vec.reshape(1, -1))
        pred       = et_model.predict(X_scale)[0]
        proba      = et_model.predict_proba(X_scale)[0]
        predictions.append(int(pred))
        probas.append(proba.tolist())

    # ===== Weighted voting (distance-based) =====
    probas_arr = np.array(probas)
    
    # Closer neighbors ko zyada weight do
    weights = 1.0 / (distances[0] + 1e-6)
    weights = weights / weights.sum()
    avg_proba  = np.average(probas_arr, axis=0, weights=weights)
    
    final_pred = int(np.argmax(avg_proba))
    pred_str   = "REAL" if final_pred == 1 else "FAKE"
    real_conf  = float(avg_proba[1])
    fake_conf  = float(avg_proba[0])

    return jsonify({
        "prediction": pred_str,
        "confidence": round(float(avg_proba[final_pred]), 2),
        "real_prob":  round(real_conf, 2),
        "fake_prob":  round(fake_conf, 2)
    })


@app.route("/")
def home():
    return "Backend running successfully"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
