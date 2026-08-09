import os
import json
import subprocess
from collections import Counter
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    session,
    jsonify,
)
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
from pathlib import Path
import MySQLdb.cursors
import re
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from ultralytics import YOLO  # (you had this; kept as-is)
import uuid
import joblib
import numpy as np
import librosa

# ============================================================
# PATHS & CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
AUDIO_UPLOAD_DIR = BASE_DIR / "uploads"     # for audio uploads
TMP_DIR = BASE_DIR / "tmp"                  # not strictly needed, but available
MODEL_DIR = BASE_DIR / "models"

for d in (AUDIO_UPLOAD_DIR, TMP_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# AUDIO model path (this is the model you trained in Jupyter)
VOICE_MODEL_PATH = MODEL_DIR / "voice_classifier_model.joblib"

# Allowed audio types
AUDIO_ALLOWED = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"}

# ============================================================
# Initialize Flask
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# This UPLOAD_FOLDER is used for VIDEO
VIDEO_UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = VIDEO_UPLOAD_FOLDER

load_dotenv()  # Load .env

# ============================================================
# Database Config
# ============================================================
app.config["MYSQL_HOST"] = os.getenv("MYSQL_HOST", "localhost")
app.config["MYSQL_USER"] = os.getenv("MYSQL_USER", "root")
app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "root")
app.config["MYSQL_DB"] = os.getenv("MYSQL_DB", "fake")

app.secret_key = os.getenv("SECRET_KEY", "replace-this-with-a-secret-key")
mysql = MySQL(app)

# ============================================================
# VIDEO / ASD MODEL CONFIG
# ============================================================
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

VIDEO_MODEL_PATH = "autism_classifier_best.pth"
CLASS_MAP_PATH = "class_mapping.json"
FRAME_INTERVAL = 30        # sample every Nth frame
FRAME_SIZE = (224, 224)    # input size for ResNet

# Load class mapping
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError(
        f"{CLASS_MAP_PATH} not found – make sure it matches your trained model."
    )

with open(CLASS_MAP_PATH, "r") as f:
    CLASS_TO_IDX = json.load(f)   # e.g. {"ASD": 0, "NON_ASD": 1}

IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
print("✅ Loaded VIDEO classes:", CLASS_NAMES)

# Load video model (ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_model = models.resnet18(weights="IMAGENET1K_V1")
video_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(video_model.fc.in_features, len(CLASS_NAMES)),
)

if not os.path.exists(VIDEO_MODEL_PATH):
    raise FileNotFoundError(f"Video model file {VIDEO_MODEL_PATH} not found")

state_dict = torch.load(VIDEO_MODEL_PATH, map_location=device)
video_model.load_state_dict(state_dict)
video_model = video_model.to(device)
video_model.eval()

video_transform = transforms.Compose(
    [
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp4(input_path: str) -> str:
    """
    Convert uploaded video to mp4 using ffmpeg.
    Returns path to the mp4 file (in VIDEO_UPLOAD_FOLDER).
    """
    filename_root, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename_root + ".mp4")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    return output_path

def predict_video(video_path: str) -> dict:
    """
    Sample every FRAME_INTERVAL-th frame, run through video_model,
    majority vote across frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Could not open video."}

    preds = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = video_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = video_model(img)
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.item())

        frame_count += 1

    cap.release()

    if not preds:
        return {"ok": False, "error": "No frames processed."}

    counts_idx = Counter(preds)
    total = sum(counts_idx.values())
    pred_idx = counts_idx.most_common(1)[0][0]

    counts_by_name = {IDX_TO_CLASS[idx]: c for idx, c in counts_idx.items()}
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = round(counts_idx[pred_idx] / total, 3)

    return {
        "ok": True,
        "prediction": pred_label,
        "counts": counts_by_name,
        "confidence": confidence,
    }

# ============================================================
# AUDIO MODEL (Mel-spectrogram pipeline like your test code)
# ============================================================
if VOICE_MODEL_PATH.exists():
    audio_bundle = joblib.load(VOICE_MODEL_PATH)
    audio_model = audio_bundle["model"]
    audio_scaler = audio_bundle["scaler"]
    AUDIO_IDX_TO_CLASS = audio_bundle["idx_to_class"]
    print("[AUDIO INFO] Loaded voice model from", VOICE_MODEL_PATH)
else:
    audio_bundle = None
    audio_model = None
    audio_scaler = None
    AUDIO_IDX_TO_CLASS = {}
    print("[AUDIO WARN] voice_classifier_model.joblib not found at", VOICE_MODEL_PATH)

# same settings as your test code
SR = 16000
N_MELS = 64
DURATION = 3.0
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 128

def load_wav_fixed(path: str):
    y, _ = librosa.load(path, sr=SR, mono=True)
    length = int(SR * DURATION)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    else:
        y = y[:length]
    return y

def wav_to_mel(y: np.ndarray):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    return mel_db.astype(np.float32)

def predict_from_any_audio(in_path: Path):
    """
    EXACT same logic as your Jupyter audio test:
      - load_wav_fixed
      - wav_to_mel
      - flatten
      - scaler.transform
      - model.predict
    """
    if audio_model is None or audio_scaler is None or not AUDIO_IDX_TO_CLASS:
        raise RuntimeError("Audio model is not loaded.")

    print(f"[AUDIO DEBUG] Input path: {in_path}")

    # librosa can load many formats (wav, mp3, etc.) directly
    y = load_wav_fixed(str(in_path))
    mel = wav_to_mel(y)
    feat = mel.flatten().reshape(1, -1)

    print(
        f"[AUDIO DEBUG] feat shape: {feat.shape}, "
        f"min={float(feat.min()):.4f}, max={float(feat.max()):.4f}, mean={float(feat.mean()):.4f}"
    )

    feat_scaled = audio_scaler.transform(feat)
    pred_idx = audio_model.predict(feat_scaled)[0]
    pred_label = AUDIO_IDX_TO_CLASS[pred_idx]

    print(f"[AUDIO DEBUG] pred_idx={pred_idx}, pred_label={pred_label}")

    # If you want duration:
    duration_sec = float(librosa.get_duration(y=y, sr=SR))

    # Optional: if audio_model supports predict_proba, we can give probabilities
    prob = None
    if hasattr(audio_model, "predict_proba"):
        prob_vec = audio_model.predict_proba(feat_scaled)[0]
        prob = {str(cls): float(p) for cls, p in zip(audio_model.classes_, prob_vec)}
        print("[AUDIO DEBUG] predict_proba:", prob)

    return {
        "label": pred_label,
        "duration_sec": duration_sec,
        "proba_raw": prob,
    }

# ============================================================
# AUTH / ACCOUNT ROUTES
# ============================================================
@app.route("/", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        age = request.form.get("age", "").strip()

        # Basic validations, including age
        try:
            age_int = int(age)
        except (TypeError, ValueError):
            age_int = None

        if not username or not password or not age:
            msg = "Please fill out the form!"
        elif not re.match(r"^[A-Za-z0-9]+$", username):
            msg = "Invalid username!"
        elif age_int is None:
            msg = "Please select a valid age!"
        elif age_int > 10:
            msg = "This system supports only below 10 years."
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                "SELECT * FROM accounts WHERE username = %s AND password = %s",
                (username, password),
            )
            account = cursor.fetchone()

            if account:
                session["loggedin"] = True
                session["id"] = account["id"]
                session["username"] = account["username"]
                return render_template("index.html")
            else:
                msg = "Incorrect username/password!"

    return render_template("login.html", msg=msg)


@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ""
    if (
        request.method == "POST"
        and "username" in request.form
        and "password" in request.form
        and "email" in request.form
    ):
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        email = request.form["email"].strip()

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM accounts WHERE username = %s", (username,))
        account = cursor.fetchone()

        if account:
            msg = "Account already exists!"
        elif not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            msg = "Invalid email address!"
        elif not re.match(r"^[A-Za-z0-9]+$", username):
            msg = "Username must contain only characters and numbers!"
        elif len(password) < 8:
            msg = "Password must be at least 8 characters long!"
        elif not username or not password or not email:
            msg = "Please fill out the form!"
        else:
            cursor.execute(
                "INSERT INTO accounts (username, password, email) VALUES (%s, %s, %s)",
                (username, password, email),
            )
            mysql.connection.commit()
            msg = "You have successfully registered!"
    elif request.method == "POST":
        msg = "Please fill out the form!"

    return render_template("register.html", msg=msg)


@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    msg = ""
    if (
        request.method == "POST"
        and "username" in request.form
        and "newpassword" in request.form
        and "confirmpassword" in request.form
    ):
        username = request.form["username"]
        new_password = request.form["newpassword"]
        confirm_password = request.form["confirmpassword"]

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM accounts WHERE username = %s", (username,))
        account = cursor.fetchone()

        if not account:
            msg = "Account does not exist!"
        elif new_password != confirm_password:
            msg = "Passwords do not match!"
        else:
            cursor.execute(
                "UPDATE accounts SET password = %s WHERE username = %s",
                (new_password, username),
            )
            mysql.connection.commit()
            msg = "Password successfully updated!"
    elif request.method == "POST":
        msg = "Please fill out the form!"

    return render_template("reset_password.html", msg=msg)


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

# ============================================================
# VIDEO ROUTES
# ============================================================
@app.route("/asdvedio", methods=["GET"])
def asdvedio():
    return render_template(
        "asdvedio.html",
        prediction_result=None,
        video_url=None,
        total_frames=None,
        autism_level=None,
        therapy_suggestions=None,
        raw_class=None,
        counts=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        flash("No file selected")
        return redirect(url_for("asdvedio"))

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("asdvedio"))

    if not (file and allowed_video_file(file.filename)):
        flash("Unsupported file type")
        return redirect(url_for("asdvedio"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    final_path = convert_to_mp4(save_path)
    final_filename = os.path.basename(final_path)

    result = predict_video(final_path)

    if not result.get("ok", False):
        flash(result.get("error", "Prediction failed"))
        return render_template(
            "asdvedio.html",
            prediction_result=None,
            video_url=url_for("uploaded_file", filename=final_filename),
        )

    raw_class = result.get("prediction")
    raw_counts = result.get("counts") or {}

    counts = {str(k): int(v) for k, v in raw_counts.items()}
    total_frames = sum(counts.values()) if counts else 0
    asd_frames = counts.get("ASD", 0)
    non_asd_frames = counts.get("NON_ASD", 0)

    if raw_class == "ASD":
        prediction_result = "Autistic"
    elif raw_class == "NON_ASD":
        prediction_result = "Non-Autistic"
    else:
        prediction_result = "Unknown"

    autism_level = "Unknown"
    therapy_suggestions = []

    if prediction_result == "Autistic" and total_frames > 0:
        asd_ratio = asd_frames / total_frames

        if asd_ratio >= 0.66:
            autism_level = "High"
        elif asd_ratio >= 0.33:
            autism_level = "Medium"
        else:
            autism_level = "Low"

        therapy_suggestions = [
            "Book a detailed evaluation with a qualified child psychiatrist, developmental pediatrician, or clinical psychologist.",
            "Start early intervention as soon as possible (speech therapy, occupational therapy, and behavioral programs such as ABA or other evidence-based approaches).",
            "Work with a speech-language therapist to support communication, including visual schedules, picture exchange, or other AAC tools if speech is delayed.",
            "Consult an occupational therapist to address sensory issues, fine-motor skills, and daily living activities like dressing, feeding, and toileting.",
            "Use structured routines and visual supports at home (visual timetables, clear step-by-step instructions, and consistent daily schedules).",
            "Encourage play-based social interaction with trusted adults and peers in calm, low-pressure settings.",
            "Attend parent-training or caregiver-training programs so you can apply therapeutic strategies during everyday routines.",
            "Remember: this tool is only a screening aid and cannot replace a formal diagnosis.",
        ]
    elif prediction_result == "Non-Autistic":
        autism_level = "None"
        therapy_suggestions = []
    else:
        autism_level = "Unknown"
        therapy_suggestions = [
            "Please consult a qualified clinician for an in-depth evaluation.",
            "Use this tool only as a screening aid, not as a medical diagnosis.",
        ]

    return render_template(
        "asdvedio.html",
        video_url=url_for("uploaded_file", filename=final_filename),
        prediction_result=prediction_result,
        total_frames=total_frames,
        autism_level=autism_level,
        therapy_suggestions=therapy_suggestions,
        raw_class=raw_class,
        counts=counts,
    )

# ============================================================
# AUDIO ROUTES
# ============================================================
@app.route("/audio", methods=["GET"])
def audio():
    return render_template("audio.html", model_loaded=(audio_model is not None))


@app.route("/api/audio_predict", methods=["POST"])
def api_audio_predict():
    if audio_model is None:
        return jsonify({"ok": False, "error": "Audio model not loaded."}), 500

    file = request.files.get("audio_file")
    if file is None:
        return jsonify({"ok": False, "error": "No file part in request."}), 400
    if file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in AUDIO_ALLOWED:
        return jsonify({"ok": False, "error": f"Unsupported file type: {ext}"}), 400

    try:
        AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = AUDIO_UPLOAD_DIR / f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file.save(str(saved_path))
        print(f"[AUDIO DEBUG] Saved audio to {saved_path}")
    except Exception as e:
        print("[ERROR] Failed to save audio file:", e)
        return jsonify({"ok": False, "error": "Failed to save audio file."}), 500

    try:
        result = predict_from_any_audio(saved_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[ERROR] Prediction crashed:", e)
        return jsonify({"ok": False, "error": "Prediction crashed on server."}), 500

    return jsonify({"ok": True, **result})

# Debug ping
@app.route("/_debug_ping")
def _debug_ping():
    if audio_model is None:
        return {"audio_model": "missing"}
    return {"classes_": getattr(audio_model, "classes_", []).tolist()}

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


class UploadImageForm(FlaskForm):
    file = FileField("Upload Image", validators=[InputRequired()])
    submit = SubmitField("Run Detection")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("MYSQL_HOST =", app.config.get("MYSQL_HOST"))
    app.run(debug=True, port=5000)
