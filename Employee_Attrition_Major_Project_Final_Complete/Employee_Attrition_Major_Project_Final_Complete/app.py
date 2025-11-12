# app.py (paste/replace existing)
# ---------- imports (cleaned) ----------
import os
import json
import shutil
import subprocess
import traceback
from tempfile import NamedTemporaryFile
from datetime import datetime
import re

import joblib
import pandas as pd
import sqlite3
import soundfile as sf

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

# STT / Vosk
from vosk import Model, KaldiRecognizer

# Sentiment (transformers pipeline)
from transformers import pipeline

# Env / OpenAI
import base64
from dotenv import load_dotenv
from openai import OpenAI
from flask_sqlalchemy import SQLAlchemy
from flask import send_from_directory
# near other imports
from pyannote.audio import Pipeline

# ---------------------------------------#
#huggingface token#
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")

# ---------------- helper ----------------
def json_error(msg, code=500):
    """Return consistent JSON error response."""
    return jsonify({'error': str(msg)}), code
# ----------------------------------------

# ---------- Voice / OpenAI setup ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing. Add it in your .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)
# ------------------------------------------

# ================== FLASK APP SETUP ==================
app = Flask(__name__)
app.secret_key = 'replace-with-secure-key'  # change for production

#-------dbphase2--------#
# --- DB setup (add after app = Flask(__name__)) ---


# configure sqlite in current project folder
DB_FILENAME = os.environ.get("KEEPWISE_DB", "keepwise_records.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_FILENAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
diarization_pipeline = None##
class MeetingRecord(db.Model):
    __tablename__ = 'meeting_records'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    transcript = db.Column(db.Text, nullable=True)
    readable_fields = db.Column(db.Text, nullable=True)   # JSON string
    sentiment = db.Column(db.Text, nullable=True)         # JSON string
    diarization = db.Column(db.Text, nullable=True)       # JSON string (per speaker segments)
    audio_path = db.Column(db.Text, nullable=True)        # path on disk
    raw_response = db.Column(db.Text, nullable=True)      # full raw response from analyzer (optional)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "transcript": self.transcript,
            "readable_fields": json.loads(self.readable_fields) if self.readable_fields else {},
            "sentiment": json.loads(self.sentiment) if self.sentiment else {},
            "diarization": json.loads(self.diarization) if self.diarization else {},
            "audio_path": self.audio_path,
            "raw_response": json.loads(self.raw_response) if self.raw_response else {}
        }


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join('models', 'model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠️ Model file not found at: {MODEL_PATH}\nPlease place 'model.pkl' inside the models folder.")
model = joblib.load(MODEL_PATH)

DB = os.path.join('database', 'predictions.db')
os.makedirs('database', exist_ok=True)

# Vosk model path (update if different)
VOSK_MODEL_DIR = os.path.join("models", "vosk-model-small-en-us-0.15")
if not os.path.isdir(VOSK_MODEL_DIR):
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_DIR}. Download and unpack it there.")
vosk_model = Model(VOSK_MODEL_DIR)

# Sentiment pipeline (this may download weights on first run)
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load trained model
model = joblib.load(R"C:\Users\hp\OneDrive\Documents\Keepwise\Employee_Attrition_Major_Project_Final_Complete\Employee_Attrition_Major_Project_Final_Complete\models\model.pkl")

# ✅ Debug print to confirm model expects the correct feature names
print("MODEL features:", getattr(model, 'feature_names_in_', None))



# ================== DATABASE INIT ==================
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_json TEXT,
        probability REAL,
        prediction INTEGER
    )''')
    conn.commit()
    conn.close()


# ================== ROUTES ==================
@app.route('/')
def index():
    return redirect(url_for('dashboard'))




# ========== SINGLE PREDICTION ==========

@app.route('/predict', methods=['POST'])
def predict():
    # build dict from form (flat)
    form = request.form.to_dict()
    print("PREDICT: raw form keys:", list(form.keys()))

    # create DataFrame from form keys (initial)
    df = pd.DataFrame([form])
    print("PREDICT: df columns (initial):", df.columns.tolist())

    # --- Helpful defaults for the new fields (tweak if you want different defaults) ---
    # Note: these keys are the canonical column names your trained pipeline expects.
    default_values = {
        'JobRole': 'Research Scientist',
        'JobSatisfaction': 3,
        'TotalWorkingYears': 0,
        'YearsAtCompany': 0,
        'EnvironmentSatisfaction': 3,
        'WorkLifeBalance': 3,
        'PerformanceRating': 3
    }

    # --- If model exposes expected feature names, use them to align input ---
    expected_cols = getattr(model, 'feature_names_in_', None)

    # Build a lower->original mapping of incoming form keys for case-insensitive matching
    incoming_key_map = {k.lower(): k for k in form.keys()}

    if expected_cols is not None:
        # Ensure every expected column is present in df (copy or map from incoming keys)
        for col in expected_cols:
            if col not in df.columns:
                # try to find a case-insensitive match in the incoming form
                lower = col.lower()
                if lower in incoming_key_map:
                    df[col] = form[incoming_key_map[lower]]
                elif col in default_values:
                    df[col] = default_values[col]
                else:
                    # fallback to 0 or empty string depending on guessed type later
                    df[col] = 0

        # reorder / keep only expected columns
        df = df[list(expected_cols)]
    else:
        # If model doesn't have feature_names_in_, at least make sure our new fields exist so model won't choke
        for col, val in default_values.items():
            # try to map by case-insensitive key from incoming form
            if col not in df.columns:
                lower = col.lower()
                if lower in incoming_key_map:
                    df[col] = form[incoming_key_map[lower]]
                else:
                    df[col] = val
        # keep all form columns plus defaults (this is safer if you have an older model)
        # optionally, you could reorder columns to a canonical set here if you know it.

    # --- Convert appropriate columns to numeric where applicable ---
    numeric_cols = [
        'Age', 'MonthlyIncome', 'JobSatisfaction', 'TotalWorkingYears', 'YearsAtCompany',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'PerformanceRating'
    ]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0).astype(float)

    # DEBUG: final dataframe that will be passed to model
    print("PREDICT: final df columns (ordered):", df.columns.tolist())
    print("PREDICT: final df row values:", df.iloc[0].to_dict())

    # --- Prediction ---
    try:
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(df)[0][1])
        else:
            # if model only supports predict, treat predicted label as probability-like (0/1)
            prob = float(model.predict(df)[0])
    except Exception as e:
        print("PREDICT: model prediction failed:", e)
        flash('Model prediction failed. Check server logs for details.', 'error')
        return redirect(url_for('index'))

    pred = int(prob >= 0.5)
    text = '⚠️ Employee is likely to leave' if pred == 1 else '✅ Employee is likely to stay'

    # --- store in sqlite DB for history (JSON-safe) ---
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO predictions (timestamp, input_json, probability, prediction) VALUES (?, ?, ?, ?)',
            (datetime.utcnow().isoformat(), json.dumps(form), float(prob), int(pred))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("PREDICT: DB save failed:", e)

    return render_template('result.html', prediction_text=text, probability=f"{prob:.3f}")

# ========== BULK PREDICTION ==========
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    f = request.files['file']
    filename = secure_filename(f.filename)
    if filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    if filename.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    expected_cols = getattr(model, 'feature_names_in_', None)
    if expected_cols is not None:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    for i, row in df.iterrows():
        cur.execute(
            'INSERT INTO predictions (timestamp, input_json, probability, prediction) VALUES (?, ?, ?, ?)',
            (datetime.utcnow().isoformat(), str(row.to_dict()), float(probs[i]), int(preds[i]))
        )
    conn.commit()
    conn.close()

    flash(f'Successfully processed {len(df)} employee records.', 'info')
    return redirect(url_for('index'))


# ========== ADMIN PANEL ==========
@app.route('/admin')
def admin():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC LIMIT 200', conn)
    conn.close()
    return render_template('admin.html', rows=df.to_dict(orient='records'))


# ======= Records listing + audio serving ======new=


@app.route('/uploads/<path:filename>')
def uploads(filename):
    # Serve files from uploads folder
    return send_from_directory(os.path.join(os.getcwd(), 'uploads'), filename, as_attachment=False)

@app.route('/records')
def records_list():
    # Query recent meeting records (most recent first)
    try:
        rows = MeetingRecord.query.order_by(MeetingRecord.id.desc()).limit(200).all()
    except Exception as e:
        print("RECORDS: DB query failed:", e)
        rows = []

    # build simple serializable list with audio URL (basename)
    recs = []
    for r in rows:
        audio_fn = None
        if r.audio_path:
            audio_fn = os.path.basename(r.audio_path)
        recs.append({
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "transcript": (r.transcript or "")[:1000],
            "readable_fields": json.loads(r.readable_fields) if r.readable_fields else {},
            "sentiment": json.loads(r.sentiment) if r.sentiment else {},
            "audio_filename": audio_fn,
            "audio_url": url_for('uploads', filename=audio_fn) if audio_fn else None,
            "raw_response": json.loads(r.raw_response) if r.raw_response else {}
        })
    return render_template('records.html', records=recs)

#------------Dashboard PAGE---------#
@app.route('/dashboard')
def dashboard():
    # pass a callable so template can call now()
    return render_template('dashboard.html', now=datetime.utcnow)

# add these to app.py (paste below your dashboard route)


@app.route('/single')
def single():
    # renders the single prediction page (make sure template file exists)
    return render_template('single_prediction.html')

@app.route('/recorder')
def recorder():
    # renders the audio recorder page
    return render_template('audio_recorder.html')

@app.route('/bulk')
def bulk():
    # renders the bulk prediction page (or analytics if you prefer)
    return render_template('bulk_prediction.html')



# ========== Analytics PAGE ==========
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')




# ========== VOICE FRONTEND ==========
@app.route('/voice')
def voice_page():
    return render_template('voice.html')


# ---------------- SAFE /transcribe route ----------------
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return json_error('No audio file uploaded', 400)

        audio_file = request.files['audio']

        # save incoming file to a temp path
        tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1] or ".webm")
        tmp_path = tmp.name
        tmp.close()
        audio_file.save(tmp_path)

        wav_path = tmp_path + ".wav"
        ff = shutil.which("ffmpeg") or "ffmpeg"
        try:
            subprocess.run([
                ff, "-y",
                "-i", tmp_path,
                "-ar", "16000",
                "-ac", "1",
                wav_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            # cleanup
            try: os.remove(tmp_path)
            except: pass
            return json_error('ffmpeg conversion failed. Is ffmpeg installed and accessible? ' + str(e), 500)

        data, samplerate = sf.read(wav_path, dtype="int16")
        if 'vosk_model' not in globals() or vosk_model is None:
            return json_error("Vosk model not loaded on server (vosk_model is missing).", 500)

        rec = KaldiRecognizer(vosk_model, samplerate)
        rec.AcceptWaveform(data.tobytes())
        result_text = rec.FinalResult()
        try:
            result = json.loads(result_text)
        except Exception:
            result = {}
        text = result.get('text', '')

        return jsonify({'transcript': text})
    except Exception as e:
        traceback.print_exc()
        return json_error("Server exception in /transcribe: " + str(e), 500)
    finally:
        try:
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if 'wav_path' in locals() and wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


# ---------------- SAFE /voice_predict route (LLM mapping) ----------------
@app.route('/voice_predict', methods=['POST'])
def voice_predict():
    try:
        data = request.get_json(force=True)
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            return json_error('No transcript provided', 400)

        if 'client' not in globals() or client is None:
            return json_error('OpenAI client not configured on server (cannot parse transcript).', 503)

        expected_cols = getattr(model, 'feature_names_in_', None)
        if expected_cols is None:
            return json_error('Model does not expose feature_names_in_. Cannot auto-map.', 500)

        feature_list = ", ".join(expected_cols)
        system_prompt = (
            "You are a strict JSON generator. Receive a short natural-language transcript "
            "containing an employee's details (age, department, business travel, monthly income, overtime etc.). "
            "Return ONLY a valid JSON object with keys exactly matching this list (in any order): "
            f"{feature_list}. Use sensible default values if a field is missing. Types: numbers for numeric fields, "
            "strings for categorical. Do NOT include any extra keys or commentary."
        )
        user_prompt = f"Transcript: \"{transcript}\""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400
        )

        assistant_text = None
        try:
            assistant_text = completion.choices[0].message.content.strip()
        except Exception:
            try:
                assistant_text = completion['choices'][0]['message']['content'].strip()
            except Exception:
                assistant_text = str(completion)

        parsed = None
        try:
            parsed = json.loads(assistant_text)
        except Exception:
            start = assistant_text.find('{')
            end = assistant_text.rfind('}') + 1
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(assistant_text[start:end])
                except Exception as e:
                    return jsonify({'error': 'Failed to parse JSON from assistant output', 'assistant_output': assistant_text}), 500
            else:
                return jsonify({'error': 'Assistant did not return JSON', 'assistant_output': assistant_text}), 500

        # ✅ Add missing columns if the assistant missed some
        for col in expected_cols:
            if col not in parsed:
                parsed[col] = 0

        import pandas as _pd
        df_row = _pd.DataFrame([parsed])

        ### 🟩 NEW DEBUG SNIPPETS (step 4B)
        print("VOICE_PREDICT: assistant parsed:", parsed)
        print("VOICE_PREDICT: expected_cols:", expected_cols)
        print("VOICE_PREDICT: df_row columns before cast:", list(df_row.columns))

        ### 🟩 NEW FEATURE FIX SNIPPETS (step 5)
        # Ensure all expected features exist and align with model input
        for col in expected_cols:
            if col not in df_row.columns:
                df_row[col] = 0  # fill any missing columns with 0

        # align column order to match model’s expectation
        df_row = df_row[list(expected_cols)]

        # try to cast numeric columns
        df_row = df_row.apply(_pd.to_numeric, errors='ignore')

        print("VOICE_PREDICT: final df_row columns (ordered):", df_row.columns.tolist())

        # ---- Prediction ----
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(df_row)[0][1])
        else:
            prob = float(model.predict(df_row)[0])

        pred = int(prob >= 0.5)
        text = '⚠️ Employee likely to leave' if pred == 1 else '✅ Employee likely to stay'

        return jsonify({
            'prediction': pred,
            'probability': prob,
            'text': text,
            'assistant_json': parsed,
            'assistant_text': assistant_text
        })

    except Exception as e:
        traceback.print_exc()
        return json_error("Server exception in /voice_predict: " + str(e), 500)


# --------- helper logic --------- #
KEYWORD_RISK = {
    "leave": 1.0, "resign": 1.0, "looking for": 0.9, "quit": 1.0, "find another job": 1.0,
    "not happy": 0.8, "unhappy": 0.8, "no growth": 0.9, "overworked": 0.8, "burnout": 1.0,
    "salary": 0.3, "raise": 0.4, "pay": 0.3, "underpaid": 0.8, "manager problem": 0.7
}


def compute_risk_score(text, sentiment_result):
    text_l = (text or "").lower()
    score = 0.0
    for k, w in KEYWORD_RISK.items():
        if k in text_l:
            score += w
    if sentiment_result and isinstance(sentiment_result, list):
        s = sentiment_result[0]
        if s.get('label', '').upper().startswith('NEG'):
            score += s.get('score', 0) * 0.8
        else:
            score -= s.get('score', 0) * 0.4
    max_possible = sum(KEYWORD_RISK.values()) + 0.8
    risk = min(1.0, max(0.0, score / max_possible))
    return risk


# -------- meeting page & upload handler -------- #
@app.route('/meeting', methods=['GET'])
def meeting_page():
    return render_template('meeting.html')

#-------MEETING/UPLOAD----------------------#
@app.route("/meeting_upload", methods=["POST", "OPTIONS"], strict_slashes=False)
def meeting_upload():
    """
    Browser uploads recorded clip (FormData 'audio').
    Convert -> WAV, transcribe with Vosk, sentiment + heuristic mapping,
    compute risk score, save DB record and return JSON.
    """
    try:
        if request.method == "OPTIONS":
            return ("", 200)

        print("MEETING_UPLOAD: request method:", request.method, "files:", list(request.files.keys()))

        if "audio" not in request.files:
            print("MEETING_UPLOAD: no 'audio' in request.files")
            return jsonify({"error": "No 'audio' file in request"}), 400

        f = request.files["audio"]
        print(f"MEETING_UPLOAD: got file: filename={f.filename}, content_type={f.content_type}")

        # Save uploaded file to temp path (close handle for Windows)
        tmp_file = NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1] or ".webm")
        tmp_path = tmp_file.name
        tmp_file.close()
        f.save(tmp_path)
        print("MEETING_UPLOAD: saved uploaded file to:", tmp_path)

        # Convert uploaded file -> wav (mono, 16k)
        wav_path = tmp_path + ".wav"
        ff = shutil.which("ffmpeg") or "ffmpeg"
        try:
            subprocess.run([
                ff, "-y",
                "-i", tmp_path,
                "-ar", "16000",
                "-ac", "1",
                wav_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print("MEETING_UPLOAD: ffmpeg failed:", e)
            try:
                os.remove(tmp_path)
            except:
                pass
            return jsonify({"error": "ffmpeg conversion failed. Is ffmpeg in PATH?"}), 500

        # Run Vosk transcription on wav
        try:
            data, samplerate = sf.read(wav_path, dtype="int16")
            rec = KaldiRecognizer(vosk_model, samplerate)
            rec.AcceptWaveform(data.tobytes())
            raw_result = rec.FinalResult()
            try:
                raw_json = json.loads(raw_result)
            except Exception:
                raw_json = {}
            transcript_text = raw_json.get("text", "") or ""
            print("MEETING_UPLOAD: transcript:", transcript_text[:200])
        except Exception as e:
            print("MEETING_UPLOAD: Vosk transcription failed:", e)
            try:
                os.remove(tmp_path)
            except:
                pass
            try:
                os.remove(wav_path)
            except:
                pass
            return jsonify({"error": "Vosk transcription failed: " + str(e)}), 500

        # Sentiment
        try:
            sent = sentiment_pipe(transcript_text) if transcript_text else []
        except Exception as e:
            print("MEETING_UPLOAD: sentiment error:", e)
            sent = []

        # Heuristic extraction
        readable = {}
        txt_l = transcript_text.lower()
        ages = [int(n) for n in re.findall(r"\b([1-9][0-9])\b", txt_l) if 18 <= int(n) <= 80]
        if ages:
            readable['Age'] = ages[0]
        moneys = [int(n) for n in re.findall(r"\b([1-9][0-9]{3,6})\b", txt_l)]
        if moneys:
            readable['Monthly Income'] = moneys[0]
        for dept in ["research and development", "hr", "human resources", "sales", "marketing", "research"]:
            if dept in txt_l:
                readable['Department'] = dept.title()
                break
        if "travel" in txt_l or "traveling" in txt_l:
            if "frequent" in txt_l or "often" in txt_l:
                readable['Business Travel'] = "Travel_Frequently"
            else:
                readable['Business Travel'] = "Travel_Rarely"
        if any(p in txt_l for p in ["overtime", "work late", "work overnight"]):
            readable['OverTime'] = "Yes"
        elif any(p in txt_l for p in ["no overtime", "don't work overtime", "do not work overtime"]):
            readable['OverTime'] = "No"

        # compute risk & save (legacy sqlite predictions table)
        risk_score = compute_risk_score(transcript_text, sent)
        probability = float(risk_score)
        prediction = int(probability >= 0.5)
        pred_text = '⚠️ Employee likely to leave' if prediction == 1 else '✅ Employee likely to stay'

        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO predictions (timestamp, input_json, probability, prediction) VALUES (?, ?, ?, ?)',
                (datetime.utcnow().isoformat(), json.dumps({"transcript": transcript_text, "readable": readable}), probability, prediction)
            )
            conn.commit()
            conn.close()
            print("MEETING_UPLOAD: saved DB record")
        except Exception as e:
            print("MEETING_UPLOAD: DB save failed:", e)

        # Build the response payload
        resp = {
            "ok": True,
            "transcript": transcript_text,
            "readable_fields": readable,
            "probability": probability,
            "prediction": prediction,
            "text": pred_text,
            "sentiment": sent
        }
        print("MEETING_UPLOAD: resp preview:", {k: resp.get(k) for k in ('ok','prediction','probability','record_id')})
        # print diarization if present
        if 'record_id' in resp:
            print("MEETING_UPLOAD: diarization saved for id", resp.get('record_id'))

        # --- Persist (SQLAlchemy) a MeetingRecord (with diarization if possible) ---
        try:
            uploads_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            audio_save_path = None
            # Prefer the converted WAV (wav_path) if it exists and is readable by pyannote,
            # otherwise fall back to the original uploaded tmp_path.
            source_audio = None
            if 'wav_path' in locals() and wav_path and os.path.exists(wav_path):
                source_audio = wav_path
            elif 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                source_audio = tmp_path

            if source_audio:
                try:
                    fname = os.path.basename(source_audio)
                    audio_save_path = os.path.join(uploads_dir, fname)
                    shutil.copyfile(source_audio, audio_save_path)
                except Exception as copy_err:
                    print("MEETING_UPLOAD: could not copy source audio to uploads:", copy_err)
                    # fallback: keep original tmp_path if present
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        audio_save_path = tmp_path

            # Attempt diarization (best-effort). This can be slow; keep it guarded.
            diar_json = {}
            try:
                token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
                # lazy-load pipeline into global cache
                global diarization_pipeline
                if 'diarization_pipeline' not in globals() or diarization_pipeline is None:
                    if token:
                        print("DIARIZE: loading pyannote pipeline (authenticated)...")
                        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
                    else:
                        print("DIARIZE: no HF token found in environment; skipping diarization.")
                        diarization_pipeline = None

                if diarization_pipeline is not None and audio_save_path and os.path.exists(audio_save_path):
                    print("DIARIZE: running diarization on", audio_save_path)
                    # pass local file path directly
                    # Ensure diarization uses proper WAV input (16kHz mono)
                    safe_wav_path = audio_save_path
                    if not audio_save_path.lower().endswith(".wav"):
                        safe_wav_path = audio_save_path + ".wav"
                        ff = shutil.which("ffmpeg") or "ffmpeg"
                        subprocess.run([ff, "-y", "-i", audio_save_path, "-ar", "16000", "-ac", "1", safe_wav_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print("DIARIZE: converted audio to WAV for diarization:", safe_wav_path)

                    diar = diarization_pipeline(safe_wav_path)

                    segments = []
                    per_speaker = {}
                    # iterate tracks: yields (segment, track, label) when yield_label=True
                    try:
                        for segment, track, label in diar.itertracks(yield_label=True):
                            seg = {"start": round(float(segment.start), 2), "end": round(float(segment.end), 2), "speaker": str(label)}
                            segments.append(seg)
                            per_speaker.setdefault(str(label), []).append(seg)
                    except Exception:
                        # fallback: the pipeline may have different iterator shape on some versions
                        for turn, _, speaker in diar.itertracks(yield_label=True):
                            seg = {"start": round(float(turn.start), 2), "end": round(float(turn.end), 2), "speaker": str(speaker)}
                            segments.append(seg)
                            per_speaker.setdefault(str(speaker), []).append(seg)

                    diar_json = {"segments": segments, "per_speaker": per_speaker}
                    print(f"DIARIZE: segments={len(segments)}, speakers={len(per_speaker)}")
                else:
                    print("DIARIZE: pipeline not loaded or audio missing; skipping diarization.")
            except Exception as d_err:
                print("DIARIZE: failed:", d_err)
                diar_json = {}

            # ---------- PER-SPEAKER ANALYSIS & PREDICTION ----------
            per_speaker_results = {}
            try:
                ff = shutil.which("ffmpeg") or "ffmpeg"
                if diar_json.get("per_speaker"):
                    for speaker_label, segments in diar_json["per_speaker"].items():
                        speaker_texts = []
                        for seg in segments:
                            s = float(seg.get("start", 0.0))
                            e = float(seg.get("end", 0.0))
                            if e <= s:
                                continue
                            seg_tmp = NamedTemporaryFile(delete=False, suffix=".wav")
                            seg_tmp_path = seg_tmp.name
                            seg_tmp.close()
                            try:
                                subprocess.run([
                                    ff, "-y", "-i", audio_save_path,
                                    "-ss", str(s),
                                    "-to", str(e),
                                    "-ar", "16000", "-ac", "1", seg_tmp_path
                                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                                seg_data, seg_sr = sf.read(seg_tmp_path, dtype="int16")
                                rec_seg = KaldiRecognizer(vosk_model, seg_sr)
                                rec_seg.AcceptWaveform(seg_data.tobytes())
                                seg_raw = rec_seg.FinalResult()
                                try:
                                    seg_json = json.loads(seg_raw)
                                except Exception:
                                    seg_json = {}
                                seg_text = (seg_json.get("text", "") or "").strip()
                                if seg_text:
                                    speaker_texts.append(seg_text)
                            except Exception as seg_err:
                                print(f"PER_SPK: segment failed for {speaker_label} {s}-{e}:", seg_err)
                            finally:
                                try:
                                    if os.path.exists(seg_tmp_path):
                                        os.remove(seg_tmp_path)
                                except Exception:
                                    pass

                        combined_text = " ".join(speaker_texts).strip()
                        try:
                            sp_sent = sentiment_pipe(combined_text) if combined_text else []
                        except Exception as sp_err:
                            print("PER_SPK: sentiment failed for", speaker_label, sp_err)
                            sp_sent = []

                        try:
                            sp_prob = compute_risk_score(combined_text, sp_sent) if combined_text else 0.0
                        except Exception as sp_comp_err:
                            print("PER_SPK: compute_risk_score failed for", speaker_label, sp_comp_err)
                            sp_prob = 0.0

                        per_speaker_results[speaker_label] = {
                            "transcript": combined_text,
                            "segments": segments,
                            "sentiment": sp_sent,
                            "probability": float(sp_prob),
                            "prediction": int(sp_prob >= 0.5)
                        }

                resp["per_speaker"] = per_speaker_results
                print("✅ PER-SPEAKER ANALYSIS COMPLETE:", list(per_speaker_results.keys()))
            except Exception as ps_err:
                print("❌ PER-SPEAKER ANALYSIS FAILED:", ps_err)
                per_speaker_results = {}
                resp["per_speaker"] = per_speaker_results
            # ---------- END PER-SPEAKER ANALYSIS ----------

            # prepare values for DB row
            transcript_val = transcript_text if 'transcript_text' in locals() else (locals().get('transcript') or "")
            readable_val = readable if 'readable' in locals() else (locals().get('readable_fields') or {})
            sentiment_val = sent if 'sent' in locals() else (locals().get('sentiment') or [])
            raw_resp_val = resp

            try:
                rec = MeetingRecord(
                    transcript = transcript_val,
                    readable_fields = json.dumps(readable_val or {}, ensure_ascii=False),
                    sentiment = json.dumps(sentiment_val or {}, ensure_ascii=False),
                    diarization = json.dumps(diar_json or {}, ensure_ascii=False),
                    audio_path = audio_save_path,
                    raw_response = json.dumps(raw_resp_val or {}, ensure_ascii=False)
                )
                print("✅ Saving diarization JSON:", json.dumps(diar_json or {}))

                db.session.add(rec)
                db.session.commit()
                print(f"✅ MEETING_UPLOAD: Saved MeetingRecord id={rec.id}, diar_segments={len((diar_json or {}).get('segments', []))}")
                resp['record_id'] = rec.id
            except Exception as db_err:
                print("❌ MEETING_UPLOAD: SQLAlchemy save failed:", db_err)
        except Exception as persist_err:
            print("MEETING_UPLOAD: persistence step failed:", persist_err)

        # cleanup temp files
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        return jsonify(resp), 200

    except Exception as e:
        traceback.print_exc()
        try:
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        return jsonify({"error": "Server exception: " + str(e)}), 500

    
# --- Replace your existing /records route with this ---
@app.route('/records')
def meeting_records_list():
    try:
        rows = MeetingRecord.query.order_by(MeetingRecord.id.desc()).limit(200).all()
    except Exception as e:
        print("RECORDS: DB query failed:", e)
        rows = []

    recs = []
    for r in rows:
        audio_fn = os.path.basename(r.audio_path) if r.audio_path else None

        # load diarization JSON robustly
        diar_json = {}
        try:
            if getattr(r, "diarization", None):
                diar_json = json.loads(r.diarization)
                if not isinstance(diar_json, dict):
                    diar_json = {}
            else:
                # fallback: sometimes stored in raw_response
                raw = json.loads(r.raw_response) if r.raw_response else {}
                if isinstance(raw, dict) and raw.get("diarization"):
                    diar_json = raw.get("diarization") or {}
        except Exception as e:
            print(f"RECORDS: failed to parse diarization for record {getattr(r,'id',None)}:", e)
            diar_json = {}

        recs.append({
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "transcript": (r.transcript or "")[:2000],
            "readable_fields": json.loads(r.readable_fields) if r.readable_fields else {},
            "sentiment": json.loads(r.sentiment) if r.sentiment else {},
            "diarization_parsed": diar_json,   # important key the template will use
            "audio_filename": audio_fn,
            "audio_url": (url_for('uploads', filename=audio_fn) if audio_fn else None),
            "raw_response": json.loads(r.raw_response) if r.raw_response else {}
        })

    print(f"RECORDS: prepared {len(recs)} records (sample diar for first 3):", [r.get('diarization_parsed') for r in recs[:3]])
    return render_template('records.html', records=recs)


# ----- temporary debug endpoint -----
@app.route('/upload-logger', methods=['POST'])
def upload_logger():
    try:
        import time
        t0 = time.time()
        got = 'audio' in request.files
        size = None
        if got:
            f = request.files['audio']
            sample = f.stream.read(1024)
            size = len(sample)
        print(f"UPLOAD-LOGGER: request arrived, audio_present={got}, sample_bytes={size}, elapsed={time.time()-t0}")
        return jsonify({'ok': True, 'arrived': True, 'audio_present': got, 'sample_bytes': size}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    


# ================== MAIN ==================
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
