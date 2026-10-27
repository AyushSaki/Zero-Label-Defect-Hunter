"""
Zero-Label Defect Hunter – Flask web application
"""

import os
import csv
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from defect_hunter import DefectHunter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
HISTORY_CSV = BASE_DIR / "results" / "inspection_history.csv"
MODEL_PATH = BASE_DIR / "models" / "defect_hunter_model.pkl"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.secret_key = "zero-label-defect-hunter-secret-key-change-in-prod"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "results").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------
_hunter: DefectHunter | None = None


def get_hunter() -> DefectHunter:
    global _hunter
    if _hunter is None:
        _hunter = DefectHunter()
        if MODEL_PATH.exists():
            _hunter.load(str(MODEL_PATH))
        else:
            # Train on the sample data if no model is present
            print("No saved model found – training on sample dataset …")
            from defect_hunter import train_default_model

            _hunter = train_default_model(
                data_root=str(BASE_DIR / "data"),
                model_path=str(MODEL_PATH),
            )
    return _hunter


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def append_history(row: dict) -> None:
    write_header = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Date", "Image", "Anomaly Score", "Threshold", "Result"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", result=None)


@app.route("/inspect", methods=["POST"])
def inspect():
    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload PNG/JPG/JPEG/BMP.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{filename}"
    upload_path = UPLOAD_FOLDER / saved_name
    file.save(upload_path)

    try:
        hunter = get_hunter()
        prefix = f"pred_{timestamp}"
        out = hunter.predict_and_visualize(
            str(upload_path),
            out_dir=str(RESULT_FOLDER),
            prefix=prefix,
        )

        # Make paths relative to static for the template
        out["original_url"] = url_for(
            "static", filename=f"results/{Path(out['original']).name}"
        )
        out["heatmap_url"] = url_for(
            "static", filename=f"results/{Path(out['heatmap']).name}"
        )
        out["overlay_url"] = url_for(
            "static", filename=f"results/{Path(out['overlay']).name}"
        )

        # History
        append_history(
            {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Image": filename,
                "Anomaly Score": out["anomaly_score"],
                "Threshold": out["threshold"],
                "Result": out["result"],
            }
        )

        return render_template("index.html", result=out, filename=filename)

    except Exception as e:
        flash(f"Inspection failed: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/history")
def history():
    rows = []
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[::-1]  # newest first
    return render_template("history.html", rows=rows)


@app.route("/health")
def health():
    model_ok = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_ok,
        "device": "cpu",  # simplified
    }


if __name__ == "__main__":
    # Ensure model exists before serving
    get_hunter()
    app.run(host="0.0.0.0", port=5000, debug=True)
