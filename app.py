"""
app.py — MaizeScan Flask Application (FIXED)
=============================================
FIX APPLIED: CLASS_ORDER_KEYS is now loaded dynamically from
model/class_indices.json at startup instead of being hardcoded.

This permanently fixes the "everything predicted as Common Rust" bug
caused by a mismatch between training folder names (alphabetical) and
the hardcoded class order in deployment.

Author : Brice Gaetan Nono Youmbi | Roll No. 202211043
Supervisor: Prof. Jonas Niyitegeka
Institution: Kigali Independent University ULK | Data Science 2025/2026
"""

import os, time, logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from utils.leaf_image   import LeafImage
from utils.cnn_model    import CNNModel
from utils.diagnosis    import DiagnosisResult
from utils.disease_data import DISEASE_REGISTRY, load_class_order, FALLBACK_CLASS_ORDER

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("FlaskApp")

app = Flask(__name__)
app.config["SECRET_KEY"]         = os.environ.get("SECRET_KEY", "maizescan-ulk-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"]      = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Load class order from JSON (THE FIX) ─────────────────────────
CLASS_INDICES_JSON = os.path.join("model", "class_indices.json")
CLASS_ORDER_KEYS   = load_class_order(CLASS_INDICES_JSON)
log.info("CLASS_ORDER_KEYS resolved to: %s", CLASS_ORDER_KEYS)

# ── Model startup ─────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "vgg16_maize_best.h5")
IMG_SIZE   = (224, 224)
DEMO_MODE  = False

cnn = CNNModel(num_classes=4, img_size=IMG_SIZE)
if os.path.exists(MODEL_PATH):
    try:
        cnn.load(MODEL_PATH)
        log.info("Model ready.")
    except Exception as exc:
        log.warning("Model load failed: %s — DEMO MODE active", exc)
        DEMO_MODE = True
else:
    log.warning("Model file not found — DEMO MODE active")
    DEMO_MODE = True


def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"])


def _demo_probs():
    """Demo probs for when no model is loaded."""
    import numpy as np
    # Returns one prob per class in CLASS_ORDER_KEYS order
    # Simulates a Common_Rust prediction
    probs = [0.05, 0.05, 0.05, 0.05]
    if "Common_Rust" in CLASS_ORDER_KEYS:
        probs[CLASS_ORDER_KEYS.index("Common_Rust")] = 0.85
    else:
        probs[0] = 0.85
    return __import__('numpy').array(probs, dtype=float)


@app.route("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/about")
def about():
    return render_template("about.html", demo_mode=DEMO_MODE)


@app.route("/diseases")
def diseases():
    return render_template("diseases.html", diseases=DISEASE_REGISTRY,
                           class_order=CLASS_ORDER_KEYS, demo_mode=DEMO_MODE)


@app.route("/predict", methods=["POST"])
def predict():
    if "leaf_image" not in request.files:
        return render_template("index.html",
                               error="No file received. Please choose a leaf image.",
                               demo_mode=DEMO_MODE)
    file = request.files["leaf_image"]
    if not file or file.filename == "":
        return render_template("index.html",
                               error="No file selected.",
                               demo_mode=DEMO_MODE)
    if not allowed_file(file.filename):
        return render_template("index.html",
                               error="Invalid file type. Please upload JPG or PNG.",
                               demo_mode=DEMO_MODE)
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)

        # Preprocess — identical to Colab LeafImage._apply_preprocessing
        leaf  = LeafImage.from_path(save_path, img_size=IMG_SIZE)

        # Inference
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)

        # Log raw probs with class labels for debugging
        prob_str = " | ".join(f"{k}:{probs[i]:.3f}"
                              for i, k in enumerate(CLASS_ORDER_KEYS))
        log.info("Raw probs — %s", prob_str)

        # Build diagnosis with confidence score
        result   = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                   registry=DISEASE_REGISTRY, leaf_image=leaf)
        response  = result.build_response()
        image_url = url_for("static", filename=f"uploads/{safe_name}")

        log.info("Prediction: %s | Confidence: %s | Level: %s",
                 response["prediction"], response["confidence_pct"],
                 response["confidence_level"])

        return render_template("result.html", response=response,
                               image_url=image_url, demo_mode=DEMO_MODE)

    except Exception as exc:
        log.error("Prediction error: %s", exc, exc_info=True)
        return render_template("index.html",
                               error=f"Processing error: {exc}",
                               demo_mode=DEMO_MODE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "leaf_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["leaf_image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        leaf  = LeafImage.from_path(save_path, img_size=IMG_SIZE)
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)
        result = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                 registry=DISEASE_REGISTRY, leaf_image=leaf)
        return jsonify(result.build_response())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":             "ok",
        "demo_mode":          DEMO_MODE,
        "model_loaded":       not DEMO_MODE,
        "class_order":        CLASS_ORDER_KEYS,
        "class_indices_json": os.path.exists(CLASS_INDICES_JSON),
        "img_size":           list(IMG_SIZE),
    })


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error="File too large. Max 16 MB.",
                           demo_mode=DEMO_MODE), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error="Page not found.",
                           demo_mode=DEMO_MODE), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
