"""
app.py — MaizeScan Flask Application
======================================
v2: Added MaizeLeafValidator — two-stage gate that runs before the
    disease CNN to reject non-maize images with a clear user message.

Author : Brice Gaetan Nono Youmbi | Roll No. 202211043
Supervisor: Prof. Jonas Niyitegeka
Institution: Kigali Independent University ULK | Data Science 2025/2026
"""

import os

# Reduce TensorFlow memory usage on Render
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import time, logging

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from utils.leaf_image     import LeafImage
from utils.cnn_model      import CNNModel
from utils.diagnosis      import DiagnosisResult
from utils.leaf_validator import MaizeLeafValidator
from utils.disease_data   import DISEASE_REGISTRY, load_class_order, FALLBACK_CLASS_ORDER

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

# ── Class order from class_indices.json ──────────────────────────
CLASS_INDICES_JSON = os.path.join("model", "class_indices.json")
CLASS_ORDER_KEYS   = load_class_order(CLASS_INDICES_JSON)
log.info("CLASS_ORDER_KEYS: %s", CLASS_ORDER_KEYS)

# ── Disease CNN model ─────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "vgg16_maize_best.h5")
IMG_SIZE   = (224, 224)
DEMO_MODE  = False

cnn = CNNModel(num_classes=4, img_size=IMG_SIZE)
if os.path.exists(MODEL_PATH):
    try:
        cnn.load(MODEL_PATH)
        log.info("Disease model ready.")
    except Exception as exc:
        log.warning("Model load failed: %s — DEMO MODE", exc)
        DEMO_MODE = True
else:
    log.warning("Model file not found — DEMO MODE")
    DEMO_MODE = True

# ── Maize leaf validator (always active) ─────────────────────────
validator = MaizeLeafValidator()
log.info("MaizeLeafValidator ready.")


# ── Helpers ───────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"])


def _demo_probs():
    import numpy as np
    probs = [0.05, 0.05, 0.05, 0.05]
    if "Common_Rust" in CLASS_ORDER_KEYS:
        probs[CLASS_ORDER_KEYS.index("Common_Rust")] = 0.85
    else:
        probs[0] = 0.85
    return np.array(probs, dtype=float)


# ── Routes ────────────────────────────────────────────────────────
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
    """
    Prediction pipeline with maize leaf validation gate:

    Step 1 : Validate file (extension, size)
    Step 2 : Save to uploads/
    Step 3 : LeafImage.from_path() — preprocess
    Step 4 : MaizeLeafValidator.validate_visual() — Stage 1 gate
             → if NOT a maize leaf → render invalid.html with clear message
    Step 5 : CNNModel.predict() — disease inference
    Step 6 : MaizeLeafValidator.validate_cnn() — Stage 2 entropy gate
             → if model completely uncertain → render invalid.html
    Step 7 : DiagnosisResult.build_response()
    Step 8 : render result.html
    """
    # Step 1: file validation
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
        # Step 2: save
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        image_url = url_for("static", filename=f"uploads/{safe_name}")

        # Step 3: preprocess
        leaf = LeafImage.from_path(save_path, img_size=IMG_SIZE)

        # Step 4: Stage 1 — visual validation
        v1 = validator.validate_visual(leaf)
        log.info("Stage1 validator: valid=%s code=%s details=%s",
                 v1.is_valid, v1.reason_code, v1.details)
        if not v1.is_valid:
            return render_template(
                "invalid.html",
                image_url  = image_url,
                reason_text= v1.reason_text,
                hint       = v1.hint,
                reason_code= v1.reason_code,
                demo_mode  = DEMO_MODE,
            )

        # Step 5: disease CNN inference
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)

        # Log raw probs
        prob_str = " | ".join(
            f"{k}:{probs[i]:.3f}" for i, k in enumerate(CLASS_ORDER_KEYS)
        )
        log.info("Raw probs — %s", prob_str)

        # Step 6: Stage 2 — CNN entropy check
        v2 = validator.validate_cnn(probs)
        log.info("Stage2 validator: valid=%s code=%s details=%s",
                 v2.is_valid, v2.reason_code, v2.details)
        if not v2.is_valid:
            return render_template(
                "invalid.html",
                image_url  = image_url,
                reason_text= v2.reason_text,
                hint       = v2.hint,
                reason_code= v2.reason_code,
                demo_mode  = DEMO_MODE,
            )

        # Step 7: build full diagnosis
        result   = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                   registry=DISEASE_REGISTRY, leaf_image=leaf)
        response = result.build_response()

        log.info("Prediction: %s | Confidence: %s | Level: %s",
                 response["prediction"], response["confidence_pct"],
                 response["confidence_level"])

        # Step 8: render result
        return render_template("result.html", response=response,
                               image_url=image_url, demo_mode=DEMO_MODE)

    except Exception as exc:
        log.error("Prediction error: %s", exc, exc_info=True)
        return render_template("index.html",
                               error=f"Processing error: {exc}",
                               demo_mode=DEMO_MODE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API with validation — same gate as /predict."""
    if "leaf_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["leaf_image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        leaf = LeafImage.from_path(save_path, img_size=IMG_SIZE)

        # Stage 1
        v1 = validator.validate_visual(leaf)
        if not v1.is_valid:
            return jsonify({
                "valid":       False,
                "reason_code": v1.reason_code,
                "message":     v1.reason_text,
                "hint":        v1.hint,
            }), 422

        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)

        # Stage 2
        v2 = validator.validate_cnn(probs)
        if not v2.is_valid:
            return jsonify({
                "valid":       False,
                "reason_code": v2.reason_code,
                "message":     v2.reason_text,
                "hint":        v2.hint,
            }), 422

        result = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                 registry=DISEASE_REGISTRY, leaf_image=leaf)
        resp = result.build_response()
        resp["valid"] = True
        return jsonify(resp)

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
        "validator":          "MaizeLeafValidator v2 (2-stage)",
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
