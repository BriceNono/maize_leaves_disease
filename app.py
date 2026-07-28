"""app.py — MaizeScan Flask Application (with Maize Leaf Validation)
==================================================================
NEW: MaizeLeafValidator runs before every prediction.
     Non-maize images are rejected with a clear user message.
     Only real maize leaf images proceed to disease classification.

Author : Brice Gaetan Nono Youmbi | Roll No. 202211043
Supervisor: Prof. Jonas Niyitegeka
Institution: Kigali Independent University ULK | Data Science 2025/2026
"""

import os, time, logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from utils.leaf_image     import LeafImage
from utils.cnn_model      import CNNModel
from utils.diagnosis      import DiagnosisResult
from utils.disease_data   import DISEASE_REGISTRY, load_class_order
from utils.leaf_validator import MaizeLeafValidator

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("FlaskApp")

# ── Flask ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]         = os.environ.get("SECRET_KEY", "maizescan-ulk-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"]      = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Class order from class_indices.json (THE FIX) ─────────────────
CLASS_INDICES_JSON = os.path.join("model", "class_indices.json")
CLASS_ORDER_KEYS   = load_class_order(CLASS_INDICES_JSON)
log.info("CLASS_ORDER_KEYS: %s", CLASS_ORDER_KEYS)

# ── Disease model — loaded once at startup ────────────────────────
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

# ── Validator — single shared instance ───────────────────────────
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
    return __import__('numpy').array(probs, dtype=float)


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

@app.route("/rejected")
def rejected():
    """Standalone rejection page — also reachable via query param."""
    reason = request.args.get("reason", "")
    return render_template("rejected.html", reason=reason, demo_mode=DEMO_MODE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Full prediction pipeline with maize leaf validation gate.

    Step 1 : File validation (extension, size)
    Step 2 : Save to uploads/
    Step 3 : LeafImage preprocessing
    Step 4 : Stage-1 colour/texture heuristic validation
    Step 5 : CNN inference (disease model)
    Step 6 : Stage-2 CNN confidence gate validation
    Step 7 : If rejected → render rejected.html
    Step 8 : Build DiagnosisResult and render result.html
    """
    # ── Step 1: File check ────────────────────────────────────────
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
        # ── Step 2: Save ──────────────────────────────────────────
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        log.info("Saved upload: %s", safe_name)

        # ── Step 3: Preprocess ────────────────────────────────────
        leaf = LeafImage.from_path(save_path, img_size=IMG_SIZE)

        # ── Step 4: Stage-1 heuristic validation (no CNN yet) ────
        stage1 = validator.validate(leaf._pil_img, probs=None)
        log.info("Stage-1 validation: valid=%s stage=%s green=%.3f texture=%.4f",
                 stage1.is_valid, stage1.stage, stage1.green_ratio, stage1.texture_score)

        # Hard reject: blank, too dark, too bright, no plant colour at all
        if not stage1.is_valid and stage1.stage == 'heuristic':
            log.info("REJECTED at Stage 1: %s", stage1.reason)
            image_url = url_for("static", filename=f"uploads/{safe_name}")
            return render_template("rejected.html",
                                   reason=stage1.reason,
                                   image_url=image_url,
                                   demo_mode=DEMO_MODE)

        # ── Step 5: CNN inference ─────────────────────────────────
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)
        prob_str = " | ".join(f"{k}:{probs[i]:.3f}"
                              for i, k in enumerate(CLASS_ORDER_KEYS))
        log.info("Raw probs — %s", prob_str)

        # ── Step 6: Stage-2 CNN confidence gate ──────────────────
        stage2 = validator.validate(leaf._pil_img, probs=probs)
        log.info("Stage-2 validation: valid=%s conf=%.3f", stage2.is_valid, stage2.confidence)

        if not stage2.is_valid:
            log.info("REJECTED at Stage 2: %s", stage2.reason)
            image_url = url_for("static", filename=f"uploads/{safe_name}")
            return render_template("rejected.html",
                                   reason=stage2.reason,
                                   image_url=image_url,
                                   demo_mode=DEMO_MODE)

        # ── Step 7: Build diagnosis ───────────────────────────────
        result   = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                   registry=DISEASE_REGISTRY, leaf_image=leaf)
        response  = result.build_response()
        image_url = url_for("static", filename=f"uploads/{safe_name}")

        log.info("ACCEPTED — Prediction: %s | Confidence: %s",
                 response["prediction"], response["confidence_pct"])

        # ── Step 8: Render result ─────────────────────────────────
        return render_template("result.html", response=response,
                               image_url=image_url, demo_mode=DEMO_MODE)

    except Exception as exc:
        log.error("Prediction pipeline error: %s", exc, exc_info=True)
        return render_template("index.html",
                               error=f"Processing error: {exc}",
                               demo_mode=DEMO_MODE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API with same validation gate."""
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

        # Stage-1
        s1 = validator.validate(leaf._pil_img, probs=None)
        if not s1.is_valid:
            return jsonify({
                "rejected": True,
                "reason":   s1.reason,
                "message":  MaizeLeafValidator.REJECTION_MESSAGE,
            }), 422

        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)

        # Stage-2
        s2 = validator.validate(leaf._pil_img, probs=probs)
        if not s2.is_valid:
            return jsonify({
                "rejected": True,
                "reason":   s2.reason,
                "message":  MaizeLeafValidator.REJECTION_MESSAGE,
            }), 422

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
        "validator":          "MaizeLeafValidator v2 (heuristic + CNN gate)",
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
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )