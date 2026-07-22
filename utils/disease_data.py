"""
utils/disease_data.py
=====================
FIXED: CLASS_ORDER_KEYS now auto-loads from class_indices.json at startup.

The root cause of "everything predicted as Common Rust":
  - PlantVillage folder names: Corn___Common_Rust, Corn___Gray_Leaf_Spot,
    Corn___Northern_Leaf_Blight, Corn___healthy
  - Keras assigns indices ALPHABETICALLY from folder names:
      0 → Corn___Common_Rust
      1 → Corn___Gray_Leaf_Spot
      2 → Corn___Northern_Leaf_Blight
      3 → Corn___healthy
  - Old deployment assumed: 0=Blight, 1=Common_Rust, 2=Gray_Leaf_Spot, 3=Healthy
  - 3 out of 4 indices were WRONG → every prediction was mislabelled

FIX: app.py now reads model/class_indices.json (saved from Colab)
     and builds CLASS_ORDER_KEYS dynamically at startup.
     If class_indices.json is absent, falls back to FALLBACK_CLASS_ORDER.

TO GENERATE class_indices.json: run DIAGNOSTIC_COLAB_CELL.py in Colab,
then download MaizeCNN_Results/class_indices.json and place it in model/.
"""

import os
import json
import logging

log = logging.getLogger('DiseaseData')

# ── Fallback order (PlantVillage original folder names, alphabetical) ─
# These are the ACTUAL class_indices when trained with PlantVillage folders:
#   Corn___Common_Rust=0, Corn___Gray_Leaf_Spot=1,
#   Corn___Northern_Leaf_Blight=2, Corn___healthy=3
# Map to our short keys:
FALLBACK_CLASS_ORDER = [
    'Common_Rust',      # index 0 — Corn___Common_Rust
    'Gray_Leaf_Spot',   # index 1 — Corn___Gray_Leaf_Spot
    'Blight',           # index 2 — Corn___Northern_Leaf_Blight
    'Healthy',          # index 3 — Corn___healthy
]

# ── Folder-name-to-key mapping ────────────────────────────────────
# Maps any possible folder name variant to our DISEASE_REGISTRY key
FOLDER_NAME_MAP = {
    # PlantVillage original
    'Corn___Common_Rust':            'Common_Rust',
    'Corn___Gray_Leaf_Spot':         'Gray_Leaf_Spot',
    'Corn___Northern_Leaf_Blight':   'Blight',
    'Corn___healthy':                'Healthy',
    # Short names
    'Blight':                        'Blight',
    'Common_Rust':                   'Common_Rust',
    'Gray_Leaf_Spot':                'Gray_Leaf_Spot',
    'Healthy':                       'Healthy',
    'healthy':                       'Healthy',
    # Other common variants
    'Northern_Leaf_Blight':          'Blight',
    'Common Rust':                   'Common_Rust',
    'Gray Leaf Spot':                'Gray_Leaf_Spot',
    'Corn_Blight':                   'Blight',
    'Corn_Common_Rust':              'Common_Rust',
    'Corn_Gray_Leaf_Spot':           'Gray_Leaf_Spot',
    'Corn_Healthy':                  'Healthy',
}


def load_class_order(json_path: str) -> list:
    """
    Load class_indices.json produced by Colab DIAGNOSTIC_COLAB_CELL.py
    and return CLASS_ORDER_KEYS sorted by index value.

    Args:
        json_path: path to model/class_indices.json

    Returns:
        list of disease registry keys in model output index order
        e.g. ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
    """
    if not os.path.exists(json_path):
        log.warning(
            "class_indices.json not found at '%s'. "
            "Using FALLBACK_CLASS_ORDER: %s. "
            "Run DIAGNOSTIC_COLAB_CELL.py in Colab to generate the correct file.",
            json_path, FALLBACK_CLASS_ORDER
        )
        return FALLBACK_CLASS_ORDER

    with open(json_path, 'r') as f:
        class_indices = json.load(f)

    log.info("Loaded class_indices.json: %s", class_indices)

    # Sort folder names by their index value (0, 1, 2, 3)
    sorted_folders = sorted(class_indices, key=class_indices.get)
    log.info("Sorted folder order: %s", sorted_folders)

    # Map each folder name to its DISEASE_REGISTRY key
    class_order = []
    for folder in sorted_folders:
        key = FOLDER_NAME_MAP.get(folder)
        if key is None:
            log.error(
                "Unknown folder name '%s' in class_indices.json. "
                "Add it to FOLDER_NAME_MAP in utils/disease_data.py.",
                folder
            )
            raise ValueError(
                f"Unknown folder name '{folder}' in class_indices.json.\n"
                f"Add it to FOLDER_NAME_MAP in utils/disease_data.py.\n"
                f"Known names: {list(FOLDER_NAME_MAP.keys())}"
            )
        class_order.append(key)

    log.info("Resolved CLASS_ORDER_KEYS: %s", class_order)
    return class_order


# ── Disease metadata registry ─────────────────────────────────────
DISEASE_REGISTRY = {
    "Blight": {
        "label":          "Northern Leaf Blight",
        "scientific":     "Exserohilum turcicum",
        "severity":       "High",
        "severity_level": 3,
        "spread":         "Airborne spores — spreads very rapidly",
        "color":          "#E24B4A",
        "icon":           "🔴",
        "description": (
            "Large, cigar-shaped, grayish-green to tan necrotic lesions "
            "running parallel to leaf veins (2.5-15 cm). Yield losses of "
            "30-80% when infection occurs before tasseling."
        ),
        "actions": [
            "Isolate affected rows immediately to limit airborne spread.",
            "Apply mancozeb + metalaxyl systemic fungicide every 7 days.",
            "Remove and destroy heavily infected leaves.",
            "Report outbreak to agricultural extension officer.",
            "Consider early harvest on plots with >50% canopy affected.",
        ],
        "prevention": (
            "Plant Ht-gene resistant hybrids. Two-year crop rotation with "
            "soybean or legumes. Preventive fungicide at V6 stage in "
            "high-risk seasons."
        ),
    },
    "Common_Rust": {
        "label":          "Common Rust",
        "scientific":     "Puccinia sorghi",
        "severity":       "Moderate",
        "severity_level": 2,
        "spread":         "Wind-borne urediniospores — spreads rapidly",
        "color":          "#EF9F27",
        "icon":           "🟡",
        "description": (
            "Small (0.2-2 mm), circular to elongate, powdery, brick-red to "
            "dark-brown pustules on both leaf surfaces. Thrives at 16-23 C "
            "with >95% relative humidity."
        ),
        "actions": [
            "Apply triazole fungicide (propiconazole/tebuconazole) within 48 h.",
            "Remove leaves with heaviest pustule load.",
            "Increase inter-row spacing to improve airflow.",
            "Scout neighbouring plots — rust spreads by wind rapidly.",
            "Apply second treatment if new pustules appear after 10 days.",
        ],
        "prevention": (
            "Select rust-resistant varieties. Apply preventive fungicide "
            "at V6-V8 stage. Maintain adequate soil potassium >100 ppm."
        ),
    },
    "Gray_Leaf_Spot": {
        "label":          "Gray Leaf Spot",
        "scientific":     "Cercospora zeae-maydis",
        "severity":       "High",
        "severity_level": 3,
        "spread":         "Residue-borne — moderate to high in humid conditions",
        "color":          "#888780",
        "icon":           "⚫",
        "description": (
            "Rectangular, tan-to-gray lesions with sharply defined parallel "
            "margins bounded by leaf veins. Favoured by >12 h leaf wetness "
            "and dense plant populations."
        ),
        "actions": [
            "Apply strobilurin + triazole fungicide immediately (delay risks 30-40% yield loss).",
            "Switch to drip irrigation to reduce leaf wetness duration.",
            "Remove lower canopy leaves to improve air circulation.",
            "Deep-plow infected residue after harvest.",
            "Rotate to sorghum, legumes, or cassava next season.",
        ],
        "prevention": (
            "Crop rotation is most effective. Use certified disease-free seed. "
            "Avoid minimum-till in fields with GLS history."
        ),
    },
    "Healthy": {
        "label":          "Healthy",
        "scientific":     "No pathogen detected",
        "severity":       "None",
        "severity_level": 0,
        "spread":         "Not applicable",
        "color":          "#639922",
        "icon":           "🟢",
        "description": (
            "Leaf shows no evidence of fungal infection, lesions, or disease "
            "stress. The plant appears in good physiological condition."
        ),
        "actions": [
            "Continue routine monitoring every 3-4 days.",
            "Maintain soil nitrogen at 50-70 kg/ha.",
            "Ensure proper field drainage.",
            "Scout neighbouring plots for early signs of outbreak.",
            "Record observation in farm field diary.",
        ],
        "prevention": (
            "Continue current practices. Monitor weather — warm, humid "
            "conditions increase disease risk. Consider preventive fungicide "
            "at V6 if seasonal forecast indicates prolonged wet conditions."
        ),
    },
}
