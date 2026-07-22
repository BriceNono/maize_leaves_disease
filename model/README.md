# Model Directory

## Files required here:

    model/
    ├── vgg16_maize_best.h5       ← your trained model
    └── class_indices.json        ← CRITICAL: saved from Colab

## How to generate class_indices.json (REQUIRED to fix misclassification):

1. Open your Colab notebook
2. After Cell 5 (AugmentationPipeline), run DIAGNOSTIC_COLAB_CELL.py
3. It saves: /content/drive/MyDrive/MaizeCNN_Results/class_indices.json
4. Download it and place it HERE in model/

## What class_indices.json looks like:

If trained with PlantVillage folders:
{
  "Corn___Common_Rust": 0,
  "Corn___Gray_Leaf_Spot": 1,
  "Corn___Northern_Leaf_Blight": 2,
  "Corn___healthy": 3
}

If trained with short folders:
{
  "Blight": 0,
  "Common_Rust": 1,
  "Gray_Leaf_Spot": 2,
  "Healthy": 3
}

## Why this matters:
Keras assigns class indices ALPHABETICALLY from your folder names.
The old hardcoded CLASS_ORDER_KEYS = ['Blight','Common_Rust',...]
assumed short folder names but your dataset used PlantVillage names,
causing 3 out of 4 predictions to be WRONG labels.
class_indices.json fixes this automatically at every startup.

## Preprocessing:
Training:  ImageDataGenerator(rescale=1.0/255)
Inference: arr = np.array(img, dtype=np.float32) / 255.0
Both produce float32 in [0.0, 1.0] — DO NOT change to preprocess_input.
