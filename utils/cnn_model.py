"""
utils/cnn_model.py
==================
CNNModel — loads vgg16_maize_best.h5 and runs inference.
Mirrors Colab Cell 6 and Cell 15 exactly.
"""

import logging
import numpy as np
log = logging.getLogger('CNNModel')


class CNNModel:
    def __init__(self, num_classes=4, img_size=(224,224)):
        self.num_classes = num_classes
        self.img_size    = img_size
        self.model       = None

    def load(self, model_path: str) -> None:
        import tensorflow as tf
        try:
            self.model = tf.keras.models.load_model(model_path)
            log.info("Model loaded from '%s' | params: %s",
                     model_path, f"{self.model.count_params():,}")
        except Exception as exc:
            raise RuntimeError(f"Model load failed: {exc}") from exc

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """Returns softmax probs shape (4,). Mirrors Colab: model.predict(arr)[0]"""
        if self.model is None:
            raise RuntimeError("Call CNNModel.load(path) before predict().")
        return self.model.predict(img_array, verbose=0)[0]
