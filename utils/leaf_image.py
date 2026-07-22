"""
utils/leaf_image.py
===================
LeafImage — single-image preprocessing for Flask inference.

MIRRORS COLAB _apply_preprocessing EXACTLY:
    arr = np.array(resized, dtype=np.float32) / 255.0
    self.img_array = np.expand_dims(arr, axis=0)

Training used: ImageDataGenerator(rescale=1.0/255)
Inference uses: arr / 255.0
Both produce float32 in [0.0, 1.0] — identical ranges.
"""

import os, io
import numpy as np
from PIL import Image as PILImage


class LeafImage:
    ALLOWED_EXTENSIONS = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}

    def __init__(self):
        self.source    = None
        self.img_array = None
        self._pil_img  = None

    @classmethod
    def from_path(cls, path: str, img_size=(224,224)) -> 'LeafImage':
        ext = os.path.splitext(path)[1]
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: '{path}'")
        obj = cls()
        obj.source   = path
        obj._pil_img = PILImage.open(path).convert('RGB')
        obj._apply_preprocessing(img_size)
        return obj

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, img_size=(224,224)) -> 'LeafImage':
        obj = cls()
        obj.source   = '<bytes>'
        obj._pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert('RGB')
        obj._apply_preprocessing(img_size)
        return obj

    def _apply_preprocessing(self, img_size) -> None:
        """
        Mirrors Colab Cell 12 _apply_preprocessing exactly:
        1. Resize with LANCZOS
        2. float32 array / 255.0  (matches rescale=1/255 from training)
        3. expand_dims → (1, H, W, 3)
        """
        resized        = self._pil_img.resize(img_size, PILImage.LANCZOS)
        arr            = np.array(resized, dtype=np.float32) / 255.0
        self.img_array = np.expand_dims(arr, axis=0)
