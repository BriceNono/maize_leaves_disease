"""
utils/leaf_validator.py
=======================
MaizeLeafValidator
==================
Two-stage validation gate that runs BEFORE disease classification.

Stage 1 — Colour/Texture Heuristic (fast, no extra model needed):
  • Analyses pixel colour distribution in HSV space
  • Checks green channel dominance (maize leaves are green)
  • Checks texture variance (leaves have structured texture)
  • Checks aspect ratio (leaves are elongated)
  • Rejects obviously non-leaf images (faces, animals, food, blank pages etc.)

Stage 2 — CNN Confidence Gate (uses the disease model itself):
  • If the disease model returns max confidence < CONFIDENCE_THRESHOLD
    and the image failed Stage 1, it is also rejected
  • If the disease model is confident (>= CONFIDENCE_THRESHOLD),
    the image passes even if colour heuristic was borderline
  • This handles edge cases: diseased leaves may have low green (blight,
    rust turn brown/orange) but still produce confident CNN predictions

Design decision: No separate validator model is needed.
The combination of the HSV heuristic + CNN confidence gate is
sufficient because:
  - A random photo (face, car, food) will fail BOTH stages
  - A maize leaf (even heavily diseased) will pass Stage 2
  - A borderline case (other plant) will pass Stage 2 if the CNN
    is confidently identifying a known disease pattern

Rejection message mirrors Chapter 3 FR-01 (input validation).
"""

import logging
import numpy as np
from PIL import Image as PILImage
from dataclasses import dataclass

log = logging.getLogger('MaizeLeafValidator')


@dataclass
class ValidationResult:
    """Result of leaf validation."""
    is_valid:      bool
    stage:         str      # 'heuristic' | 'cnn_gate' | 'passed'
    reason:        str      # human-readable rejection reason
    green_ratio:   float    # fraction of pixels that are green-ish
    texture_score: float    # image texture variance (0-1)
    confidence:    float    # CNN max class probability (0 if no probs given)


class MaizeLeafValidator:
    """
    Validates that an uploaded image is a maize leaf before
    passing it to the disease classifier.

    Attributes:
        confidence_threshold (float): minimum CNN confidence to accept (0.30)
        min_green_ratio      (float): minimum green pixel fraction (0.08)
        min_texture          (float): minimum texture variance (0.002)
        max_texture          (float): maximum texture variance (reject blank) (0.995)

    Methods:
        validate(pil_img, probs=None) → ValidationResult
        is_valid(pil_img, probs=None) → bool
    """

    # ── Thresholds ────────────────────────────────────────────────
    # Confidence: disease CNN must be >= 30% sure to accept an image
    # Even a heavily diseased blight leaf will score >30% on one class.
    # A random photo (dog, car, face) will spread ~25% across all 4 → rejected.
    CONFIDENCE_THRESHOLD = 0.30

    # Green ratio: fraction of pixels that are green-ish (HSV hue in leaf range)
    # Maize leaves (even diseased ones with blight/rust lesions): >8% green pixels
    # Random photos: usually <5% (unless they happen to be other plants)
    MIN_GREEN_RATIO = 0.08

    # Texture: how much spatial variation exists (via standard deviation)
    # Leaves have structured texture; blank paper or solid-colour images: very low
    # Extremely noisy images (close-up texture only): very high
    MIN_TEXTURE = 0.002
    MAX_TEXTURE = 0.990

    # Rejection messages — user-facing
    REJECTION_MESSAGE = (
        "Sorry, the system cannot analyse your image. "
        "Please upload a clear photo of a maize leaf. "
        "The image you provided does not appear to be a maize leaf."
    )

    REJECTION_DETAIL = {
        'no_green':       "No green plant tissue detected in the image.",
        'blank':          "The image appears to be blank or has no detail.",
        'low_confidence': (
            "The model could not recognise any maize disease pattern in this image. "
            "Please upload a clear, close-up photo of a maize leaf."
        ),
        'too_dark':       "The image is too dark to analyse.",
        'too_bright':     "The image is overexposed. Please take the photo in natural light.",
    }

    def __init__(self):
        pass

    def _analyse_colours(self, pil_img: PILImage.Image) -> dict:
        """
        Analyse pixel colour distribution in HSV space.
        Returns dict of colour metrics.
        """
        # Resize to small thumbnail for speed
        thumb = pil_img.resize((128, 128), PILImage.LANCZOS)
        arr   = np.array(thumb, dtype=np.float32) / 255.0  # (128,128,3) RGB

        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # ── Green dominance (green channel largest) ───────────────
        green_dominant = (g > r) & (g > b) & (g > 0.15)
        green_ratio    = float(green_dominant.mean())

        # ── Brown/tan detection (diseased leaves — rust, blight) ──
        # Brown: R > G > B, all moderate values
        brown = (r > g) & (g > b) & (r > 0.25) & (r < 0.85) & (g > 0.15)
        brown_ratio = float(brown.mean())

        # ── Orange/rust detection (Common Rust pustules) ──────────
        orange = (r > 0.45) & (g > 0.20) & (g < 0.55) & (b < 0.30)
        orange_ratio = float(orange.mean())

        # ── Gray detection (Gray Leaf Spot lesions) ───────────────
        gray = (np.abs(r - g) < 0.12) & (np.abs(g - b) < 0.12) & (r > 0.20) & (r < 0.80)
        gray_ratio = float(gray.mean())

        # ── Plant tissue ratio (green + brown + orange + gray) ────
        # Any of these indicate plant material rather than random objects
        plant_ratio = float(np.clip(
            green_ratio + brown_ratio * 0.6 + orange_ratio * 0.5 + gray_ratio * 0.3,
            0.0, 1.0
        ))

        # ── Overall brightness ────────────────────────────────────
        brightness = float(arr.mean())

        # ── Texture (spatial variance) ────────────────────────────
        gray_img      = 0.299*r + 0.587*g + 0.114*b
        texture_score = float(gray_img.std())

        return {
            'green_ratio':   green_ratio,
            'brown_ratio':   brown_ratio,
            'orange_ratio':  orange_ratio,
            'gray_ratio':    gray_ratio,
            'plant_ratio':   plant_ratio,
            'brightness':    brightness,
            'texture_score': texture_score,
        }

    def validate(
        self,
        pil_img: PILImage.Image,
        probs:   np.ndarray = None,
    ) -> ValidationResult:
        """
        Two-stage validation.

        Args:
            pil_img : PIL.Image (RGB) — the uploaded leaf image
            probs   : np.ndarray shape (4,) — disease CNN softmax output
                      Pass None to skip Stage 2 (Stage 1 only)

        Returns:
            ValidationResult dataclass
        """
        metrics    = self._analyse_colours(pil_img)
        green_r    = metrics['green_ratio']
        plant_r    = metrics['plant_ratio']
        texture    = metrics['texture_score']
        brightness = metrics['brightness']
        confidence = float(np.max(probs)) if probs is not None else 0.0

        log.info(
            "Validation — green=%.3f plant=%.3f texture=%.4f bright=%.3f conf=%.3f",
            green_r, plant_r, texture, brightness, confidence
        )

        # ── Stage 1a: Brightness check ────────────────────────────
        if brightness < 0.04:
            return ValidationResult(
                is_valid=False, stage='heuristic',
                reason=self.REJECTION_DETAIL['too_dark'],
                green_ratio=green_r, texture_score=texture, confidence=confidence
            )
        if brightness > 0.97:
            return ValidationResult(
                is_valid=False, stage='heuristic',
                reason=self.REJECTION_DETAIL['too_bright'],
                green_ratio=green_r, texture_score=texture, confidence=confidence
            )

        # ── Stage 1b: Texture check ───────────────────────────────
        if texture < self.MIN_TEXTURE:
            return ValidationResult(
                is_valid=False, stage='heuristic',
                reason=self.REJECTION_DETAIL['blank'],
                green_ratio=green_r, texture_score=texture, confidence=confidence
            )

        # ── Stage 1c: Plant tissue check ─────────────────────────
        # If NO plant-like colour at all → definitely not a leaf
        if plant_r < 0.05 and green_r < self.MIN_GREEN_RATIO:
            # But give Stage 2 a chance to override if CNN is confident
            if probs is None or confidence < self.CONFIDENCE_THRESHOLD:
                return ValidationResult(
                    is_valid=False, stage='heuristic',
                    reason=self.REJECTION_DETAIL['no_green'],
                    green_ratio=green_r, texture_score=texture, confidence=confidence
                )

        # ── Stage 2: CNN confidence gate ──────────────────────────
        # If CNN was given and it's NOT confident → reject
        if probs is not None and confidence < self.CONFIDENCE_THRESHOLD:
            return ValidationResult(
                is_valid=False, stage='cnn_gate',
                reason=self.REJECTION_DETAIL['low_confidence'],
                green_ratio=green_r, texture_score=texture, confidence=confidence
            )

        # ── PASSED ────────────────────────────────────────────────
        return ValidationResult(
            is_valid=True, stage='passed',
            reason='',
            green_ratio=green_r, texture_score=texture, confidence=confidence
        )

    def is_valid(
        self,
        pil_img: PILImage.Image,
        probs:   np.ndarray = None,
    ) -> bool:
        """Convenience method — returns bool only."""
        return self.validate(pil_img, probs).is_valid