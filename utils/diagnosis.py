"""
utils/diagnosis.py
==================
DiagnosisResult — mirrors Colab Cell 13 exactly.
Includes confidence score level (High/Medium/Low).
"""

import numpy as np


class DiagnosisResult:
    HIGH_CONF   = 0.80
    MEDIUM_CONF = 0.55

    def __init__(self, probs, class_order, registry, leaf_image=None):
        self.probs       = np.array(probs, dtype=float)
        self.class_order = class_order
        self.registry    = registry
        self.leaf_image  = leaf_image
        self._pred_idx   = int(np.argmax(self.probs))
        self._pred_key   = class_order[self._pred_idx]
        self._confidence = float(self.probs[self._pred_idx])
        self._info       = registry[self._pred_key]

    def format_confidence(self) -> str:
        return f"{self._confidence * 100:.1f}%"

    def confidence_level(self) -> str:
        if self._confidence >= self.HIGH_CONF:   return "High"
        if self._confidence >= self.MEDIUM_CONF: return "Medium"
        return "Low"

    def confidence_badge_color(self) -> str:
        if self._confidence >= self.HIGH_CONF:   return "#4caf1f"
        if self._confidence >= self.MEDIUM_CONF: return "#EF9F27"
        return "#E24B4A"

    def get_all_probs_sorted(self) -> list:
        return sorted(
            [(self.registry[k]['label'], float(p))
             for k, p in zip(self.class_order, self.probs)],
            key=lambda x: x[1], reverse=True
        )

    def build_response(self) -> dict:
        all_probs = sorted(
            [{"label": self.registry[k]["label"], "key": k,
              "prob":  round(float(p), 4),
              "pct":   f"{float(p)*100:.1f}%",
              "color": self.registry[k]["color"],
              "icon":  self.registry[k].get("icon",""),
              "bar_width": int(round(float(p)*100))}
             for k, p in zip(self.class_order, self.probs)],
            key=lambda x: x["prob"], reverse=True
        )
        return {
            "prediction":             self._info["label"],
            "class_key":              self._pred_key,
            "confidence":             round(self._confidence, 4),
            "confidence_pct":         self.format_confidence(),
            "confidence_level":       self.confidence_level(),
            "confidence_color":       self.confidence_badge_color(),
            "confidence_bar_pct":     int(round(self._confidence * 100)),
            "scientific":             self._info["scientific"],
            "severity":               self._info["severity"],
            "severity_level":         self._info.get("severity_level", 0),
            "spread":                 self._info["spread"],
            "color":                  self._info["color"],
            "icon":                   self._info.get("icon",""),
            "description":            self._info["description"],
            "actions":                self._info["actions"],
            "prevention":             self._info["prevention"],
            "all_probs":              all_probs,
            "is_healthy":             self._pred_key == "Healthy",
            "is_high_severity":       self._info.get("severity_level",0) >= 3,
            "low_confidence_warning": self._confidence < self.MEDIUM_CONF,
        }
