"""
Zero-Label Defect Hunter
Core anomaly detection module using a simplified PatchCore-style approach.
Trains ONLY on good/normal product images.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
import joblib


class FeatureExtractor(nn.Module):
    """Extract mid-level features from a pretrained Wide-ResNet50 backbone."""

    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        backbone = wide_resnet50_2(weights=weights)
        # Use layers up to layer2 / layer3 for rich spatial features
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class DefectHunter:
    """
    Zero-label anomaly detector.
    Learns a memory bank of normal patch features and scores new images
    by nearest-neighbour distance in feature space.
    """

    def __init__(
        self,
        image_size: int = 224,
        device: Optional[str] = None,
        n_neighbors: int = 1,
        projection_dim: int = 128,
    ):
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_neighbors = n_neighbors
        self.projection_dim = projection_dim

        self.extractor = FeatureExtractor().to(self.device)
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.memory_bank: Optional[np.ndarray] = None
        self.projector: Optional[SparseRandomProjection] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.threshold: float = 0.5
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _extract_patches(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Return flattened spatial feature vectors for one image."""
        with torch.no_grad():
            feat = self.extractor(img_tensor.unsqueeze(0).to(self.device))  # (1, C, H, W)
        feat = feat.squeeze(0).cpu().numpy()  # (C, H, W)
        c, h, w = feat.shape
        # reshape to (H*W, C)
        patches = feat.reshape(c, -1).T
        return patches.astype(np.float32)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    # ------------------------------------------------------------------
    # Training (good images only)
    # ------------------------------------------------------------------
    def fit(self, train_dir: str, max_images: Optional[int] = None) -> "DefectHunter":
        """
        Build memory bank from all images under train_dir (expected: .../good/).
        """
        train_path = Path(train_dir)
        image_files = sorted(
            [
                p
                for p in train_path.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if max_images:
            image_files = image_files[:max_images]

        if not image_files:
            raise FileNotFoundError(f"No training images found in {train_dir}")

        all_patches = []
        for i, p in enumerate(image_files):
            img = self._load_image(str(p))
            tensor = self.transform(img)
            patches = self._extract_patches(tensor)
            all_patches.append(patches)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} training images")

        memory = np.vstack(all_patches)
        print(f"Memory bank size before reduction: {memory.shape}")

        # Optional dimensionality reduction for speed
        if memory.shape[1] > self.projection_dim:
            self.projector = SparseRandomProjection(
                n_components=self.projection_dim, random_state=42
            )
            memory = self.projector.fit_transform(memory).astype(np.float32)
            print(f"Projected to: {memory.shape}")

        # Subsample if still huge (simple random coreset)
        max_coreset = 25000
        if memory.shape[0] > max_coreset:
            idx = np.random.choice(memory.shape[0], max_coreset, replace=False)
            memory = memory[idx]
            print(f"Coreset size: {memory.shape}")

        self.memory_bank = memory
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto")
        self.nn_model.fit(self.memory_bank)

        self.stats = {
            "n_train_images": len(image_files),
            "memory_size": int(self.memory_bank.shape[0]),
            "feature_dim": int(self.memory_bank.shape[1]),
        }
        print("Training finished.")
        return self

    def calibrate_threshold(
        self, val_dir: str, percentile: float = 95.0
    ) -> float:
        """
        Set anomaly threshold from validation *good* images.
        Uses the given percentile of max-patch scores.
        """
        val_path = Path(val_dir)
        files = sorted(
            [
                p
                for p in val_path.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        scores = []
        for p in files:
            s, _ = self.predict(str(p), return_map=False)
            scores.append(s)

        if not scores:
            print("Warning: no validation images – keeping default threshold")
            return self.threshold

        self.threshold = float(np.percentile(scores, percentile))
        self.stats["threshold"] = self.threshold
        self.stats["val_scores_mean"] = float(np.mean(scores))
        self.stats["val_scores_std"] = float(np.std(scores))
        print(f"Calibrated threshold (p{percentile}): {self.threshold:.4f}")
        return self.threshold

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self, image_path: str, return_map: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Returns (anomaly_score, anomaly_map).
        anomaly_map is HxW float map (higher = more anomalous) or None.
        """
        if self.nn_model is None or self.memory_bank is None:
            raise RuntimeError("Model not trained. Call fit() first or load a checkpoint.")

        img = self._load_image(image_path)
        tensor = self.transform(img)
        patches = self._extract_patches(tensor)  # (N, C)

        if self.projector is not None:
            patches = self.projector.transform(patches).astype(np.float32)

        distances, _ = self.nn_model.kneighbors(patches)
        # use max distance among neighbours as patch score
        patch_scores = distances.max(axis=1)

        # spatial map
        # feature map size from WideResNet layer3 is roughly 14x14 for 224 input
        side = int(np.sqrt(len(patch_scores)))
        if side * side != len(patch_scores):
            # pad / truncate to square
            side = int(np.ceil(np.sqrt(len(patch_scores))))
            pad = side * side - len(patch_scores)
            patch_scores = np.pad(patch_scores, (0, pad), mode="edge")

        anomaly_map = patch_scores.reshape(side, side)
        # image-level score = max of the map (common for anomaly detection)
        score = float(anomaly_map.max())

        if return_map:
            # upsample to original transformed size
            anomaly_map = cv2.resize(
                anomaly_map, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            return score, anomaly_map
        return score, None

    def predict_and_visualize(
        self, image_path: str, out_dir: str, prefix: str = "result"
    ) -> Dict[str, Any]:
        """
        Run prediction and save original / heatmap / overlay images.
        Returns a dict with score, threshold, result label and file paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        score, amap = self.predict(image_path, return_map=True)
        assert amap is not None

        # Load original at display size
        orig = cv2.imread(image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (self.image_size, self.image_size))

        # Normalise map for visualisation
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heatmap = cv2.applyColorMap(
            (amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)

        orig_path = os.path.join(out_dir, f"{prefix}_original.png")
        heat_path = os.path.join(out_dir, f"{prefix}_heatmap.png")
        over_path = os.path.join(out_dir, f"{prefix}_overlay.png")

        Image.fromarray(orig).save(orig_path)
        Image.fromarray(heatmap).save(heat_path)
        Image.fromarray(overlay).save(over_path)

        label = "SUSPICIOUS" if score >= self.threshold else "NORMAL"

        return {
            "anomaly_score": round(score, 4),
            "threshold": round(self.threshold, 4),
            "result": label,
            "original": orig_path,
            "heatmap": heat_path,
            "overlay": over_path,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save memory bank, projector, threshold and stats."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "memory_bank": self.memory_bank,
            "projector": self.projector,
            "threshold": self.threshold,
            "stats": self.stats,
            "image_size": self.image_size,
            "n_neighbors": self.n_neighbors,
            "projection_dim": self.projection_dim,
        }
        joblib.dump(payload, path)
        print(f"Model saved → {path}")

    def load(self, path: str) -> "DefectHunter":
        payload = joblib.load(path)
        self.memory_bank = payload["memory_bank"]
        self.projector = payload["projector"]
        self.threshold = payload["threshold"]
        self.stats = payload.get("stats", {})
        self.image_size = payload.get("image_size", 224)
        self.n_neighbors = payload.get("n_neighbors", 1)
        self.projection_dim = payload.get("projection_dim", 128)

        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn_model.fit(self.memory_bank)
        print(f"Model loaded ← {path}")
        return self


def train_default_model(
    data_root: str = "data",
    model_path: str = "models/defect_hunter_model.pkl",
) -> DefectHunter:
    """Convenience function used by the notebook and CLI."""
    hunter = DefectHunter(image_size=224)
    train_dir = os.path.join(data_root, "train", "good")
    val_dir = os.path.join(data_root, "validation", "good")

    print("=== Training on good images only ===")
    hunter.fit(train_dir)
    print("=== Calibrating threshold ===")
    hunter.calibrate_threshold(val_dir, percentile=90.0)
    hunter.save(model_path)
    return hunter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Zero-Label Defect Hunter")
    parser.add_argument("--data", default="data", help="Data root directory")
    parser.add_argument(
        "--out", default="models/defect_hunter_model.pkl", help="Output model path"
    )
    args = parser.parse_args()
    train_default_model(args.data, args.out)
