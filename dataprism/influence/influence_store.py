"""
Persistent storage for influence scores.

Only stores SCALAR influence scores — never full gradient vectors.
For 100K samples, this is ~800KB (vs ~800GB for full 2M-dim vectors).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("dataprism.influence.store")


class InfluenceStore:
    """Stores and retrieves scalar influence scores.

    Supports two storage backends:
    - HDF5 (recommended): for large datasets, supports partial reads
    - NPZ (fallback): no extra dependency, simple
    """

    def __init__(self, path: str, format: str = "npz"):
        """Initialize influence store.

        Args:
            path: File path for the store (without extension).
            format: Storage format: 'npz' or 'h5'.
        """
        self._base_path = Path(path).with_suffix("")
        self._format = format

    def save_self_influence(
        self,
        scores: np.ndarray,
        sample_indices: list[int],
        metadata: Optional[dict] = None,
    ) -> str:
        """Save self-influence scores.

        Args:
            scores: Array of shape (n_samples,) with self-influence values.
            sample_indices: Corresponding sample indices.
            metadata: Optional dict with config info, timestamps, etc.

        Returns:
            Path to the saved file.
        """
        data = {
            "scores": scores,
            "sample_indices": np.array(sample_indices),
            "metadata": json.dumps(metadata or {}),
        }

        if self._format == "h5":
            return self._save_hdf5(data, "self_influence")
        else:
            return self._save_npz(data, "self_influence")

    def load_self_influence(self) -> tuple[np.ndarray, np.ndarray]:
        """Load self-influence scores.

        Returns:
            Tuple of (scores, sample_indices).
        """
        if self._format == "h5":
            data = self._load_hdf5("self_influence")
        else:
            data = self._load_npz("self_influence")

        return data["scores"], data["sample_indices"]

    def save_validation_influence(
        self,
        scores: dict[str, np.ndarray],
        sample_indices: list[int],
        metadata: Optional[dict] = None,
    ) -> str:
        """Save multi-dimensional validation influence scores (Phase 3).

        Args:
            scores: Dict mapping dimension name → score array.
            sample_indices: Corresponding sample indices.
            metadata: Optional metadata.

        Returns:
            Path to the saved file.
        """
        data = {
            **scores,
            "sample_indices": np.array(sample_indices),
            "metadata": json.dumps(metadata or {}),
        }

        if self._format == "h5":
            return self._save_hdf5(data, "validation_influence")
        else:
            return self._save_npz(data, "validation_influence")

    def load_validation_influence(self) -> dict[str, np.ndarray]:
        """Load multi-dimensional validation influence scores.

        Returns:
            Dict mapping dimension name → score array, plus 'sample_indices'.
        """
        if self._format == "h5":
            return self._load_hdf5("validation_influence")
        else:
            return self._load_npz("validation_influence")

    def save_selection_result(
        self,
        selected_indices: list[int],
        labels: dict[int, str],
        metadata: Optional[dict] = None,
    ) -> str:
        """Save final selection results with per-sample labels.

        Args:
            selected_indices: Indices of selected samples.
            labels: Dict mapping index → label (clean, outlier, redundant, etc.).
            metadata: Optional metadata.

        Returns:
            Path to the saved file.
        """
        data = {
            "selected_indices": np.array(selected_indices),
            "labels": json.dumps(labels),
            "metadata": json.dumps(metadata or {}),
        }

        if self._format == "h5":
            return self._save_hdf5(data, "selection_result")
        else:
            return self._save_npz(data, "selection_result")

    # ── Backend Implementations ──────────────────────────────────────

    def _save_npz(self, data: dict, prefix: str) -> str:
        path = str(self._base_path) + f"_{prefix}.npz"
        save_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                save_data[key] = np.array([value.encode('utf-8')])
            else:
                save_data[key] = value
        np.savez_compressed(path, **save_data)
        logger.info("Saved %s to %s", prefix, path)
        return path

    def _load_npz(self, prefix: str) -> dict:
        path = str(self._base_path) + f"_{prefix}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Influence store not found: {path}")
        loaded = np.load(path, allow_pickle=True)
        result = {}
        for key in loaded.keys():
            result[key] = loaded[key]
        return result

    def _save_hdf5(self, data: dict, prefix: str) -> str:
        import h5py
        path = str(self._base_path) + f"_{prefix}.h5"
        with h5py.File(path, "w") as f:
            for key, value in data.items():
                if isinstance(value, str):
                    f.create_dataset(key, data=np.array([value.encode('utf-8')]))
                else:
                    f.create_dataset(key, data=value, compression="gzip")
        logger.info("Saved %s to %s", prefix, path)
        return path

    def _load_hdf5(self, prefix: str) -> dict:
        import h5py
        path = str(self._base_path) + f"_{prefix}.h5"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Influence store not found: {path}")
        result = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                result[key] = f[key][:]
        return result
