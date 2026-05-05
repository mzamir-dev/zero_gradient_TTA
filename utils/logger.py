"""
JSON Training Logger
Saves all training/validation metrics to a structured JSON log file.
Used later to plot loss and accuracy curves for the paper.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Logs training metrics to JSON. Format:
    {
      "experiment": "...",
      "started_at": "...",
      "config": {...},
      "epochs": [
        {
          "epoch": 1,
          "train": {"loss": 0.5, "lr": 1e-4, ...},
          "val": {"mAP@0.5": 0.72, "precision": ..., ...},
          "efficiency": {"fps": 32.1, "gflops": 45.2},
          "timestamp": "..."
        },
        ...
      ],
      "iterations": [
        {"epoch": 1, "iter": 10, "loss": 0.8, "lr": 1e-4, ...},
        ...
      ],
      "continual_events": [...],
      "tta_results": {...}
    }
    """

    def __init__(self, log_dir: str, experiment_name: str, config: dict = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.log_path = self.log_dir / f"{experiment_name}.json"
        self.data = {
            "experiment": experiment_name,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config or {},
            "epochs": [],
            "iterations": [],
            "continual_events": [],
            "test_results": {},
            "tta_results": {},
        }

        print(f"[Logger] Logging to {self.log_path}")

    def log_iteration(self, epoch: int, iteration: int, metrics: Dict[str, Any]):
        """Log a single training iteration."""
        entry = {
            "epoch": epoch,
            "iter": iteration,
            "timestamp": time.strftime("%H:%M:%S"),
            **metrics,
        }
        self.data["iterations"].append(entry)
        self._save()

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]] = None,
        efficiency: Optional[Dict[str, Any]] = None,
        lr: float = None,
    ):
        """Log end-of-epoch summary."""
        entry = {
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train": train_metrics,
        }
        if val_metrics:
            entry["val"] = val_metrics
        if efficiency:
            entry["efficiency"] = efficiency
        if lr is not None:
            entry["lr"] = lr

        self.data["epochs"].append(entry)
        self._save()
        self._print_epoch_summary(epoch, train_metrics, val_metrics)

    def log_test(self, dataset_name: str, metrics: Dict[str, Any]):
        """Log test set results."""
        self.data["test_results"][dataset_name] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **metrics,
        }
        self._save()
        print(f"\n[Test: {dataset_name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    def log_tta(self, scene_name: str, metrics: Dict[str, Any],
                adaptation_frames: int = None):
        """Log TTA evaluation results per scene."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **metrics,
        }
        if adaptation_frames is not None:
            entry["adaptation_frames"] = adaptation_frames
        self.data["tta_results"][scene_name] = entry
        self._save()

    def log_continual_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log continual learning events:
          - new_class_discovered
          - replay_training_done
          - forgetting_rate_measured
        """
        entry = {
            "event": event_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **details,
        }
        self.data["continual_events"].append(entry)
        self._save()
        print(f"[ContinualEvent] {event_type}: {details}")

    def log_forgetting_rate(
        self,
        after_class_id: int,
        ap_before: Dict,
        ap_after: Dict,
        forgetting_rate: Dict,
    ):
        """Log forgetting rate measurement after incremental learning."""
        self.log_continual_event("forgetting_rate_measured", {
            "after_class_id": after_class_id,
            "ap_before": ap_before,
            "ap_after": ap_after,
            "forgetting_rate": forgetting_rate,
        })

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _print_epoch_summary(self, epoch, train, val):
        parts = [f"Epoch {epoch:3d}"]
        if "loss" in train:
            parts.append(f"loss={train['loss']:.4f}")
        if "loss_detection" in train:
            parts.append(f"det={train['loss_detection']:.4f}")
        if val:
            if "mAP@0.5" in val:
                parts.append(f"mAP={val['mAP@0.5']:.4f}")
            if "precision" in val:
                parts.append(f"P={val['precision']:.4f}")
            if "recall" in val:
                parts.append(f"R={val['recall']:.4f}")
            if "F1" in val:
                parts.append(f"F1={val['F1']:.4f}")
        print("  ".join(parts))

    def get_best_epoch(self, metric: str = "mAP@0.5") -> int:
        """Return epoch number with best validation metric."""
        best_val = -1
        best_epoch = 0
        for entry in self.data["epochs"]:
            v = entry.get("val", {}).get(metric, -1)
            if v > best_val:
                best_val = v
                best_epoch = entry["epoch"]
        return best_epoch, best_val
