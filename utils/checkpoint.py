"""Checkpoint save/load utilities."""

import torch
from pathlib import Path
from typing import Optional


class CheckpointManager:
    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(self, model, optimizer, scheduler, epoch: int,
         metrics: dict = None, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics": metrics or {},
        }

        # Always overwrite last checkpoint
        last_path = self.output_dir / "last_model.pth"
        torch.save(state, last_path)

        # Save best separately only when improved
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"[Checkpoint] Best model updated → {best_path}")

        print(f"[Checkpoint] Last model saved → {last_path} (epoch {epoch})")

        # Remove any old numbered checkpoints from previous runs
        for old in self.output_dir.glob("checkpoint_epoch*.pth"):
            old.unlink()

    def load(self, model, path: str, optimizer=None, scheduler=None,
         device: str = "cpu"):
        state = torch.load(path, map_location=device)
        model_state = model.state_dict()

        # Filter out keys with size mismatch — keeps missing keys as defaults
        filtered = {}
        skipped  = []
        for k, v in state["model"].items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt={v.shape} model={model_state[k].shape}")
            else:
                filtered[k] = v

        if skipped:
            print(f"[Checkpoint] Skipped {len(skipped)} size-mismatched keys:")
            for s in skipped:
                print(f"  {s}")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[Checkpoint] Missing keys (using defaults): {len(missing)}")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys (ignored): {len(unexpected)}")

        if optimizer and "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except ValueError as e:
                print(f"[Checkpoint] Optimizer state skipped: {e}")

        if scheduler and state.get("scheduler"):
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"[Checkpoint] Scheduler state skipped: {e}")

        epoch   = state.get("epoch", 0)
        metrics = state.get("metrics", {})
        print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
        return epoch, metrics

    def load_best(self, model, device: str = "cpu"):
        best_path = self.output_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best model at {best_path}")
        return self.load(model, best_path, device=device)
