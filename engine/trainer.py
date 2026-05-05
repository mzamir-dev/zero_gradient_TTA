"""
Main Trainer — Anti-UAV + Anti-UAV410
Handles:
  - Combined dataset training
  - Per-epoch validation
  - Continual learning incremental schedule
  - JSON logging
  - Checkpoint management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from data import AntiUAVDataset, AntiUAV410Dataset
from data import build_train_transforms, build_val_transforms
from data.antiuav_dataset import collate_fn
from models.ctta_owod_model import CTTAOWODModel
from engine.evaluator import Evaluator
from utils.logger import TrainingLogger
from utils.checkpoint import CheckpointManager
from utils.metrics import measure_fps, measure_gflops, count_parameters


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] Device: {self.device}")

        self._build_datasets()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()

        exp_name = cfg["experiment"]["name"]
        self.logger = TrainingLogger(
            log_dir=cfg["experiment"]["log_dir"],
            experiment_name=exp_name,
            config=cfg,
        )
        self.ckpt_manager = CheckpointManager(
            output_dir=cfg["experiment"]["output_dir"],
            keep_last_n=cfg["logging"]["keep_last_n"],
        )

        self.global_iter = 0
        self.best_map = 0.0
        self._baseline_ap = None
        self.early_stop_patience = 5
        self.early_stop_counter = 0
        self.min_delta = 0.001
        self.start_epoch = 1

    # ──────────────────────────────────────────────────────────────
    # Dataset & DataLoader
    # ──────────────────────────────────────────────────────────────

    def _build_datasets(self):
        from data import build_dataset
        tcfg     = self.cfg["train"]
        train_tf = build_train_transforms()
        val_tf   = build_val_transforms()

        # ── Training datasets
        train_datasets = []
        for ds_cfg in self.cfg["train_datasets"]:
            ds = build_dataset(ds_cfg, split="train", transforms=train_tf)
            train_datasets.append(ds)
            print(f"  Added train: {ds_cfg['name']} — {len(ds)} samples")

        combined_train = ConcatDataset(train_datasets)
        self.train_loader = DataLoader(
            combined_train,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=tcfg["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        # ── Validation datasets
        self.val_loaders = {}
        for ds_cfg in self.cfg["val_datasets"]:
            name = ds_cfg["name"]
            ds   = build_dataset(ds_cfg, split="val", transforms=val_tf)
            self.val_loaders[name] = DataLoader(
                ds,
                batch_size=tcfg["batch_size"],
                shuffle=False,
                num_workers=tcfg["num_workers"],
                collate_fn=collate_fn,
                pin_memory=True,
            )
            print(f"  Added val:   {name} — {len(ds)} samples")

        print(f"\n[Trainer] Total train samples: {len(combined_train)}")
        for name, loader in self.val_loaders.items():
            print(f"[Trainer] Val [{name}]: {len(loader.dataset)} samples")

    # ──────────────────────────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────────────────────────

    def _build_model(self):
        self.model = CTTAOWODModel(self.cfg).to(self.device)
        params = count_parameters(self.model)
        print(
            f"[Trainer] Parameters: {params['total_params_M']}M total, "
            f"{params['trainable_params_M']}M trainable"
        )

    # ──────────────────────────────────────────────────────────────
    # Optimizer & Scheduler
    # ──────────────────────────────────────────────────────────────

    def _build_optimizer(self):
        tcfg = self.cfg["train"]
        # Only train non-frozen parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        if tcfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
            )
        else:
            self.optimizer = torch.optim.SGD(
                params, lr=tcfg["lr"], momentum=0.9,
                weight_decay=tcfg["weight_decay"]
            )

    def _build_scheduler(self):
        tcfg = self.cfg["train"]
        if tcfg["lr_scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tcfg["epochs"]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )

    # ──────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────

    def train(self):
        tcfg = self.cfg["train"]
        # scaler = GradScaler(enabled=tcfg.get("amp", True))
        scaler = torch.amp.GradScaler('cuda')

        # Log efficiency stats once before training
        params = count_parameters(self.model)
        gflops = measure_gflops(self.model, img_size=tcfg["img_size"],
                                device=self.device)
        self.logger.log_iteration(0, 0, {
            "total_params_M": params["total_params_M"],
            "trainable_params_M": params["trainable_params_M"],
            "gflops": gflops,
        })

        for epoch in range(self.start_epoch, tcfg["epochs"] + 1):
            train_metrics = self._train_epoch(epoch, scaler)
            val_metrics = self._validate(epoch)
            # Store baseline AP at epoch 1 for forgetting rate reference
            if epoch == 1:
                self._baseline_ap = {
                    cls_id: val_metrics.get(f"antiuav/AP_class{cls_id}", 0.0)
                    for cls_id in range(self.cfg["num_known_classes"])
                }

            # Compute forgetting rate every 5 epochs
            if epoch > 1 and epoch % 5 == 0 and self._baseline_ap is not None:
                current_ap = {
                    cls_id: val_metrics.get(f"antiuav/AP_class{cls_id}", 0.0)
                    for cls_id in range(self.cfg["num_known_classes"])
                }
                from utils.metrics import compute_forgetting_rate
                fr = compute_forgetting_rate(self._baseline_ap, current_ap)
                val_metrics["forgetting_rate"] = fr["forgetting_rate_overall"]
                val_metrics["forgetting_rate_per_class"] = fr["forgetting_rate_per_class"]
                self.logger.log_continual_event("forgetting_rate_measured", {
                    "epoch": epoch,
                    "forgetting_rate_overall": fr["forgetting_rate_overall"],
                    "forgetting_rate_per_class": fr["forgetting_rate_per_class"],
                    "baseline_ap": self._baseline_ap,
                    "current_ap": current_ap,
                })
                print(f"  [Forgetting Rate] Overall: {fr['forgetting_rate_overall']:.4f}")
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            efficiency = {"fps": -1, "gflops": gflops}
            if epoch % 2 == 0:
                fps = measure_fps(
                    self.model, self.device, img_size=tcfg["img_size"],
                    warmup=20, iters=100
                )
                efficiency["fps"] = fps

            map_score = val_metrics.get("mAP@0.5", 0.0)
            is_best = map_score > self.best_map + self.min_delta

            if is_best:
                self.best_map = map_score
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Overfitting check: warn if train loss very low but val mAP stagnating
            train_loss = train_metrics.get("loss", 0.0)
            if train_loss < 0.05 and map_score < 0.4 and epoch > 5:
                print(f"  [Warning] Possible overfitting: train_loss={train_loss:.4f} "
                    f"but val_mAP={map_score:.4f}")

            if epoch % self.cfg["logging"]["save_every"] == 0 or is_best:
                self.ckpt_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch, metrics=val_metrics, is_best=is_best
                )

            # Log only epoch-level summary to JSON
            self.logger.log_epoch(epoch, train_metrics, val_metrics,
                                efficiency=efficiency, lr=lr)
            print(
            f"[Epoch {epoch} Val] "
            f"mAP={val_metrics.get('mAP@0.5', 0.0):.4f}  "
            f"P={val_metrics.get('antiuav/precision', val_metrics.get('precision', 0.0)):.4f}  "
            f"R={val_metrics.get('antiuav/recall',    val_metrics.get('recall',    0.0)):.4f}  "
            f"F1={val_metrics.get('antiuav/F1',       val_metrics.get('F1',       0.0)):.4f}  "
            f"lr={lr:.6f}  "
            f"early_stop={self.early_stop_counter}/{self.early_stop_patience}"
            )
            print("-" * 70)

            print(f"  [EarlyStopping] Counter: {self.early_stop_counter}/{self.early_stop_patience}")

            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\n[EarlyStopping] No improvement for {self.early_stop_patience} epochs. "
                    f"Best mAP: {self.best_map:.4f}. Stopping.")
                break

        print(f"\n[Trainer] Training complete. Best mAP: {self.best_map:.4f}")

    def _train_epoch(self, epoch: int, scaler) -> dict:
        self.model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        energy_loss_sum = 0.0
        rpn_loss_sum = 0.0
        n_batches = 0
        ctr_loss_sum = 0.0

        log_every = self.cfg["logging"]["log_every"]
        tcfg = self.cfg["train"]

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            use_amp = tcfg.get("amp", True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss_dict = self.model(images, targets)
                loss = loss_dict["total"]

            # Capture training feature statistics for TTA alignment
            # Do this outside autocast to avoid float16 statistics
            with torch.no_grad():
                feats = self.model.backbone(images)
                fpn   = self.model.neck(feats)

                # if "0" in fpn:
                #     self.model.tta_adapter.update_train_statistics(fpn["0"].float())

                for level_idx, k in enumerate(sorted(fpn.keys())):
                    self.model.tta_adapter.update_train_statistics(
                        fpn[k].float(), level=level_idx
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), tcfg["grad_clip"]
            )
            scaler.step(self.optimizer)
            scaler.update()

            # EMA teacher update
            self.model.update_ema_teacher(
                self.cfg["model"]["tta"]["ema_decay"]
            )

            total_loss      += loss.item()
            det_loss_sum    += loss_dict.get("cls",        torch.tensor(0.0)).item()
            energy_loss_sum += loss_dict.get("energy_reg", torch.tensor(0.0)).item()
            rpn_loss_sum    += loss_dict.get("bbox",       torch.tensor(0.0)).item()
            ctr_loss_sum    += loss_dict.get("ctr",        torch.tensor(0.0)).item()
            n_batches += 1
            self.global_iter += 1

            if batch_idx % log_every == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"cls={loss_dict.get('cls', torch.tensor(0.0)).item():.4f}  "
                    f"bbox={loss_dict.get('bbox', torch.tensor(0.0)).item():.4f}  "
                    f"ctr={loss_dict.get('ctr', torch.tensor(0.0)).item():.4f}"
                )
        print(
        f"\n[Epoch {epoch} Complete] "
        f"loss={round(total_loss      / max(n_batches, 1), 4):.4f}  "
        f"cls={round(det_loss_sum     / max(n_batches, 1), 4):.4f}  "
        f"bbox={round(rpn_loss_sum    / max(n_batches, 1), 4):.4f}  "
        f"energy={round(energy_loss_sum / max(n_batches, 1), 4):.4f}"
        )

        return {
        "loss":           round(total_loss      / max(n_batches, 1), 4),
        "loss_cls":       round(det_loss_sum    / max(n_batches, 1), 4),
        "loss_bbox":      round(rpn_loss_sum    / max(n_batches, 1), 4),
        "loss_ctr":    round(ctr_loss_sum    / max(n_batches, 1), 4),
        "loss_energy":    round(energy_loss_sum / max(n_batches, 1), 4),
        }

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────

    def _validate(self, epoch: int) -> dict:
        evaluator = Evaluator(
            num_classes=self.cfg["num_known_classes"],
            iou_threshold=self.cfg["eval"]["iou_threshold"],
            conf_threshold=self.cfg["eval"]["conf_threshold"],
            nms_threshold=self.cfg["eval"]["nms_threshold"],
            device=self.device,
        )

        all_metrics = {}
        for ds_name, loader in self.val_loaders.items():
            metrics = evaluator.evaluate(self.model, loader)
            for k, v in metrics.items():
                all_metrics[f"{ds_name}/{k}"] = v

        # Primary metric = mean mAP across val datasets
        maps = [v for k, v in all_metrics.items() if k.endswith("mAP@0.5")]
        all_metrics["mAP@0.5"] = round(
            sum(maps) / max(len(maps), 1), 4
        )

        # Combined precision, recall, F1 across val datasets
        for metric_key in ["precision", "recall", "F1"]:
            vals = [v for k, v in all_metrics.items() if k.endswith(metric_key)]
            if vals:
                all_metrics[metric_key] = round(sum(vals) / len(vals), 4)

        # , add per-class APs
        for ds_name, loader in self.val_loaders.items():
            for cls_id in range(self.cfg["num_known_classes"]):
                key = f"{ds_name}/AP_class{cls_id}"
                if key not in all_metrics:
                    all_metrics[key] = 0.0

        return all_metrics
