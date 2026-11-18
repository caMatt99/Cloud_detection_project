"""
Experiment runner: one config + one seed -> one W&B run.

Protocol (Phase 0):
  - every source of randomness is seeded; the split is verified and fingerprinted
  - the checkpoint is selected on VALIDATION (`selection.monitor`), never on test
  - test is evaluated once with the selected weights ("best") and, for
    comparability with the old runs, once with the last-epoch weights ("last")
  - headline metrics are binary (positive = 'true'), as in the paper
  - several seeds per config, reported as mean ± std; W&B groups runs by
    experiment.name so the group view gives the same aggregate

Usage:
    from experiment import run_seeds, run_seeds_parallel
    table = run_seeds(load_config("configs/convnext_legacy.yaml"))       # sequential, in-process
    table = run_seeds_parallel("configs/convnext_legacy.yaml")           # one process per GPU (Kaggle T4 x2)

    python src/experiment.py --config configs/convnext_legacy.yaml --seeds 0 1 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import wandb

from dataset import check_split, get_dataloaders, split_fingerprint
from engine import BestCheckpoint, EarlyStopping, build_optimizer, evaluate, train_one_epoch
from evaluate import PAPER_RESNET50, compute_metrics
from models import apply_legacy_dropout, count_parameters, get_device, get_model, get_unfrozen_params
from utils import flatten_dict, load_config, seed_everything

REPORT_METRICS = ("accuracy", "n_correct", "f1", "precision", "recall")


def _build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    m = cfg["model"]
    model = get_model(
        m["name"],
        num_classes=m.get("num_classes", 2),
        pretrained=m.get("pretrained", True),
        drop_rate=m.get("drop_rate", 0.0),
        drop_path_rate=m.get("drop_path_rate", 0.0),
    )
    if m.get("legacy_dropout") is not None:
        n = apply_legacy_dropout(model, m["legacy_dropout"])
        print(f"[model] LEGACY dropout p={m['legacy_dropout']} on {n} nn.Dropout layers")
    total, _ = count_parameters(model)
    print(f"[model] {m['name']}  params={total:,}")
    return model.to(device)


def run_experiment(cfg: Dict[str, Any], seed: int, use_wandb: bool = True) -> Dict[str, Any]:
    exp, d, t, sel = cfg["experiment"], cfg["dataset"], cfg["training"], cfg.get("selection", {})
    seed_everything(seed, exp.get("deterministic", True))
    device = get_device()

    fingerprint = check_split(d["path"]) if exp.get("check_split", True) else split_fingerprint(d["path"])
    print(f"[data] split fingerprint {fingerprint}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        d["path"],
        batch_size=d["batch_size"],
        image_size=d["image_size"],
        num_workers=d.get("num_workers", 2),
        normalization=d.get("normalization", "imagenet"),
        augmentation=d.get("augmentation", "paper"),
        aug_cfg=cfg.get("augmentation"),
        seed=seed,
    )

    model = _build_model(cfg, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    opt_kwargs = dict(
        name=t["optimizer"],
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
        momentum=t.get("momentum", 0.9),
    )
    schedule = t.get("schedule", "full")
    if schedule not in ("full", "gradual_unfreeze"):
        raise ValueError(f"Unknown training.schedule '{schedule}'")
    optimizer = build_optimizer(model.parameters(), **opt_kwargs) if schedule == "full" else None

    amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = BestCheckpoint(sel.get("monitor", "val_loss"))
    stopper = EarlyStopping(**t.get("early_stopping", {"mode": "none"}))

    run = None
    if use_wandb:
        run = wandb.init(
            project=exp.get("wandb_project", "cloud-detection"),
            group=exp["name"],
            name=f"{exp['name']}-s{seed}",
            job_type="train",
            tags=exp.get("tags", []),
            config={**flatten_dict({k: v for k, v in cfg.items() if not k.startswith("_")}),
                    "seed": seed, "split_fingerprint": fingerprint, "amp_active": amp},
            reinit="finish_previous",
        )

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    try:
        epoch = 0
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            log: Dict[str, Any] = {"epoch": epoch}

            if schedule == "gradual_unfreeze":
                # LEGACY: optimizer rebuilt every epoch (momentum reset), as in commit 89cf8be.
                groups, phase = get_unfrozen_params(
                    model, epoch, lr_head=t["learning_rate"],
                    lr_backbone=t["learning_rate"] * t.get("backbone_lr_ratio", 0.1),
                )
                optimizer = build_optimizer(groups, **opt_kwargs)
                log["unfreeze_phase"] = phase

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, t.get("grad_clip", 0.0)
            )
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device, amp)
            improved = best.update(model, epoch, va_loss, va_acc)

            for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
                history[k].append(v)
            log.update({"train/loss": tr_loss, "train/accuracy": tr_acc,
                        "val/loss": va_loss, "val/accuracy": va_acc, "epoch_time_s": time.time() - t0})
            if run:
                run.log(log)
            print(f"[s{seed}] ep {epoch:3d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
                  f"val {va_loss:.4f}/{va_acc:.4f}{'  *' if improved else ''}  ({log['epoch_time_s']:.0f}s)")

            if stopper.step(va_loss):
                print(f"[s{seed}] early stopping at epoch {epoch} ({stopper.mode})")
                break
        stopped_epoch = epoch

        results: Dict[str, Any] = {
            "seed": seed,
            "split_fingerprint": fingerprint,
            "best_epoch": best.best_epoch,
            "stopped_epoch": stopped_epoch,
            "val_loss_best": history["val_loss"][best.best_epoch - 1],
            "val_acc_best": history["val_acc"][best.best_epoch - 1] * 100,
        }
        preds_best = None
        for tag in ("last", "best"):
            if tag == "best":
                best.restore(model)
            te_loss, _, probs, labels = evaluate(model, test_loader, criterion, device, amp)
            preds = (probs >= 0.5).astype(int)
            metrics = compute_metrics(labels, preds)
            results.update({f"test_{tag}_loss": te_loss, **{f"test_{tag}_{k}": v for k, v in metrics.items()}})
            if tag == "best":
                preds_best = preds

        print(f"[s{seed}] best epoch {best.best_epoch} | "
              f"test acc best={results['test_best_accuracy']:.2f}% ({results['test_best_n_correct']}/"
              f"{results['test_best_n_total']})  last={results['test_last_accuracy']:.2f}%")

        _save_outputs(cfg, seed, model, results, history)

        if run:
            run.summary.update({
                **{f"test/{k}": results[f"test_best_{k}"] for k in (*REPORT_METRICS, "loss")},
                **{f"test_last/{k}": results[f"test_last_{k}"] for k in (*REPORT_METRICS, "loss")},
                "best_epoch": results["best_epoch"],
                "stopped_epoch": stopped_epoch,
                "val/accuracy_best": results["val_acc_best"],
                "val/loss_best": results["val_loss_best"],
            })
            run.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels.tolist(), preds=preds_best.tolist(), class_names=class_names)})
    finally:
        if run:
            run.finish()

    results["history"] = history
    return results


def _save_outputs(cfg, seed, model, results, history) -> None:
    exp = cfg["experiment"]
    paths = cfg.get("paths", {})
    if exp.get("save_checkpoint", True) and paths.get("checkpoints"):
        ckpt_dir = os.path.join(paths["checkpoints"], exp["name"])
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"seed{seed}.pt"))
    if paths.get("results"):
        res_dir = os.path.join(paths["results"], exp["name"])
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, f"seed{seed}.json"), "w") as f:
            json.dump({**results, "history": history}, f, indent=2, default=float)


def summarize(per_seed: List[Dict[str, Any]]) -> pd.DataFrame:
    """mean / std / min / max over seeds of the scalar results."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in per_seed]).set_index("seed")
    numeric = df.select_dtypes("number")
    return pd.concat([df, numeric.agg(["mean", "std", "min", "max"])])


def _results_dir(cfg: Dict[str, Any]) -> str:
    return os.path.join(cfg["paths"]["results"], cfg["experiment"]["name"])


def load_seed_results(cfg: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """Reads the per-seed JSON files written by run_experiment."""
    out = []
    for s in seeds:
        with open(os.path.join(_results_dir(cfg), f"seed{s}.json")) as f:
            out.append(json.load(f))
    return out


def report(cfg: Dict[str, Any], per_seed: List[Dict[str, Any]]) -> pd.DataFrame:
    """Builds the summary table, writes summary.csv and prints the headline columns."""
    table = summarize(per_seed)
    if cfg.get("paths", {}).get("results"):
        out = os.path.join(_results_dir(cfg), "summary.csv")
        table.to_csv(out)
        print(f"[summary] written to {out}")

    cols = [f"test_{tag}_{m}" for tag in ("best", "last") for m in ("accuracy", "f1", "precision", "recall")]
    print(f"\n=== {cfg['experiment']['name']}  ({len(per_seed)} seeds) ===")
    print(table[["best_epoch", "stopped_epoch", "val_acc_best", *cols]].round(4).to_string())
    print(f"paper ResNet50 (single best run): {PAPER_RESNET50}")
    return table


def run_seeds(cfg: Dict[str, Any], seeds: Optional[List[int]] = None, use_wandb: bool = True) -> pd.DataFrame:
    seeds = seeds if seeds is not None else cfg["experiment"].get("seeds", [0, 1, 2])
    return report(cfg, [run_experiment(cfg, s, use_wandb) for s in seeds])


def run_seeds_parallel(
    config_path: str,
    seeds: Optional[List[int]] = None,
    use_wandb: bool = True,
    poll_s: int = 60,
) -> pd.DataFrame:
    """
    One worker process per visible GPU, seeds assigned round-robin
    (T4 x2, seeds 0 1 2 -> GPU0: 0, 2 · GPU1: 1). Each worker sees a single GPU
    through CUDA_VISIBLE_DEVICES, so a seed gives the same run as in-process.
    Worker output goes to <results>/<experiment>/logs/gpu<i>.log; the last
    line of each log is printed every `poll_s` seconds.
    With 0-1 GPUs it falls back to run_seeds().
    """
    cfg = load_config(config_path)
    seeds = seeds if seeds is not None else cfg["experiment"].get("seeds", [0, 1, 2])
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 1 or len(seeds) == 1:
        return run_seeds(cfg, seeds, use_wandb)

    assignment = {g: seeds[g::n_gpu] for g in range(min(n_gpu, len(seeds)))}
    log_dir = os.path.join(_results_dir(cfg), "logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
    for gpu, gpu_seeds in assignment.items():
        cmd = [sys.executable, os.path.abspath(__file__), "--config", cfg["_config_path"],
               "--seeds", *map(str, gpu_seeds), "--worker"]
        if not use_wandb:
            cmd.append("--no-wandb")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        log_path = os.path.join(log_dir, f"gpu{gpu}.log")
        with open(log_path, "w") as log_file:   # the child keeps its own copy of the fd
            procs[gpu] = (subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_path)
        print(f"[parallel] GPU{gpu}: seeds {gpu_seeds}  log {log_path}")

    def last_line(path: str) -> str:
        with open(path, errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else "(no output yet)"

    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(poll_s)
            print(f"[parallel] {(time.time() - t0) / 60:5.1f} min")
            for gpu, (p, log) in procs.items():
                state = "running" if p.poll() is None else f"exit {p.returncode}"
                print(f"  GPU{gpu} [{state}] {last_line(log)}")
    except KeyboardInterrupt:
        for p, _ in procs.values():
            p.terminate()
        raise

    failed = {gpu: log for gpu, (p, log) in procs.items() if p.returncode != 0}
    if failed:
        tails = "\n".join(f"--- GPU{g} ({log}) ---\n" + "".join(open(log, errors="replace").readlines()[-30:])
                          for g, log in failed.items())
        raise RuntimeError(f"Worker(s) failed:\n{tails}")

    print(f"[parallel] done in {(time.time() - t0) / 60:.1f} min")
    return report(cfg, load_seed_results(cfg, seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one config over several seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="one process per GPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # used by run_seeds_parallel
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if args.parallel:
        run_seeds_parallel(args.config, args.seeds, use_wandb)
        return
    cfg = load_config(args.config)
    if args.worker:
        for s in args.seeds:
            run_experiment(cfg, s, use_wandb)
        return
    run_seeds(cfg, args.seeds, use_wandb)


if __name__ == "__main__":
    main()
