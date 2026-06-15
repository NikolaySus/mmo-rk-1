from __future__ import annotations

import copy
import csv
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from experiment import (
    BASE_EMBEDDING_DIM,
    DATASETS,
    DEVICE,
    LOSS_TO_VIEW,
    MODEL_NAME,
    TrainConfig,
    build_experiment_plan,
    choose_primary_metric,
    collect_texts,
    compute_loss,
    embed_texts,
    evaluate_epoch_loss,
    evaluate_identity_baselines,
    evaluate_stsb,
    evaluate_view,
    load_or_build_base_embeddings,
    make_batches,
    plot_results,
    read_jsonl,
    set_seed,
    write_json,
)


DATA_DIR = Path("data") / "processed"
ARTIFACTS_DIR = Path("artifacts")
OUTPUT_JSON = ARTIFACTS_DIR / "next_early_stop_results.json"
OUTPUT_CSV = ARTIFACTS_DIR / "next_early_stop_summary.csv"
MAIN_RESULTS_JSON = ARTIFACTS_DIR / "results.json"
MAIN_METRICS_CSV = ARTIFACTS_DIR / "metrics_summary.csv"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOSS_ORDER = ["contrastive", "triplet", "infonce", "nt_xent", "supcon", "circle"]

PATIENCE = 2
MIN_DELTA = 1e-4
TRAIN_ROWS = 4_000
VAL_ROWS = 1_000

NEXT_CONFIGS: dict[tuple[str, str], dict[str, Any]] = {
    ("all_nli", "contrastive"): {"max_epochs": 12, "learning_rate": 0.003, "weight_decay": 0.0001, "contrastive_margin": 0.6},
    ("all_nli", "triplet"): {"max_epochs": 8, "learning_rate": 0.01, "weight_decay": 0.0, "triplet_margin": 0.2},
    ("all_nli", "infonce"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0, "temperature": 0.05},
    ("all_nli", "nt_xent"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0, "temperature": 0.05},
    ("quora_duplicates", "contrastive"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.001, "contrastive_margin": 0.6},
    ("quora_duplicates", "triplet"): {"max_epochs": 8, "learning_rate": 0.003, "weight_decay": 0.0, "triplet_margin": 0.1},
    ("quora_duplicates", "infonce"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0, "temperature": 0.1},
    ("quora_duplicates", "nt_xent"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0, "temperature": 0.03},
    ("banking77", "contrastive"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.001, "contrastive_margin": 0.4},
    ("banking77", "triplet"): {"max_epochs": 10, "learning_rate": 0.001, "weight_decay": 0.0, "triplet_margin": 0.2},
    ("banking77", "infonce"): {"max_epochs": 12, "learning_rate": 0.003, "weight_decay": 0.0, "temperature": 0.05},
    ("banking77", "nt_xent"): {"max_epochs": 12, "learning_rate": 0.01, "weight_decay": 0.001, "temperature": 0.05},
    ("banking77", "supcon"): {"max_epochs": 12, "learning_rate": 0.003, "weight_decay": 0.0, "temperature": 0.05},
    ("banking77", "circle"): {"max_epochs": 10, "learning_rate": 0.003, "weight_decay": 0.0, "circle_margin": 0.25},
    ("twenty_newsgroups", "contrastive"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0, "contrastive_margin": 0.2},
    ("twenty_newsgroups", "triplet"): {"max_epochs": 10, "learning_rate": 0.01, "weight_decay": 0.0, "triplet_margin": 0.2},
    ("twenty_newsgroups", "infonce"): {"max_epochs": 8, "learning_rate": 0.01, "weight_decay": 0.0, "temperature": 0.03},
    ("twenty_newsgroups", "nt_xent"): {"max_epochs": 8, "learning_rate": 0.01, "weight_decay": 0.0, "temperature": 0.03},
    ("twenty_newsgroups", "supcon"): {"max_epochs": 12, "learning_rate": 0.001, "weight_decay": 0.0001, "temperature": 0.1},
    ("twenty_newsgroups", "circle"): {"max_epochs": 10, "learning_rate": 0.001, "weight_decay": 0.0, "circle_margin": 0.25},
}


def main() -> None:
    base_config = TrainConfig(batch_size=256)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plan = build_experiment_plan()
    text_index = collect_texts(DATA_DIR, plan)
    embeddings = load_or_build_base_embeddings(DATA_DIR, ARTIFACTS_DIR, text_index, base_config)
    baseline = evaluate_identity_baselines(DATA_DIR, embeddings, text_index)

    runs = []
    for dataset_id, loss_name, view in plan:
        next_values = NEXT_CONFIGS[(dataset_id, loss_name)]
        config = config_from_next(base_config, next_values)
        run = train_with_early_stopping(DATA_DIR, embeddings, text_index, dataset_id, loss_name, view, config)
        run["next_config"] = next_values
        runs.append(run)
        print(
            f"{dataset_id:18s} {loss_name:12s} "
            f"best_epoch={run['best_epoch']:2d} val_{run['primary_metric_name']}={run['best_val_metric']:.4f} "
            f"test={run['test_primary_metric']:.4f}"
        )

    result = {
        "config": asdict(base_config),
        "model": {
            "name": MODEL_NAME,
            "architecture": "DistilBERT encoder + mean pooling + Dense(768, 768, bias=False)",
            "training_mode": "Frozen transformer and pooling; Dense projection head reinitialized and trained.",
        },
        "datasets": DATASETS,
        "loss_to_view": LOSS_TO_VIEW,
        "baseline": baseline,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "train_rows": TRAIN_ROWS,
        "val_rows": VAL_ROWS,
        "runs": runs,
    }
    write_json(OUTPUT_JSON, result)
    write_csv(OUTPUT_CSV, runs)
    write_json(MAIN_RESULTS_JSON, result)
    write_csv(MAIN_METRICS_CSV, runs)
    plot_results(PLOTS_DIR / "primary_metrics.png", result)
    plot_dataset_loss_panels(PLOTS_DIR, runs)
    plot_loss_function_panels(PLOTS_DIR, runs)


def config_from_next(base_config: TrainConfig, values: dict[str, Any]) -> TrainConfig:
    overrides = {key: value for key, value in values.items() if key != "max_epochs"}
    return replace(base_config, epochs=int(values["max_epochs"]), **overrides)


def train_with_early_stopping(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    loss_name: str,
    view: str,
    config: TrainConfig,
) -> dict[str, Any]:
    rows = list(read_jsonl(data_dir / dataset_id / view / "train.jsonl"))
    train_rows, val_rows = split_train_val(rows, dataset_id, loss_name)
    test_rows = list(read_jsonl(data_dir / dataset_id / view / "test.jsonl"))

    set_seed(config.seed)
    head = torch.nn.Linear(BASE_EMBEDDING_DIM, config.projection_dim, bias=False).to(DEVICE)
    torch.nn.init.xavier_uniform_(head.weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_metric = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        losses = []
        for batch in make_batches(train_rows, config.batch_size, config.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(loss_name, batch, embeddings, text_index, head, config)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_metrics = evaluate_train_primary(data_dir, embeddings, text_index, dataset_id, view, loss_name, train_rows, head)
        train_metric_name, train_metric = choose_primary_metric(loss_name, train_metrics)
        val_metrics = evaluate_primary(data_dir, embeddings, text_index, dataset_id, view, loss_name, val_rows, train_rows, head)
        val_metric_name, val_metric = choose_primary_metric(loss_name, val_metrics)
        if train_metric_name != val_metric_name:
            raise ValueError(f"Train/val metric mismatch: {train_metric_name} != {val_metric_name}")
        val_loss = evaluate_epoch_loss(loss_name, val_rows, embeddings, text_index, head, config)
        train_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metric": float(train_metric),
                "val_metric": float(val_metric),
            }
        )

        if val_metric > best_metric + MIN_DELTA:
            best_metric = float(val_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    test_metrics = evaluate_primary(data_dir, embeddings, text_index, dataset_id, view, loss_name, test_rows, train_rows, head)
    test_metric_name, test_metric = choose_primary_metric(loss_name, test_metrics)
    return {
        "dataset_id": dataset_id,
        "loss": loss_name,
        "view": view,
        "primary_metric_name": test_metric_name,
        "primary_metric": float(test_metric),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_val_metric": best_metric,
        "test_primary_metric": float(test_metric),
        "test_metrics": test_metrics,
        "stsb_metrics": evaluate_stsb(data_dir, embeddings, text_index, head),
        "final_train_loss": history[best_epoch - 1]["train_loss"] if best_epoch else history[-1]["train_loss"],
        "final_val_loss": history[best_epoch - 1]["val_loss"] if best_epoch else history[-1]["val_loss"],
        "final_train_metric": history[best_epoch - 1]["train_metric"] if best_epoch else history[-1]["train_metric"],
        "final_val_metric": history[best_epoch - 1]["val_metric"] if best_epoch else history[-1]["val_metric"],
        "history": history,
        "config": asdict(config),
    }


def split_train_val(rows: list[dict[str, Any]], dataset_id: str, loss_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(f"{dataset_id}:{loss_name}:val").shuffle(shuffled)
    return shuffled[:TRAIN_ROWS], shuffled[TRAIN_ROWS : TRAIN_ROWS + VAL_ROWS]


def evaluate_primary(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    view: str,
    loss_name: str,
    rows: list[dict[str, Any]],
    train_reference_rows: list[dict[str, Any]],
    head: torch.nn.Module,
) -> dict[str, float]:
    if view == "class_text":
        return evaluate_class_text(embeddings, text_index, rows, train_reference_rows, head)
    _ = loss_name
    return evaluate_view(data_dir, embeddings, text_index, dataset_id, view, rows, head)


def evaluate_train_primary(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    view: str,
    loss_name: str,
    train_rows: list[dict[str, Any]],
    head: torch.nn.Module,
) -> dict[str, float]:
    if view == "class_text":
        return evaluate_class_text_leave_one_out(embeddings, text_index, train_rows, head)
    return evaluate_primary(data_dir, embeddings, text_index, dataset_id, view, loss_name, train_rows, train_rows, head)


def evaluate_class_text(
    embeddings: np.ndarray,
    text_index: dict[str, int],
    rows: list[dict[str, Any]],
    train_reference_rows: list[dict[str, Any]],
    head: torch.nn.Module,
) -> dict[str, float]:
    train_x = embed_texts([row["text"] for row in train_reference_rows], embeddings, text_index, head)
    train_y = np.array([int(row["label"]) for row in train_reference_rows])
    x = embed_texts([row["text"] for row in rows], embeddings, text_index, head)
    y = np.array([int(row["label"]) for row in rows])
    similarity = x @ train_x.T
    pred = train_y[np.argmax(similarity, axis=1)]
    return {"knn1_accuracy": float(accuracy_score(y, pred))}


def evaluate_class_text_leave_one_out(
    embeddings: np.ndarray,
    text_index: dict[str, int],
    rows: list[dict[str, Any]],
    head: torch.nn.Module,
) -> dict[str, float]:
    x = embed_texts([row["text"] for row in rows], embeddings, text_index, head)
    y = np.array([int(row["label"]) for row in rows])
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    pred = y[np.argmax(similarity, axis=1)]
    return {"knn1_accuracy": float(accuracy_score(y, pred))}


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [
        {
            "dataset_id": run["dataset_id"],
            "loss": run["loss"],
            "view": run["view"],
            "primary_metric_name": run["primary_metric_name"],
            "primary_metric": run["primary_metric"],
            "best_epoch": run["best_epoch"],
            "epochs_ran": run["epochs_ran"],
            "best_val_metric": run["best_val_metric"],
            "test_primary_metric": run["test_primary_metric"],
            "final_train_loss": run["final_train_loss"],
            "final_val_loss": run["final_val_loss"],
            "final_train_metric": run["final_train_metric"],
            "final_val_metric": run["final_val_metric"],
            "stsb_spearman": run["stsb_metrics"]["spearman"],
            **run["next_config"],
        }
        for run in runs
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset_loss_panels(plots_dir: Path, runs: list[dict[str, Any]]) -> None:
    for dataset_id in DATASETS:
        dataset_runs = [run for run in runs if run["dataset_id"] == dataset_id]
        pair_columns = 2
        rows = int(np.ceil(len(dataset_runs) / pair_columns))
        fig, axes = plt.subplots(rows, pair_columns * 2, figsize=(13, 3.0 * rows), squeeze=False)
        for run_index, run in enumerate(dataset_runs):
            row_idx = run_index // pair_columns
            col_idx = (run_index % pair_columns) * 2
            loss_axis = axes[row_idx][col_idx]
            metric_axis = axes[row_idx][col_idx + 1]
            epochs = [item["epoch"] for item in run["history"]]
            train_losses = [item["train_loss"] for item in run["history"]]
            val_losses = [item["val_loss"] for item in run["history"]]
            train_metrics = [item["train_metric"] for item in run["history"]]
            val_metrics = [item["val_metric"] for item in run["history"]]
            loss_axis.plot(epochs, train_losses, marker="o", linewidth=1.3, label="train")
            loss_axis.plot(epochs, val_losses, marker="s", linewidth=1.3, label="val")
            loss_axis.axvline(
                run["best_epoch"],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="best target metric",
            )
            loss_axis.set_title(f"{run['loss']} loss", fontsize=9)
            loss_axis.set_xlabel("epoch", fontsize=8)
            loss_axis.set_ylabel("loss", fontsize=8)
            loss_axis.tick_params(axis="both", labelsize=8)
            loss_axis.grid(alpha=0.25)

            metric_axis.plot(epochs, train_metrics, marker="o", linewidth=1.3, label="train")
            metric_axis.plot(epochs, val_metrics, marker="s", linewidth=1.3, label="val")
            metric_axis.axvline(
                run["best_epoch"],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="best target metric",
            )
            metric_axis.set_title(f"{run['loss']} {run['primary_metric_name']}", fontsize=9)
            metric_axis.set_xlabel("epoch", fontsize=8)
            metric_axis.set_ylabel("target metric", fontsize=8)
            metric_axis.tick_params(axis="both", labelsize=8)
            metric_axis.grid(alpha=0.25)
        used_axes = len(dataset_runs) * 2
        for axis in axes.ravel()[used_axes:]:
            axis.axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(plots_dir / f"loss_panels_{dataset_id}.png", dpi=180)
        plt.close(fig)


def plot_loss_function_panels(plots_dir: Path, runs: list[dict[str, Any]]) -> None:
    for loss_name in LOSS_ORDER:
        loss_runs = [run for run in runs if run["loss"] == loss_name]
        if not loss_runs:
            continue
        fig, axes = plt.subplots(len(loss_runs), 2, figsize=(10, 2.35 * len(loss_runs)), squeeze=False)
        for row_idx, run in enumerate(loss_runs):
            loss_axis = axes[row_idx][0]
            metric_axis = axes[row_idx][1]
            epochs = [item["epoch"] for item in run["history"]]
            train_losses = [item["train_loss"] for item in run["history"]]
            val_losses = [item["val_loss"] for item in run["history"]]
            train_metrics = [item["train_metric"] for item in run["history"]]
            val_metrics = [item["val_metric"] for item in run["history"]]
            dataset_title = short_dataset(run["dataset_id"])

            loss_axis.plot(epochs, train_losses, marker="o", linewidth=1.3, label="train")
            loss_axis.plot(epochs, val_losses, marker="s", linewidth=1.3, label="val")
            loss_axis.axvline(
                run["best_epoch"],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="best target metric",
            )
            loss_axis.set_title(f"{dataset_title}: loss", fontsize=9)
            loss_axis.set_xlabel("epoch", fontsize=8)
            loss_axis.set_ylabel("loss", fontsize=8)
            loss_axis.tick_params(axis="both", labelsize=8)
            loss_axis.grid(alpha=0.25)

            metric_axis.plot(epochs, train_metrics, marker="o", linewidth=1.3, label="train")
            metric_axis.plot(epochs, val_metrics, marker="s", linewidth=1.3, label="val")
            metric_axis.axvline(
                run["best_epoch"],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="best target metric",
            )
            metric_axis.set_title(f"{dataset_title}: {run['primary_metric_name']}", fontsize=9)
            metric_axis.set_xlabel("epoch", fontsize=8)
            metric_axis.set_ylabel("target metric", fontsize=8)
            metric_axis.tick_params(axis="both", labelsize=8)
            metric_axis.grid(alpha=0.25)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / f"loss_panels_by_loss_{loss_name}.png", dpi=180)
        plt.close(fig)


def short_dataset(dataset_id: str) -> str:
    return {
        "all_nli": "NLI",
        "quora_duplicates": "Quora",
        "banking77": "Banking77",
        "twenty_newsgroups": "20News",
    }.get(dataset_id, dataset_id)


if __name__ == "__main__":
    main()
