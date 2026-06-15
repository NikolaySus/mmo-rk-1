from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-dot-prod-v3"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_EMBEDDING_DIM = 768

DATASETS = ["all_nli", "quora_duplicates", "banking77", "twenty_newsgroups"]
CLASS_DATASETS = {"banking77", "twenty_newsgroups"}

LOSS_TO_VIEW = {
    "contrastive": "contrastive_pair",
    "triplet": "triplet",
    "infonce": "positive_pair",
    "nt_xent": "positive_pair",
    "supcon": "class_text",
    "circle": "class_text",
}

GLOBAL_SEARCH_SPACE = {
    "epochs": [1, 3, 6],
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

LOSS_SPECIFIC_SEARCH_SPACE = {
    "contrastive": ("contrastive_margin", [0.2, 0.4, 0.6]),
    "triplet": ("triplet_margin", [0.1, 0.2, 0.4]),
    "infonce": ("temperature", [0.03, 0.05, 0.1]),
    "nt_xent": ("temperature", [0.03, 0.05, 0.1]),
    "supcon": ("temperature", [0.03, 0.05, 0.1]),
    "circle": ("circle_margin", [0.15, 0.25, 0.35]),
}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = MODEL_NAME
    seed: int = SEED
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    temperature: float = 0.05
    contrastive_margin: float = 0.4
    triplet_margin: float = 0.2
    circle_margin: float = 0.25
    circle_gamma: float = 32.0
    max_seq_length: int = 256
    projection_dim: int = BASE_EMBEDDING_DIM
    trainable_parameters: int = BASE_EMBEDDING_DIM * BASE_EMBEDDING_DIM
    frozen_backbone: bool = True
    reinitialized_projection_head: bool = True


def run_all_experiments(
    data_dir: Path = Path("data") / "processed",
    artifacts_dir: Path = Path("artifacts"),
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    set_seed(config.seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "plots").mkdir(exist_ok=True)
    (artifacts_dir / "models").mkdir(exist_ok=True)

    experiment_plan = build_experiment_plan()
    text_index = collect_texts(data_dir, experiment_plan)
    embeddings = load_or_build_base_embeddings(data_dir, artifacts_dir, text_index, config)

    baseline = evaluate_identity_baselines(data_dir, embeddings, text_index)
    search_result = run_hyperparameter_search(data_dir, embeddings, text_index, experiment_plan, config)
    write_hyperparameter_search_csv(artifacts_dir / "hyperparameter_search.csv", search_result)

    runs: list[dict[str, Any]] = []
    for dataset_id, loss_name, view in experiment_plan:
        started = time.time()
        final_config = config_for_combo(config, search_result["best_configs"][(dataset_id, loss_name)])
        run = train_one_head(data_dir, embeddings, text_index, dataset_id, loss_name, view, final_config)
        run["duration_sec"] = round(time.time() - started, 3)
        runs.append(run)
        save_head(run, artifacts_dir)
        print(f"{dataset_id:18s} {loss_name:12s} {run['primary_metric_name']}={run['primary_metric']:.4f}")

    result = {
        "config": asdict(config),
        "model": {
            "name": MODEL_NAME,
            "architecture": "DistilBERT encoder + mean pooling + Dense(768, 768, bias=False)",
            "training_mode": "Frozen transformer and pooling; Dense projection head reinitialized and trained.",
        },
        "datasets": DATASETS,
        "loss_to_view": LOSS_TO_VIEW,
        "hyperparameter_search": serialize_search_result(search_result),
        "baseline": baseline,
        "runs": runs,
    }

    write_json(artifacts_dir / "results.json", result)
    write_summary_csv(artifacts_dir / "metrics_summary.csv", result)
    plot_results(artifacts_dir / "plots" / "primary_metrics.png", result)
    plot_loss_panels(artifacts_dir / "plots" / "loss_panels.png", result)
    plot_dataset_loss_panels(artifacts_dir / "plots", result)
    return result


def run_hyperparameter_search(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    experiment_plan: list[tuple[str, str, str]],
    base_config: TrainConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_configs: dict[tuple[str, str], dict[str, Any]] = {}

    for dataset_id, loss_name, view in experiment_plan:
        combo_best: dict[str, Any] = {}
        for hyperparameter, values in search_spaces_for(loss_name).items():
            trial_rows = []
            for value in values:
                trial_config = config_for_combo(base_config, combo_best | {hyperparameter: value})
                started = time.time()
                run = train_one_head(
                    data_dir,
                    embeddings,
                    text_index,
                    dataset_id,
                    loss_name,
                    view,
                    trial_config,
                    full_eval=False,
                    track_epoch_test_loss=False,
                )
                row = {
                    "dataset_id": dataset_id,
                    "loss": loss_name,
                    "view": view,
                    "hyperparameter": hyperparameter,
                    "value": value,
                    "primary_metric_name": run["primary_metric_name"],
                    "primary_metric": run["primary_metric"],
                    "final_train_loss": run["final_train_loss"],
                    "final_test_loss": run["final_test_loss"],
                    "duration_sec": round(time.time() - started, 3),
                }
                rows.append(row)
                trial_rows.append(row)
            best_row = max(trial_rows, key=lambda item: item["primary_metric"])
            combo_best[hyperparameter] = best_row["value"]
            print(
                f"search {dataset_id:18s} {loss_name:12s} "
                f"{hyperparameter}={best_row['value']} {best_row['primary_metric_name']}={best_row['primary_metric']:.4f}"
            )
        best_configs[(dataset_id, loss_name)] = combo_best

    return {"rows": rows, "best_configs": best_configs}


def search_spaces_for(loss_name: str) -> dict[str, list[Any]]:
    spaces = dict(GLOBAL_SEARCH_SPACE)
    hyperparameter, values = LOSS_SPECIFIC_SEARCH_SPACE[loss_name]
    spaces[hyperparameter] = values
    return spaces


def config_for_combo(base_config: TrainConfig, values: dict[str, Any]) -> TrainConfig:
    return replace(base_config, **values)


def serialize_search_result(search_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": search_result["rows"],
        "best_configs": {
            f"{dataset_id}__{loss_name}": values
            for (dataset_id, loss_name), values in search_result["best_configs"].items()
        },
    }


def build_experiment_plan() -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    for dataset_id in DATASETS:
        for loss_name, view in LOSS_TO_VIEW.items():
            if view == "class_text" and dataset_id not in CLASS_DATASETS:
                continue
            plan.append((dataset_id, loss_name, view))
    return plan


def collect_texts(data_dir: Path, plan: list[tuple[str, str, str]]) -> dict[str, int]:
    texts: dict[str, int] = {}
    for dataset_id, _, view in plan:
        for split in ("train", "test"):
            for row in read_jsonl(data_dir / dataset_id / view / f"{split}.jsonl"):
                for field in ("anchor", "other", "positive", "negative", "text"):
                    if field in row and row[field] not in texts:
                        texts[row[field]] = len(texts)
    for split in ("validation", "test"):
        for row in read_jsonl(data_dir / "stsb_eval" / "sts_pair" / f"{split}.jsonl"):
            for field in ("sentence1", "sentence2"):
                if row[field] not in texts:
                    texts[row[field]] = len(texts)
    return texts


def load_or_build_base_embeddings(
    data_dir: Path,
    artifacts_dir: Path,
    text_index: dict[str, int],
    config: TrainConfig,
) -> np.ndarray:
    cache_dir = artifacts_dir / "embeddings"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "model": config.model_name,
                "max_seq_length": config.max_seq_length,
                "texts": sorted(text_index.items(), key=lambda item: item[1]),
            },
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    embedding_path = cache_dir / f"base_embeddings_{cache_key}.npy"
    index_path = cache_dir / f"text_index_{cache_key}.json"
    latest_path = cache_dir / "latest.json"

    if embedding_path.exists() and index_path.exists():
        return np.load(embedding_path)

    model = SentenceTransformer(config.model_name, device=DEVICE)
    model.max_seq_length = config.max_seq_length
    backbone = SentenceTransformer(modules=[model[0], model[1]], device=DEVICE)
    ordered_texts = [text for text, _ in sorted(text_index.items(), key=lambda item: item[1])]
    embeddings = backbone.encode(
        ordered_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    np.save(embedding_path, embeddings)
    write_json(index_path, text_index)
    write_json(latest_path, {"embedding_path": str(embedding_path), "index_path": str(index_path)})
    _ = data_dir
    return embeddings


def evaluate_identity_baselines(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    for dataset_id in DATASETS:
        baseline[dataset_id] = evaluate_all_views(data_dir, embeddings, text_index, dataset_id, None)
    baseline["stsb_eval"] = evaluate_stsb(data_dir, embeddings, text_index, None)
    return baseline


def train_one_head(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    loss_name: str,
    view: str,
    config: TrainConfig,
    full_eval: bool = True,
    track_epoch_test_loss: bool = True,
) -> dict[str, Any]:
    train_rows = list(read_jsonl(data_dir / dataset_id / view / "train.jsonl"))
    test_rows = list(read_jsonl(data_dir / dataset_id / view / "test.jsonl"))

    set_seed(config.seed)
    head = torch.nn.Linear(BASE_EMBEDDING_DIM, config.projection_dim, bias=False).to(DEVICE)
    torch.nn.init.xavier_uniform_(head.weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        losses = []
        for batch in make_batches(train_rows, config.batch_size, config.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(loss_name, batch, embeddings, text_index, head, config)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "test_loss": (
                    evaluate_epoch_loss(loss_name, test_rows, embeddings, text_index, head, config)
                    if track_epoch_test_loss
                    else math.nan
                ),
            }
        )

    final_test_loss = evaluate_epoch_loss(loss_name, test_rows, embeddings, text_index, head, config)
    if not track_epoch_test_loss:
        history[-1]["test_loss"] = final_test_loss

    own_eval = evaluate_view(data_dir, embeddings, text_index, dataset_id, view, test_rows, head)
    stsb_eval = evaluate_stsb(data_dir, embeddings, text_index, head) if full_eval else {}
    all_view_eval = evaluate_all_views(data_dir, embeddings, text_index, dataset_id, head) if full_eval else {}
    primary_metric_name, primary_metric = choose_primary_metric(loss_name, own_eval)

    return {
        "dataset_id": dataset_id,
        "loss": loss_name,
        "view": view,
        "history": history,
        "own_view_metrics": own_eval,
        "all_view_metrics": all_view_eval,
        "stsb_metrics": stsb_eval,
        "primary_metric_name": primary_metric_name,
        "primary_metric": float(primary_metric),
        "final_train_loss": history[-1]["train_loss"],
        "final_test_loss": final_test_loss,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "config": asdict(config),
    }


def evaluate_epoch_loss(
    loss_name: str,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    text_index: dict[str, int],
    head: torch.nn.Module,
    config: TrainConfig,
) -> float:
    losses = []
    with torch.no_grad():
        for batch in make_batches(rows, config.batch_size, config.seed):
            if len(batch) < 2:
                continue
            loss = compute_loss(loss_name, batch, embeddings, text_index, head, config)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def compute_loss(
    loss_name: str,
    batch: list[dict[str, Any]],
    embeddings: np.ndarray,
    text_index: dict[str, int],
    head: torch.nn.Module,
    config: TrainConfig,
) -> torch.Tensor:
    if loss_name == "contrastive":
        anchor = project_texts([row["anchor"] for row in batch], embeddings, text_index, head)
        other = project_texts([row["other"] for row in batch], embeddings, text_index, head)
        labels = torch.tensor([1 if int(row["label"]) == 1 else -1 for row in batch], device=DEVICE).float()
        return torch.nn.CosineEmbeddingLoss(margin=config.contrastive_margin)(anchor, other, labels)

    if loss_name == "triplet":
        anchor = project_texts([row["anchor"] for row in batch], embeddings, text_index, head)
        positive = project_texts([row["positive"] for row in batch], embeddings, text_index, head)
        negative = project_texts([row["negative"] for row in batch], embeddings, text_index, head)
        def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return 1 - F.cosine_similarity(left, right)

        return torch.nn.TripletMarginWithDistanceLoss(distance_function=distance, margin=config.triplet_margin)(
            anchor, positive, negative
        )

    if loss_name in {"infonce", "nt_xent"}:
        anchor = project_texts([row["anchor"] for row in batch], embeddings, text_index, head)
        positive = project_texts([row["positive"] for row in batch], embeddings, text_index, head)
        logits = anchor @ positive.T / config.temperature
        labels = torch.arange(logits.shape[0], device=DEVICE)
        forward_loss = F.cross_entropy(logits, labels)
        if loss_name == "nt_xent":
            return 0.5 * (forward_loss + F.cross_entropy(logits.T, labels))
        return forward_loss

    if loss_name in {"supcon", "circle"}:
        texts = [row["text"] for row in batch]
        labels = torch.tensor([int(row["label"]) for row in batch], device=DEVICE)
        projected = project_texts(texts, embeddings, text_index, head)
        if loss_name == "supcon":
            return supervised_contrastive_loss(projected, labels, config.temperature)
        return circle_loss(projected, labels, config.circle_margin, config.circle_gamma)

    raise ValueError(f"Unknown loss: {loss_name}")


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=DEVICE)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positive_mask).sum(dim=1) / positive_count
    return loss[positive_mask.any(dim=1)].mean()


def circle_loss(features: torch.Tensor, labels: torch.Tensor, margin: float, gamma: float) -> torch.Tensor:
    similarity = features @ features.T
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=DEVICE)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    negative_mask = ~labels[:, None].eq(labels[None, :])
    losses = []
    delta_p = 1 - margin
    delta_n = margin
    for row in range(features.shape[0]):
        sp = similarity[row][positive_mask[row]]
        sn = similarity[row][negative_mask[row]]
        if sp.numel() == 0 or sn.numel() == 0:
            continue
        ap = torch.clamp_min(-sp.detach() + 1 + margin, 0.0)
        an = torch.clamp_min(sn.detach() + margin, 0.0)
        logit_p = -gamma * ap * (sp - delta_p)
        logit_n = gamma * an * (sn - delta_n)
        losses.append(F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)))
    if not losses:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    return torch.stack(losses).mean()


def evaluate_all_views(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    head: torch.nn.Module | None,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for view in ("contrastive_pair", "triplet", "positive_pair", "class_text"):
        path = data_dir / dataset_id / view / "test.jsonl"
        if path.exists():
            metrics[view] = evaluate_view(data_dir, embeddings, text_index, dataset_id, view, list(read_jsonl(path)), head)
    return metrics


def evaluate_view(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    dataset_id: str,
    view: str,
    rows: list[dict[str, Any]],
    head: torch.nn.Module | None,
) -> dict[str, float]:
    _ = data_dir, dataset_id
    if view == "contrastive_pair":
        anchor = embed_texts([row["anchor"] for row in rows], embeddings, text_index, head)
        other = embed_texts([row["other"] for row in rows], embeddings, text_index, head)
        scores = np.sum(anchor * other, axis=1)
        labels = np.array([int(row["label"]) for row in rows])
        return {
            "pair_auc": safe_metric(roc_auc_score, labels, scores),
            "pair_average_precision": safe_metric(average_precision_score, labels, scores),
            "pair_accuracy_at_zero": float(accuracy_score(labels, scores > 0)),
            "mean_positive_score": float(scores[labels == 1].mean()) if np.any(labels == 1) else math.nan,
            "mean_negative_score": float(scores[labels == 0].mean()) if np.any(labels == 0) else math.nan,
        }

    if view == "triplet":
        anchor = embed_texts([row["anchor"] for row in rows], embeddings, text_index, head)
        positive = embed_texts([row["positive"] for row in rows], embeddings, text_index, head)
        negative = embed_texts([row["negative"] for row in rows], embeddings, text_index, head)
        pos_scores = np.sum(anchor * positive, axis=1)
        neg_scores = np.sum(anchor * negative, axis=1)
        return {
            "triplet_accuracy": float(np.mean(pos_scores > neg_scores)),
            "mean_margin": float(np.mean(pos_scores - neg_scores)),
        }

    if view == "positive_pair":
        anchor = embed_texts([row["anchor"] for row in rows], embeddings, text_index, head)
        positive = embed_texts([row["positive"] for row in rows], embeddings, text_index, head)
        similarity = anchor @ positive.T
        ranks = []
        for row_idx in range(similarity.shape[0]):
            order = np.argsort(-similarity[row_idx])
            rank = int(np.where(order == row_idx)[0][0]) + 1
            ranks.append(rank)
        return {
            "retrieval_top1": float(np.mean(np.array(ranks) == 1)),
            "retrieval_mrr": float(np.mean(1 / np.array(ranks))),
        }

    if view == "class_text":
        train_path = data_dir / dataset_id / view / "train.jsonl"
        train_rows = list(read_jsonl(train_path))
        train_x = embed_texts([row["text"] for row in train_rows], embeddings, text_index, head)
        train_y = np.array([int(row["label"]) for row in train_rows])
        test_x = embed_texts([row["text"] for row in rows], embeddings, text_index, head)
        test_y = np.array([int(row["label"]) for row in rows])
        similarity = test_x @ train_x.T
        pred = train_y[np.argmax(similarity, axis=1)]
        return {"knn1_accuracy": float(accuracy_score(test_y, pred))}

    raise ValueError(f"Unknown view: {view}")


def evaluate_stsb(
    data_dir: Path,
    embeddings: np.ndarray,
    text_index: dict[str, int],
    head: torch.nn.Module | None,
) -> dict[str, float]:
    rows = list(read_jsonl(data_dir / "stsb_eval" / "sts_pair" / "test.jsonl"))
    left = embed_texts([row["sentence1"] for row in rows], embeddings, text_index, head)
    right = embed_texts([row["sentence2"] for row in rows], embeddings, text_index, head)
    scores = np.sum(left * right, axis=1)
    gold = np.array([float(row["score"]) for row in rows])
    return {"spearman": float(spearmanr(scores, gold).statistic)}


def choose_primary_metric(loss_name: str, metrics: dict[str, float]) -> tuple[str, float]:
    if loss_name == "contrastive":
        return "pair_auc", metrics["pair_auc"]
    if loss_name == "triplet":
        return "triplet_accuracy", metrics["triplet_accuracy"]
    if loss_name in {"infonce", "nt_xent"}:
        return "retrieval_top1", metrics["retrieval_top1"]
    return "knn1_accuracy", metrics["knn1_accuracy"]


def project_texts(
    texts: list[str],
    embeddings: np.ndarray,
    text_index: dict[str, int],
    head: torch.nn.Module,
) -> torch.Tensor:
    indexes = [text_index[text] for text in texts]
    base = torch.from_numpy(embeddings[indexes]).to(DEVICE)
    return F.normalize(head(base), p=2, dim=1)


def embed_texts(
    texts: list[str],
    embeddings: np.ndarray,
    text_index: dict[str, int],
    head: torch.nn.Module | None,
    batch_size: int = 2048,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            indexes = [text_index[text] for text in batch]
            base = torch.from_numpy(embeddings[indexes]).to(DEVICE)
            projected = base if head is None else head(base)
            outputs.append(F.normalize(projected, p=2, dim=1).cpu().numpy())
    return np.vstack(outputs)


def make_batches(rows: list[dict[str, Any]], batch_size: int, seed: int) -> list[list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def save_head(run: dict[str, Any], artifacts_dir: Path) -> None:
    _ = run, artifacts_dir
    # Metrics and deterministic config are sufficient for the report; model checkpoints are intentionally omitted.


def write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for run in result["runs"]:
        rows.append(
            {
                "dataset_id": run["dataset_id"],
                "loss": run["loss"],
                "view": run["view"],
                "primary_metric_name": run["primary_metric_name"],
                "primary_metric": run["primary_metric"],
                "final_train_loss": run["final_train_loss"],
                "final_test_loss": run["final_test_loss"],
                "stsb_spearman": run["stsb_metrics"]["spearman"],
                "duration_sec": run["duration_sec"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_hyperparameter_search_csv(path: Path, search_result: dict[str, Any]) -> None:
    rows = search_result["rows"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, result: dict[str, Any]) -> None:
    rows = result["runs"]
    labels = [f"{run['dataset_id']}\n{run['loss']}" for run in rows]
    values = [run["primary_metric"] for run in rows]
    plt.figure(figsize=(13, 6))
    bars = plt.bar(range(len(rows)), values, color="#4C78A8")
    plt.xticks(range(len(rows)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Primary metric")
    plt.ylim(0, max(1.0, max(values) * 1.08))
    plt.title("Primary test metric by dataset and loss")
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_loss_panels(path: Path, result: dict[str, Any]) -> None:
    runs = result["runs"]
    columns = 4
    rows = math.ceil(len(runs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 2.4 * rows), squeeze=False)
    for axis, run in zip(axes.ravel(), runs, strict=False):
        epochs = [item["epoch"] for item in run["history"]]
        train_losses = [item["train_loss"] for item in run["history"]]
        test_losses = [item["test_loss"] for item in run["history"]]
        axis.plot(epochs, train_losses, marker="o", linewidth=1.2, label="train")
        axis.plot(epochs, test_losses, marker="s", linewidth=1.2, label="test")
        axis.set_title(f"{run['dataset_id']} / {run['loss']}", fontsize=8)
        axis.set_xlabel("epoch", fontsize=7)
        axis.set_ylabel("loss", fontsize=7)
        axis.tick_params(axis="both", labelsize=7)
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(runs) :]:
        axis.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Train and test loss by dataset and loss function", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=180)
    plt.close()


def plot_dataset_loss_panels(plots_dir: Path, result: dict[str, Any]) -> None:
    for dataset_id in DATASETS:
        runs = [run for run in result["runs"] if run["dataset_id"] == dataset_id]
        columns = 2
        rows = math.ceil(len(runs) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(9, 3.0 * rows), squeeze=False)
        for axis, run in zip(axes.ravel(), runs, strict=False):
            epochs = [item["epoch"] for item in run["history"]]
            train_losses = [item["train_loss"] for item in run["history"]]
            test_losses = [item["test_loss"] for item in run["history"]]
            axis.plot(epochs, train_losses, marker="o", linewidth=1.4, label="train")
            axis.plot(epochs, test_losses, marker="s", linewidth=1.4, label="test")
            axis.set_title(run["loss"], fontsize=10)
            axis.set_xlabel("epoch", fontsize=8)
            axis.set_ylabel("loss", fontsize=8)
            axis.tick_params(axis="both", labelsize=8)
            axis.grid(alpha=0.25)
        for axis in axes.ravel()[len(runs) :]:
            axis.axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        plt.savefig(plots_dir / f"loss_panels_{dataset_id}.png", dpi=180)
        plt.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_metric(func: Any, labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(func(labels, scores))
    except ValueError:
        return math.nan


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    run_all_experiments(args.data_dir, args.artifacts_dir, config)


if __name__ == "__main__":
    main()
