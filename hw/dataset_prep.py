from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset, load_dataset_builder


SEED = 42
TRAIN_SIZE = 5_000
TEST_SIZE = 1_000

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "all_nli": {
        "hf_name": "sentence-transformers/all-nli",
        "kind": "pair_triplet",
        "source_url": "https://huggingface.co/datasets/sentence-transformers/all-nli",
    },
    "quora_duplicates": {
        "hf_name": "sentence-transformers/quora-duplicates",
        "kind": "pair_triplet",
        "source_url": "https://huggingface.co/datasets/sentence-transformers/quora-duplicates",
    },
    "banking77": {
        "hf_name": "mteb/banking77",
        "kind": "class",
        "source_url": "https://huggingface.co/datasets/mteb/banking77",
    },
    "twenty_newsgroups": {
        "hf_name": "SetFit/20_newsgroups",
        "kind": "class",
        "source_url": "https://huggingface.co/datasets/SetFit/20_newsgroups",
    },
    "stsb_eval": {
        "hf_name": "sentence-transformers/stsb",
        "kind": "eval_sts",
        "source_url": "https://huggingface.co/datasets/sentence-transformers/stsb",
    },
}

LOSS_VIEWS: dict[str, str] = {
    "contrastive": "contrastive_pair",
    "triplet": "triplet",
    "infonce": "positive_pair",
    "nt_xent": "positive_pair",
    "supcon": "class_text",
    "circle": "class_text",
}


@dataclass(frozen=True)
class SplitSample:
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]


def inspect_datasets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, spec in DATASET_SPECS.items():
        configs = _configs_for(spec)
        config_names = configs if configs else [None]
        for config in config_names:
            builder = load_dataset_builder(spec["hf_name"], config) if config else load_dataset_builder(spec["hf_name"])
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "hf_name": spec["hf_name"],
                    "kind": spec["kind"],
                    "config": config or "default",
                    "features": ", ".join(builder.info.features.keys()),
                    "splits": "; ".join(
                        f"{split.name}:{split.num_examples}" for split in builder.info.splits.values()
                    ),
                    "source_url": spec["source_url"],
                }
            )
    return rows


def prepare_all(output_dir: Path, train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "seed": SEED,
        "train_size": train_size,
        "test_size": test_size,
        "datasets": {},
        "loss_views": LOSS_VIEWS,
    }

    for dataset_id, spec in DATASET_SPECS.items():
        if spec["kind"] == "eval_sts":
            manifest["datasets"][dataset_id] = _prepare_stsb(dataset_id, spec, output_dir)
        elif spec["kind"] == "class":
            manifest["datasets"][dataset_id] = _prepare_class_dataset(
                dataset_id, spec, output_dir, train_size, test_size
            )
        else:
            manifest["datasets"][dataset_id] = _prepare_pair_triplet_dataset(
                dataset_id, spec, output_dir, train_size, test_size
            )

    _write_json(output_dir / "manifest.json", manifest)
    _write_summary_csv(output_dir / "summary.csv", summarize_outputs(output_dir))
    return manifest


def verify_outputs(output_dir: Path, train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE) -> list[dict[str, Any]]:
    summary = summarize_outputs(output_dir)
    problems: list[str] = []

    by_key = {(row["dataset_id"], row["view"], row["split"]): row for row in summary}
    for dataset_id, spec in DATASET_SPECS.items():
        if spec["kind"] == "eval_sts":
            continue

        expected_views = ["contrastive_pair", "triplet", "positive_pair"]
        if spec["kind"] == "class":
            expected_views.append("class_text")

        for view in expected_views:
            for split, expected_size in (("train", train_size), ("test", test_size)):
                row = by_key.get((dataset_id, view, split))
                if row is None:
                    problems.append(f"missing {dataset_id}/{view}/{split}")
                    continue
                if int(row["rows"]) != expected_size:
                    problems.append(
                        f"{dataset_id}/{view}/{split}: expected {expected_size}, got {row['rows']}"
                    )

        if spec["kind"] == "pair_triplet":
            for view in ("class_text",):
                for split in ("train", "test"):
                    if (dataset_id, view, split) in by_key:
                        problems.append(f"{dataset_id}/{view}/{split}: class view should not be generated")

    if problems:
        raise RuntimeError("Dataset verification failed:\n" + "\n".join(problems))

    return summary


def summarize_outputs(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/*/*.jsonl")):
        dataset_id = path.parts[-3]
        view = path.parts[-2]
        split = path.stem
        records = list(_read_jsonl(path))
        rows.append(
            {
                "dataset_id": dataset_id,
                "view": view,
                "split": split,
                "rows": len(records),
                "columns": ",".join(records[0].keys()) if records else "",
                "labels": _label_summary(records),
            }
        )
    return rows


def write_inspection(output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = inspect_datasets()
    _write_csv(output_dir / "dataset_inspection.csv", rows)
    return rows


def print_table(rows: list[dict[str, Any]], fields: list[str]) -> None:
    widths = {field: len(field) for field in fields}
    for row in rows:
        for field in fields:
            widths[field] = max(widths[field], len(str(row.get(field, ""))))

    header = " | ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("-+-".join("-" * widths[field] for field in fields))
    for row in rows:
        print(" | ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare small text embedding benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory for generated JSONL datasets and reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="Inspect selected Hugging Face datasets.")
    subparsers.add_parser("prepare", help="Generate normalized train/test dataset views.")
    subparsers.add_parser("verify", help="Verify generated dataset views.")
    return parser


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        rows = write_inspection(args.output_dir)
        print_table(rows, ["dataset_id", "config", "kind", "features", "splits"])
    elif args.command == "prepare":
        manifest = prepare_all(args.output_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        rows = verify_outputs(args.output_dir)
        print_table(rows, ["dataset_id", "view", "split", "rows", "labels"])


def _prepare_pair_triplet_dataset(
    dataset_id: str,
    spec: dict[str, Any],
    output_dir: Path,
    train_size: int,
    test_size: int,
) -> dict[str, Any]:
    dataset_dir = output_dir / dataset_id
    hf_name = spec["hf_name"]

    pair_class = _load_pair_class(hf_name)
    pair_sample = _sample_pair_class(pair_class, train_size, test_size)
    pair_sample = _with_split_metadata(pair_sample, dataset_id)
    _write_split(dataset_dir, "contrastive_pair", pair_sample)

    triplets = _load_triplets(hf_name)
    triplet_sample = _sample_records(triplets, train_size, test_size)
    triplet_sample = _with_split_metadata(triplet_sample, dataset_id)
    _write_split(dataset_dir, "triplet", triplet_sample)

    positive_pairs = _load_positive_pairs(hf_name)
    positive_sample = _sample_records(positive_pairs, train_size, test_size)
    positive_sample = _with_split_metadata(positive_sample, dataset_id)
    _write_split(dataset_dir, "positive_pair", positive_sample)

    return {
        "kind": spec["kind"],
        "source_url": spec["source_url"],
        "views": {
            "contrastive_pair": {"train": train_size, "test": test_size},
            "triplet": {"train": train_size, "test": test_size},
            "positive_pair": {"train": train_size, "test": test_size},
        },
        "not_generated": {
            "class_text": "No stable single-text class labels; SupCon/Circle are evaluated on class datasets.",
        },
    }


def _prepare_class_dataset(
    dataset_id: str,
    spec: dict[str, Any],
    output_dir: Path,
    train_size: int,
    test_size: int,
) -> dict[str, Any]:
    dataset_dir = output_dir / dataset_id
    raw = load_dataset(spec["hf_name"])
    train_rows = _dataset_to_records(raw["train"])
    test_rows = _dataset_to_records(raw["test"])

    sampled_train = _stratified_sample(train_rows, train_size, "label", SEED)
    sampled_test = _stratified_sample(test_rows, test_size, "label", SEED + 1)

    class_sample = SplitSample(
        train=[_class_text_record(row, dataset_id, "train") for row in sampled_train],
        test=[_class_text_record(row, dataset_id, "test") for row in sampled_test],
    )
    _write_split(dataset_dir, "class_text", class_sample)

    contrastive_sample = SplitSample(
        train=_class_to_contrastive(sampled_train, dataset_id, "train", train_size, SEED),
        test=_class_to_contrastive(sampled_test, dataset_id, "test", test_size, SEED + 1),
    )
    _write_split(dataset_dir, "contrastive_pair", contrastive_sample)

    triplet_sample = SplitSample(
        train=_class_to_triplets(sampled_train, dataset_id, "train", train_size, SEED),
        test=_class_to_triplets(sampled_test, dataset_id, "test", test_size, SEED + 1),
    )
    _write_split(dataset_dir, "triplet", triplet_sample)

    positive_sample = SplitSample(
        train=_class_to_positive_pairs(sampled_train, dataset_id, "train", train_size, SEED),
        test=_class_to_positive_pairs(sampled_test, dataset_id, "test", test_size, SEED + 1),
    )
    _write_split(dataset_dir, "positive_pair", positive_sample)

    return {
        "kind": spec["kind"],
        "source_url": spec["source_url"],
        "views": {
            "class_text": {"train": train_size, "test": test_size},
            "contrastive_pair": {"train": train_size, "test": test_size},
            "triplet": {"train": train_size, "test": test_size},
            "positive_pair": {"train": train_size, "test": test_size},
        },
    }


def _prepare_stsb(dataset_id: str, spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    dataset_dir = output_dir / dataset_id / "sts_pair"
    raw = load_dataset(spec["hf_name"])
    for split in ("train", "validation", "test"):
        records = [
            {
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": float(row["score"]),
                "source_dataset": dataset_id,
                "split": split,
            }
            for row in raw[split]
        ]
        _write_jsonl(dataset_dir / f"{split}.jsonl", records)

    return {
        "kind": spec["kind"],
        "source_url": spec["source_url"],
        "views": {
            "sts_pair": {
                "train": len(raw["train"]),
                "validation": len(raw["validation"]),
                "test": len(raw["test"]),
            }
        },
        "usage": "Eval-only semantic textual similarity benchmark.",
    }


def _load_pair_class(hf_name: str) -> dict[str, list[dict[str, Any]]]:
    raw = load_dataset(hf_name, "pair-class")
    splits = _ensure_train_test(raw, stratify_by_column="label")
    result: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        result[split] = []
        for row in rows:
            if "premise" in row:
                anchor = row["premise"]
                other = row["hypothesis"]
                label = 1 if int(row["label"]) == 0 else 0
            else:
                anchor = row["sentence1"]
                other = row["sentence2"]
                label = int(row["label"])
            result[split].append({"anchor": anchor, "other": other, "label": label})
    return result


def _load_triplets(hf_name: str) -> dict[str, list[dict[str, Any]]]:
    raw = load_dataset(hf_name, "triplet")
    splits = _ensure_train_test(raw)
    return {
        split: [
            {
                "anchor": row["anchor"],
                "positive": row["positive"],
                "negative": row["negative"],
                "label": 1,
            }
            for row in rows
        ]
        for split, rows in splits.items()
    }


def _load_positive_pairs(hf_name: str) -> dict[str, list[dict[str, Any]]]:
    raw = load_dataset(hf_name, "pair")
    splits = _ensure_train_test(raw)
    return {
        split: [
            {
                "anchor": row["anchor"],
                "positive": row["positive"],
                "label": 1,
            }
            for row in rows
        ]
        for split, rows in splits.items()
    }


def _ensure_train_test(
    raw: dict[str, Dataset],
    stratify_by_column: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if "test" in raw:
        return {"train": _dataset_to_records(raw["train"]), "test": _dataset_to_records(raw["test"])}

    split_kwargs: dict[str, Any] = {"test_size": TEST_SIZE * 2, "seed": SEED, "shuffle": True}
    if stratify_by_column is not None:
        split_kwargs["stratify_by_column"] = stratify_by_column
    split = raw["train"].train_test_split(**split_kwargs)
    return {"train": _dataset_to_records(split["train"]), "test": _dataset_to_records(split["test"])}


def _sample_pair_class(
    records: dict[str, list[dict[str, Any]]],
    train_size: int,
    test_size: int,
) -> SplitSample:
    return SplitSample(
        train=_balanced_binary_sample(records["train"], train_size, SEED),
        test=_balanced_binary_sample(records["test"], test_size, SEED + 1),
    )


def _sample_records(records: dict[str, list[dict[str, Any]]], train_size: int, test_size: int) -> SplitSample:
    return SplitSample(
        train=_random_sample(records["train"], train_size, SEED),
        test=_random_sample(records["test"], test_size, SEED + 1),
    )


def _balanced_binary_sample(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    positives = [row for row in records if int(row["label"]) == 1]
    negatives = [row for row in records if int(row["label"]) == 0]
    pos_size = size // 2
    neg_size = size - pos_size
    if len(positives) < pos_size or len(negatives) < neg_size:
        raise ValueError(f"Not enough positive/negative examples for balanced sample of {size}")
    sampled = _random_sample(positives, pos_size, seed) + _random_sample(negatives, neg_size, seed + 1)
    return _shuffle(sampled, seed + 2)


def _stratified_sample(records: list[dict[str, Any]], size: int, label_key: str, seed: int) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_label[int(row[label_key])].append(row)

    labels = sorted(by_label)
    if size < len(labels):
        raise ValueError(f"Sample size {size} is smaller than number of labels {len(labels)}")

    total = sum(len(by_label[label]) for label in labels)
    quotas = {label: (len(by_label[label]) / total) * size for label in labels}
    targets = {
        label: min(len(by_label[label]), max(2 if len(by_label[label]) >= 2 else 1, int(quotas[label])))
        for label in labels
    }

    while sum(targets.values()) < size:
        candidates = [label for label in labels if targets[label] < len(by_label[label])]
        if not candidates:
            raise ValueError(f"Cannot sample {size} records from {total} available records")
        label = max(candidates, key=lambda item: (quotas[item] - targets[item], len(by_label[item])))
        targets[label] += 1

    while sum(targets.values()) > size:
        candidates = [label for label in labels if targets[label] > 2]
        if not candidates:
            raise ValueError(f"Cannot reduce stratified sample to {size} records")
        label = min(candidates, key=lambda item: (quotas[item] - targets[item], -len(by_label[item])))
        targets[label] -= 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in labels:
        target = targets[label]
        rows = by_label[label]
        if len(rows) < target:
            raise ValueError(f"Label {label} has only {len(rows)} records, need {target}")
        selected.extend(rng.sample(rows, target))
    return _shuffle(selected, seed + 1)


def _class_to_contrastive(
    rows: list[dict[str, Any]],
    dataset_id: str,
    split: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = _group_by_label(rows)
    labels = sorted(by_label)
    records: list[dict[str, Any]] = []

    for index in range(size):
        is_positive = index % 2 == 0
        anchor = rows[index % len(rows)]
        anchor_label = int(anchor["label"])
        if is_positive:
            other = _sample_same_label(by_label, anchor, rng)
            label = 1
        else:
            negative_label = rng.choice([label for label in labels if label != anchor_label])
            other = rng.choice(by_label[negative_label])
            label = 0
        records.append(
            {
                "anchor": anchor["text"],
                "other": other["text"],
                "label": label,
                "source_dataset": dataset_id,
                "split": split,
            }
        )
    return _shuffle(records, seed + 1)


def _class_to_triplets(
    rows: list[dict[str, Any]],
    dataset_id: str,
    split: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = _group_by_label(rows)
    labels = sorted(by_label)
    records: list[dict[str, Any]] = []

    for index in range(size):
        anchor = rows[index % len(rows)]
        anchor_label = int(anchor["label"])
        positive = _sample_same_label(by_label, anchor, rng)
        negative_label = rng.choice([label for label in labels if label != anchor_label])
        negative = rng.choice(by_label[negative_label])
        records.append(
            {
                "anchor": anchor["text"],
                "positive": positive["text"],
                "negative": negative["text"],
                "label": anchor_label,
                "source_dataset": dataset_id,
                "split": split,
            }
        )
    return _shuffle(records, seed + 1)


def _class_to_positive_pairs(
    rows: list[dict[str, Any]],
    dataset_id: str,
    split: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = _group_by_label(rows)
    records: list[dict[str, Any]] = []

    for index in range(size):
        anchor = rows[index % len(rows)]
        positive = _sample_same_label(by_label, anchor, rng)
        records.append(
            {
                "anchor": anchor["text"],
                "positive": positive["text"],
                "label": int(anchor["label"]),
                "source_dataset": dataset_id,
                "split": split,
            }
        )
    return _shuffle(records, seed + 1)


def _class_text_record(row: dict[str, Any], dataset_id: str, split: str) -> dict[str, Any]:
    return {
        "text": row["text"],
        "label": int(row["label"]),
        "label_text": row.get("label_text", str(row["label"])),
        "source_dataset": dataset_id,
        "split": split,
    }


def _sample_same_label(
    by_label: dict[int, list[dict[str, Any]]],
    anchor: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    candidates = by_label[int(anchor["label"])]
    if len(candidates) < 2:
        raise ValueError(f"Label {anchor['label']} has fewer than 2 records")
    while True:
        candidate = rng.choice(candidates)
        if candidate["text"] != anchor["text"]:
            return candidate


def _group_by_label(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    return by_label


def _random_sample(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < size:
        raise ValueError(f"Need {size} records, got {len(records)}")
    return random.Random(seed).sample(records, size)


def _shuffle(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    records = list(records)
    random.Random(seed).shuffle(records)
    return records


def _dataset_to_records(dataset: Dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def _write_split(dataset_dir: Path, view: str, sample: SplitSample) -> None:
    view_dir = dataset_dir / view
    _write_jsonl(view_dir / "train.jsonl", sample.train)
    _write_jsonl(view_dir / "test.jsonl", sample.test)


def _with_split_metadata(sample: SplitSample, dataset_id: str) -> SplitSample:
    return SplitSample(
        train=[record | {"source_dataset": dataset_id, "split": "train"} for record in sample.train],
        test=[record | {"source_dataset": dataset_id, "split": "test"} for record in sample.test],
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _label_summary(records: list[dict[str, Any]]) -> str:
    if not records or "label" not in records[0]:
        return ""
    counts = Counter(str(row["label"]) for row in records)
    if len(counts) <= 8:
        return ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))
    values = list(counts.values())
    return f"classes={len(counts)};min={min(values)};max={max(values)}"


def _configs_for(spec: dict[str, Any]) -> list[str]:
    if spec["kind"] == "pair_triplet":
        return ["pair", "pair-class", "triplet"]
    return []
