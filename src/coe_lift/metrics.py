"""Evaluation metrics for CoE-LIFT."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np


def extract_grid(value: str) -> Any:
    text = value.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    payload = json.loads(text)
    return payload.get("grid", payload)


def exact_grid_match(prediction: str, answer: str) -> bool:
    try:
        return extract_grid(prediction) == extract_grid(answer)
    except Exception:
        return prediction.strip() == answer.strip()


def split_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(int(row["correct"]))
    metrics = {}
    for split, values in sorted(by_split.items()):
        metrics[split] = {
            "accuracy": float(np.mean(values)) if values else 0.0,
            "n": len(values),
        }

    id_values = [
        int(row["correct"])
        for row in rows
        if str(row["split"]) in {"train_id", "test_id"} or str(row["split"]).endswith("_id")
    ]
    ood_values = [int(row["correct"]) for row in rows if str(row["split"]).startswith("test_ood")]
    if id_values and ood_values:
        metrics["ood_gap"] = float(np.mean(id_values) - np.mean(ood_values))
    return metrics


def cross_lingual_consistency(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(int(row["group_id"]), int(row.get("surface_seed", 0)))].append(row)

    consistency_scores = []
    correctness_variances = []
    for group_rows in by_group.values():
        if len({row["lang"] for row in group_rows}) <= 1:
            continue
        normalized = []
        for row in group_rows:
            try:
                normalized.append(json.dumps(extract_grid(str(row["prediction"])), sort_keys=True))
            except Exception:
                normalized.append(str(row["prediction"]).strip())
        consistency_scores.append(float(len(set(normalized)) == 1))
        correctness_variances.append(float(np.var([int(row["correct"]) for row in group_rows])))

    return {
        "group_consistency": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
        "correctness_variance": float(np.mean(correctness_variances)) if correctness_variances else 0.0,
        "n_groups": float(len(consistency_scores)),
    }


def expected_calibration_error(
    correctness: list[int],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    if not correctness:
        return 0.0
    y = np.asarray(correctness, dtype=np.float64)
    c = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (c >= low) & (c < high if high < 1.0 else c <= high)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(y[mask].mean() - c[mask].mean()))
    return ece


def brier_score(correctness: list[int], confidences: list[float]) -> float:
    if not correctness:
        return 0.0
    y = np.asarray(correctness, dtype=np.float64)
    c = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((c - y) ** 2))


def pairwise_group_cosine(embeddings: np.ndarray, group_ids: list[int]) -> float:
    if len(embeddings) < 2:
        return 0.0
    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        by_group[int(group_id)].append(idx)
    scores = []
    normed = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                scores.append(float(normed[left] @ normed[right]))
    return float(np.mean(scores)) if scores else 0.0


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(xy / max(xx * yy, 1e-12))


def linear_probe_accuracy(embeddings: np.ndarray, labels: list[str]) -> float | None:
    unique = sorted(set(labels))
    if len(unique) < 2 or len(labels) < 8:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        label_to_id = {label: idx for idx, label in enumerate(unique)}
        y = np.asarray([label_to_id[label] for label in labels])
        stratify = y if min(np.bincount(y)) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            y,
            test_size=0.35,
            random_state=13,
            stratify=stratify,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))
    except Exception:
        return None
