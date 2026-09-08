from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from common import (
    GROUP_NAMES,
    DATA_DIR,
    LEVELS,
    METRICS_DIR,
    ensure_dirs,
    feature_without_group,
    group_block,
    group_categories,
    grouped_feature_matrix,
    one_hot_to_categories,
    safe_torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    parser.add_argument("--max_pairs_per_value", type=int, default=50000)
    parser.add_argument("--classifier_epochs", type=int, default=20)
    parser.add_argument("--classifier_lr", type=float, default=5e-2)
    parser.add_argument("--classifier_weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mean_within_cosine_distance(
    embeddings: torch.Tensor,
    concept_values: np.ndarray,
    max_pairs_per_value: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    x = normalize(embeddings.detach().cpu().numpy().astype(np.float32), axis=1)

    weighted_sum = 0.0
    total_pairs = 0
    for value in np.unique(concept_values):
        idx = np.flatnonzero(concept_values == value)
        if idx.size < 2:
            continue
        n_pairs = idx.size * (idx.size - 1) // 2
        n_sample = min(max_pairs_per_value, n_pairs)

        left = rng.choice(idx, size=n_sample, replace=True)
        right = rng.choice(idx, size=n_sample, replace=True)
        keep = left != right
        if keep.sum() == 0:
            continue
        left = left[keep]
        right = right[keep]

        dist = 1.0 - np.sum(x[left] * x[right], axis=1)
        weighted_sum += float(dist.mean()) * n_pairs
        total_pairs += n_pairs

    if total_pairs == 0:
        return float("nan")
    return weighted_sum / total_pairs


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> float:
    torch.manual_seed(seed)
    classes = np.unique(y_train.astype(np.int64))
    y_train_idx = np.searchsorted(classes, y_train.astype(np.int64))

    x_train_t = torch.as_tensor(x_train.astype(np.float32), device=device)
    y_train_t = torch.as_tensor(y_train_idx.astype(np.int64), device=device)
    x_test_t = torch.as_tensor(x_test.astype(np.float32), device=device)

    model = torch.nn.Linear(x_train_t.shape[1], len(classes), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_idx = model(x_test_t).argmax(dim=1).detach().cpu().numpy()
    pred = classes[pred_idx]
    return accuracy_score(y_test.astype(np.int64), pred) * 100.0


def main() -> None:
    args = parse_args()
    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = []
    independent_rows = []
    leave_one_rows = []

    for level in args.levels:
        train = safe_torch_load(args.data_dir / f"train_{level:03d}.pt")
        test = safe_torch_load(args.data_dir / f"test_{level:03d}.pt")

        train_concepts = train["concepts"]
        test_concepts = test["concepts"]
        y_train = train["labels"].long().numpy()
        y_test = test["labels"].long().numpy()

        train_cats = one_hot_to_categories(train_concepts)

        for group_idx, group in enumerate(GROUP_NAMES):
            geometry_rows.append(
                {
                    "level": level,
                    "concept": group,
                    "within_cosine_distance": mean_within_cosine_distance(
                        train["embeddings"],
                        group_categories(train_cats, group),
                        args.max_pairs_per_value,
                        seed=args.seed + level + group_idx,
                    ),
                }
            )

            acc = train_classifier(
                group_block(train_concepts, group),
                y_train,
                group_block(test_concepts, group),
                y_test,
                seed=args.seed + level + group_idx,
                epochs=args.classifier_epochs,
                lr=args.classifier_lr,
                weight_decay=args.classifier_weight_decay,
                device=args.device,
            )
            independent_rows.append(
                {
                    "level": level,
                    "concept": group,
                    "identity_accuracy_pct": acc,
                }
            )

        full_acc = train_classifier(
            grouped_feature_matrix(train_concepts),
            y_train,
            grouped_feature_matrix(test_concepts),
            y_test,
            seed=args.seed + level,
            epochs=args.classifier_epochs,
            lr=args.classifier_lr,
            weight_decay=args.classifier_weight_decay,
            device=args.device,
        )
        for group_idx, group in enumerate(GROUP_NAMES):
            ablated_acc = train_classifier(
                feature_without_group(train_concepts, group),
                y_train,
                feature_without_group(test_concepts, group),
                y_test,
                seed=args.seed + level + group_idx + 1000,
                epochs=args.classifier_epochs,
                lr=args.classifier_lr,
                weight_decay=args.classifier_weight_decay,
                device=args.device,
            )
            leave_one_rows.append(
                {
                    "level": level,
                    "concept": group,
                    "full_accuracy_pct": full_acc,
                    "ablated_accuracy_pct": ablated_acc,
                    "delta_pct": full_acc - ablated_acc,
                }
            )

    pd.DataFrame(geometry_rows).to_csv(args.out_dir / "geometry.csv", index=False)
    pd.DataFrame(independent_rows).to_csv(args.out_dir / "single_concept_identity.csv", index=False)
    pd.DataFrame(leave_one_rows).to_csv(args.out_dir / "leave_one_out.csv", index=False)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
