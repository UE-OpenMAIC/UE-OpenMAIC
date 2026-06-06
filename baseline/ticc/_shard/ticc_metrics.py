from __future__ import annotations

import math
from collections import Counter, defaultdict


def adjusted_rand_index(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_true, labels_pred))
    except Exception:
        y = list(labels_true)
        z = list(labels_pred)
        if len(y) != len(z):
            raise ValueError("ARI inputs must have equal length")
        if len(y) < 2:
            return 1.0
        def comb2(n: int) -> float:
            return 0.0 if n < 2 else n * (n - 1) / 2.0
        n = len(y)
        contingency = defaultdict(int); y_counts = Counter(); z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1; z_counts[int(b)] += 1
        sum_cells = sum(comb2(v) for v in contingency.values())
        sum_y = sum(comb2(v) for v in y_counts.values())
        sum_z = sum(comb2(v) for v in z_counts.values())
        total = comb2(n)
        expected = (sum_y * sum_z) / total if total else 0.0
        max_index = 0.5 * (sum_y + sum_z)
        denom = max_index - expected
        if denom == 0:
            return 1.0 if sum_cells == max_index else 0.0
        return float((sum_cells - expected) / denom)


def normalized_mutual_information(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_true, labels_pred, average_method="geometric"))
    except Exception:
        y = list(labels_true); z = list(labels_pred)
        if len(y) != len(z):
            raise ValueError("NMI inputs must have equal length")
        if len(y) == 0:
            return 0.0
        n = len(y)
        contingency = defaultdict(int); y_counts = Counter(); z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1; z_counts[int(b)] += 1
        mi = 0.0
        for (a, b), c in contingency.items():
            mi += (c / n) * math.log((c * n) / (y_counts[a] * z_counts[b]))
        def entropy(counts):
            out = 0.0
            for c in counts:
                p = c / n
                if p > 0:
                    out -= p * math.log(p)
            return out
        hy = entropy(y_counts.values()); hz = entropy(z_counts.values())
        if hy == 0.0 and hz == 0.0:
            return 1.0
        if hy == 0.0 or hz == 0.0:
            return 0.0
        return float(mi / math.sqrt(hy * hz))


def adjusted_mutual_information(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        return float(adjusted_mutual_info_score(labels_true, labels_pred, average_method="arithmetic"))
    except Exception:
        return float("nan")


def segments_from_labels(seq) -> list[tuple[int, int, int]]:
    seq = list(map(int, seq))
    if not seq:
        return []
    out = []
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[start]:
            out.append((start, i, seq[start]))
            start = i
    out.append((start, len(seq), seq[start]))
    return out


def segmentation_covering(labels_true, labels_pred) -> float:
    true_segments = segments_from_labels(labels_true)
    pred_segments = segments_from_labels(labels_pred)
    n = len(labels_true)
    if n == 0 or not true_segments or not pred_segments:
        return 0.0
    total = 0.0
    for ts, te, _ in true_segments:
        best = 0.0
        for ps, pe, _ in pred_segments:
            inter = max(0, min(te, pe) - max(ts, ps))
            if inter <= 0:
                continue
            union = max(te, pe) - min(ts, ps)
            best = max(best, inter / union)
        total += (te - ts) * best
    return float(total / n)


def change_points(seq) -> list[int]:
    seq = list(map(int, seq))
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


def cp_f1(labels_true, labels_pred, margin: int) -> float:
    true_cps = change_points(labels_true)
    pred_cps = change_points(labels_pred)
    if not true_cps and not pred_cps:
        return 1.0
    if not true_cps or not pred_cps:
        return 0.0
    used = set(); tp = 0
    for p in pred_cps:
        best_j = None; best_dist = None
        for j, t in enumerate(true_cps):
            if j in used:
                continue
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best_dist = dist; best_j = j
        if best_j is not None:
            used.add(best_j); tp += 1
    precision = tp / len(pred_cps) if pred_cps else 0.0
    recall = tp / len(true_cps) if true_cps else 0.0
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def compute_metrics(labels, pred, cp_margin_ratio: float) -> dict[str, float]:
    margin = max(1, int(round(len(labels) * float(cp_margin_ratio)))) if len(labels) else 1
    return {
        "ARI": adjusted_rand_index(labels, pred),
        "NMI": normalized_mutual_information(labels, pred),
        "AMI": adjusted_mutual_information(labels, pred),
        "covering_score": segmentation_covering(labels, pred),
        "f1_score": cp_f1(labels, pred, margin),
        "cp_margin": margin,
        "true_cps_count": len(change_points(labels)),
        "pred_cps_count": len(change_points(pred)),
    }
