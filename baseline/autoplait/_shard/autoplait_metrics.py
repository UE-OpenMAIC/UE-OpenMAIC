from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable


def comb2(n: int) -> float:
    return 0.0 if n < 2 else n * (n - 1) / 2.0


def adjusted_rand_index(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("ARI inputs must have equal length")
    if len(true_list) < 2:
        return 1.0
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(true_list, pred_list))
    except Exception:
        n = len(true_list)
        contingency = defaultdict(int)
        true_counts = Counter()
        pred_counts = Counter()
        for truth, pred in zip(true_list, pred_list):
            contingency[(truth, pred)] += 1
            true_counts[truth] += 1
            pred_counts[pred] += 1
        sum_cells = sum(comb2(v) for v in contingency.values())
        sum_true = sum(comb2(v) for v in true_counts.values())
        sum_pred = sum(comb2(v) for v in pred_counts.values())
        total = comb2(n)
        expected = (sum_true * sum_pred) / total if total else 0.0
        max_index = 0.5 * (sum_true + sum_pred)
        denom = max_index - expected
        if denom == 0:
            return 1.0 if sum_cells == max_index else 0.0
        return float((sum_cells - expected) / denom)


def normalized_mutual_information(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("NMI inputs must have equal length")
    if not true_list:
        return 1.0
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(true_list, pred_list, average_method="geometric"))
    except Exception:
        n = len(true_list)
        joint = Counter(zip(true_list, pred_list))
        true_counts = Counter(true_list)
        pred_counts = Counter(pred_list)
        mi = 0.0
        for (truth, pred), count in joint.items():
            pxy = count / n
            px = true_counts[truth] / n
            py = pred_counts[pred] / n
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py))
        def entropy(counts):
            out = 0.0
            for count in counts:
                p = count / n
                if p > 0:
                    out -= p * math.log(p)
            return out
        ht = entropy(true_counts.values())
        hp = entropy(pred_counts.values())
        if ht == 0.0 and hp == 0.0:
            return 1.0
        if ht == 0.0 or hp == 0.0:
            return 0.0
        return float(mi / math.sqrt(ht * hp))


def adjusted_mutual_information(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        return float(adjusted_mutual_info_score(list(labels_true), list(labels_pred)))
    except Exception:
        return float("nan")


def segments_from_labels(seq) -> list[tuple[int, int, int]]:
    seq = [int(x) for x in seq]
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
    seq = [int(x) for x in seq]
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


def cp_f1(labels_true, labels_pred, margin: int) -> float:
    true_cps = change_points(labels_true)
    pred_cps = change_points(labels_pred)
    if not true_cps and not pred_cps:
        return 1.0
    if not true_cps or not pred_cps:
        return 0.0
    used = set()
    tp = 0
    for p in pred_cps:
        best_j = None
        best_dist = None
        for j, t in enumerate(true_cps):
            if j in used:
                continue
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_j = j
        if best_j is not None:
            used.add(best_j)
            tp += 1
    precision = tp / len(pred_cps) if pred_cps else 0.0
    recall = tp / len(true_cps) if true_cps else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
