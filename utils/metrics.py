"""指标统计工具。

提供 IoU 计算、逐类别 AP、mAP 等评估函数的轻量实现，
供 eval.py 在不依赖外部评估库时使用；同时兼容 ultralytics
返回的指标字典。

主要函数:
    - compute_iou: 单对框 IoU
    - match_predictions: 贪心匹配预测框与真值框
    - compute_ap: 由 PR 点计算 Average Precision
    - summarize_metrics: 汇总多项指标
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 xyxy 框的 IoU。

    Args:
        box1: [x1, y1, x2, y2]。
        box2: [x1, y1, x2, y2]。

    Returns:
        IoU，范围 [0, 1]。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_predictions(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    pred_labels: list[int],
    gt_boxes: list[list[float]],
    gt_labels: list[int],
    iou_threshold: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """贪心匹配预测框与真值框（按类别）。

    Args:
        pred_boxes: 预测框列表（xyxy）。
        pred_scores: 预测置信度。
        pred_labels: 预测类别。
        gt_boxes: 真值框。
        gt_labels: 真值类别。
        iou_threshold: 匹配 IoU 阈值。

    Returns:
        (tp, fp) —— 长度与预测数相同，元素为是否真阳/假阳。
    """
    tp: list[bool] = [False] * len(pred_boxes)
    fp: list[bool] = [False] * len(pred_boxes)

    # 按分数降序排序
    order = np.argsort(-np.asarray(pred_scores))
    matched_gt: set[int] = set()

    for idx in order:
        pred_box = pred_boxes[idx]
        pred_label = pred_labels[idx]
        best_iou = 0.0
        best_gt = -1

        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt or gt_label != pred_label:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_gt >= 0 and best_iou >= iou_threshold:
            tp[idx] = True
            matched_gt.add(best_gt)
        else:
            fp[idx] = True

    return tp, fp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """由 PR 曲线计算 Average Precision（VOC 11 点插值）。

    Args:
        recall: 升序 recall 序列。
        precision: 对应 precision 序列。

    Returns:
        AP 值。
    """
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # 保证 precision 单调不减（右侧最大插值）
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += float((recall[i] - recall[i - 1]) * precision[i])
    return ap


def summarize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """汇总指标字典，过滤无效值并规范化。

    Args:
        metrics: 原始指标字典。

    Returns:
        规范化后的指标字典。
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out