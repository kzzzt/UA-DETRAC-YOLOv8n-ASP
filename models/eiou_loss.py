"""EIOU 边界框回归损失。

EIoU 在 CIoU 基础上，将宽高比惩罚替换为对预测框与真实框
宽、高的直接惩罚，并显式建模中心点距离，收敛更快、定位更准。

参考公式:
    L_EIoU = 1 - IoU + rho^2(b, b_gt)/c^2
             + rho^2(w, w_gt)/Cw^2 + rho^2(h, h_gt)/Ch^2

其中 b/b_gt 为中心点，w/w_gt、h/h_gt 为宽高，
c/Cw/Ch 为最小外接框的对角线、宽、高。

该模块同时提供:
    - EIoULoss: 可独立调用的损失类（输入 xyxy 格式框）。
    - eiou_loss: 函数式接口，便于接入自定义训练循环。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _bbox_xyxy_to_center_wh(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 (x1, y1, x2, y2) 转换为 (cx, cy) 与 (w, h)。"""
    x1, y1, x2, y2 = box.unbind(dim=-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy], dim=-1), torch.stack([w, h], dim=-1)


def _iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """计算两个 xyxy 框之间的 IoU。

    Args:
        box1: (N, 4) 或 (B, N, 4)
        box2: 与 box1 同形状

    Returns:
        IoU, 同 batch 形状（去掉最后一维）。
    """
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


def eiou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """EIoU 损失。

    Args:
        pred: 预测框 (..., 4)，xyxy 格式。
        target: 真实框 (..., 4)，xyxy 格式。
        eps: 防止除零的小量。

    Returns:
        逐框 EIoU 损失，形状 (..., 1)。
    """
    iou = _iou_xyxy(pred, target)

    # 中心点距离
    c_pred, _ = _bbox_xyxy_to_center_wh(pred)
    c_tgt, _ = _bbox_xyxy_to_center_wh(target)
    rho2 = (c_pred - c_tgt).pow(2).sum(dim=-1)  # 中心点距离平方

    # 最小外接框
    x1 = torch.min(pred[..., 0], target[..., 0])
    y1 = torch.min(pred[..., 1], target[..., 1])
    x2 = torch.max(pred[..., 2], target[..., 2])
    y2 = torch.max(pred[..., 3], target[..., 3])
    cw = (x2 - x1).clamp(min=0.0)
    ch = (y2 - y1).clamp(min=0.0)
    c2 = cw.pow(2) + ch.pow(2)  # 对角线平方

    # 宽高直接惩罚
    _, wh_pred = _bbox_xyxy_to_center_wh(pred)
    _, wh_tgt = _bbox_xyxy_to_center_wh(target)
    rho_w = (wh_pred[..., 0] - wh_tgt[..., 0]).pow(2)
    rho_h = (wh_pred[..., 1] - wh_tgt[..., 1]).pow(2)

    loss = (
        1.0 - iou
        + rho2 / (c2 + eps)
        + rho_w / (cw.pow(2) + eps)
        + rho_h / (ch.pow(2) + eps)
    )
    # 关键：保持最后一维，输出 (..., 1)，与 ultralytics 原生 bbox_iou 形状一致。
    # 否则在 BboxLoss 中与 (N,1) 的 weight 相乘会发生 (N,)+(N,1)->(N,N) 广播爆炸，
    # 导致 box_loss 变成上万（实测出现 1.5e4）。
    return loss.unsqueeze(-1)


class EIoULoss(nn.Module):
    """可独立调用的 EIoU 损失模块。"""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 EIoU 损失（标量平均值）。

        Args:
            pred: 预测框 (N, 4) 或 (B, N, 4), xyxy。
            target: 真实框，同形状。

        Returns:
            标量损失。
        """
        return eiou_loss(pred, target, self.eps).mean()


_PATCH_FLAG = "_asp_eiou_patched"


def patch_ultralytics_eiou() -> bool:
    """将 ultralytics 边界框回归损失由 CIoU 替换为 EIoU（幂等、可回退）。

    原理:
        ultralytics 的 ``BboxLoss`` 通过 ``bbox_iou(..., CIoU=True)`` 请求
        CIoU 相似度。本函数 monkey-patch ``ultralytics.utils.loss`` 中的
        ``bbox_iou``：当 ``CIoU=True`` 时返回 EIoU 相似度（1 - EIoU 损失），
        其余调用委托原实现，从而仅影响检测回归损失，不影响 NMS 等其他
        IoU 计算。

    兼容性:
        - 通过 ``**kwargs`` 透传未知关键字参数，兼容新版 ultralytics 对
          ``bbox_iou`` 新增的参数（如 ``overlap``、``myiou`` 等）。
        - **只补丁 ``ultralytics.utils.loss`` 模块的引用**，不补丁
          ``ultralytics.utils.metrics``：``tal.py``（TaskAlignedAssigner）也从
          metrics 导入 ``bbox_iou`` 并用 ``CIoU=True`` 计算正样本分配指标，
          若一并补丁会连带改变标签分配，污染"仅替换回归损失"的消融语义。
        - 幂等：重复调用不会二次包裹。

    Returns:
        是否成功打补丁。ultralytics 未安装或内部结构变化时返回 False。
    """
    try:
        import ultralytics.utils.loss as loss_module
        _orig = loss_module.bbox_iou
    except (ImportError, AttributeError):
        return False

    if getattr(loss_module, _PATCH_FLAG, False):
        return True

    def _eiou_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False,
                       CIoU=False, eps=1e-7, **kwargs):
        if not CIoU:
            return _orig(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU,
                         CIoU=False, eps=eps, **kwargs)
        if xywh:
            x1, y1, w, h = box1.unbind(dim=-1)
            box1 = torch.stack([x1 - w / 2.0, y1 - h / 2.0,
                                x1 + w / 2.0, y1 + h / 2.0], dim=-1)
            x1, y1, w, h = box2.unbind(dim=-1)
            box2 = torch.stack([x1 - w / 2.0, y1 - h / 2.0,
                                x1 + w / 2.0, y1 + h / 2.0], dim=-1)
        return 1.0 - eiou_loss(box1, box2, eps)

    loss_module.bbox_iou = _eiou_bbox_iou
    setattr(loss_module, _PATCH_FLAG, True)

    # 注意：这里刻意 **不** 补丁 ultralytics.utils.metrics.bbox_iou。
    # BboxLoss 调用的是 loss 模块内的全局引用（from .metrics import bbox_iou），
    # 只改这一处即可让「回归损失」用上 EIoU；而 tal.py 的标签分配器仍用原生
    # CIoU，从而保证消融实验只改动了损失函数这一个变量。
    return True


if __name__ == "__main__":
    pred = torch.tensor([[10.0, 10.0, 60.0, 60.0], [0.0, 0.0, 50.0, 50.0]])
    target = torch.tensor([[12.0, 12.0, 62.0, 62.0], [2.0, 2.0, 52.0, 52.0]])
    loss = eiou_loss(pred, target)
    print("EIoU loss:", loss.tolist())
    print("Mean EIoU loss:", loss.mean().item())