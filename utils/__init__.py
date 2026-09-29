"""UA-DETRAC-YOLOv8n-ASP 工具包。"""

from .metrics import (
    compute_iou,
    match_predictions,
    compute_ap,
    summarize_metrics,
)
from .weather_annotations import (
    SCENES,
    SCENE_CN,
    normalize_scene,
    load_scene_map,
    assign_scene,
    group_by_scene,
    available_scene_file,
)

__all__ = [
    "compute_iou",
    "match_predictions",
    "compute_ap",
    "summarize_metrics",
    "SCENES",
    "SCENE_CN",
    "normalize_scene",
    "load_scene_map",
    "assign_scene",
    "group_by_scene",
    "available_scene_file",
]