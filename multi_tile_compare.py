"""Quick multi-tile comparison to see SOLO vs PointNet++ across sites."""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

from compare_methods import (load_tile, load_gt_annotations, run_watershed,
                              match_detections, compute_itd_metrics, mask_quality)
from src.segmentation.pointnet2_model import build_model
from src.segmentation.pipeline import segment_scene

cfg = yaml.safe_load(open("configs/segmentation.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg).to(device)
ckpt = torch.load("results/segmentation/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

TILES = [
    ("MLBS",  "train",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_MLBS_3_541000_4140000_image_crop.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_MLBS_3_541000_4140000_image_crop.xml"),
    ("NIWO",  "train",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_NIWO_2_450000_4426000_image_crop.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_NIWO_2_450000_4426000_image_crop.xml"),
    ("OSBS",  "train",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_OSBS_4_405000_3286000_image.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_OSBS_4_405000_3286000_image.xml"),
    ("HARV",  "train",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_HARV_5_733000_4698000_image_crop.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_HARV_5_733000_4698000_image_crop.xml"),
    ("SJER628", "test",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/evaluation/evaluation/LiDAR/2018_SJER_3_252000_4104000_image_628.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_SJER_3_252000_4104000_image_628.xml"),
    ("TEAK315crop", "train",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_TEAK_3_315000_4094000_image_crop.laz",
     "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_TEAK_3_315000_4094000_image_crop.xml"),
]

hdr = f"{'Sitio':<12} {'Split':<6} {'GT':>4} {'Ha':>5} | {'F1-A':>6} {'F1-B':>6} {'Delta':>6} | {'P-A':>5} {'P-B':>5} | {'R-A':>5} {'R-B':>5} | {'ASPRS%':>6} {'PNet%':>6} {'MaskIoU':>8}"
print(hdr)
print("-" * len(hdr))

for site, split, laz, xml in TILES:
    print(f"  {site}...", flush=True)
    xyz, cls, area_ha = load_tile(laz)
    gt = load_gt_annotations(xml, xyz)

    # Flujo A
    am = cls == 5
    ra = run_watershed(xyz, am, cls, cfg)
    tp_a, fp_a, fn_a = match_detections(ra["centroids"], gt, 8.0)
    ma = compute_itd_metrics(tp_a, fp_a, fn_a)

    # Flujo B
    pnet = segment_scene(xyz, cls, model, device, cfg) == 1
    rb = run_watershed(xyz, pnet, cls, cfg)
    tp_b, fp_b, fn_b = match_detections(rb["centroids"], gt, 8.0)
    mb = compute_itd_metrics(tp_b, fp_b, fn_b)

    mq = mask_quality(pnet, cls)

    print(f"{site:<12} {split:<6} {len(gt):>4} {area_ha:>5.2f} | "
          f"{ma['f1']:>6.3f} {mb['f1']:>6.3f} {mb['f1']-ma['f1']:>+6.3f} | "
          f"{ma['precision']:>5.3f} {mb['precision']:>5.3f} | "
          f"{ma['recall']:>5.3f} {mb['recall']:>5.3f} | "
          f"{100*am.sum()/len(xyz):>5.1f}% {100*pnet.sum()/len(xyz):>5.1f}% "
          f"{mq['mask_vs_asprs_iou']:>8.3f}", flush=True)

print("\nDone.")
