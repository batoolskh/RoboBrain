# RoboBrain – Object Detection with YOLOv8 in AI2-THOR

Detecting kitchen objects in the AI2-THOR simulator using YOLOv8, with the dataset managed on Roboflow.

## Classes
Apple, Bowl, Bread, Knife, Mug, Tomato

## Pipeline
1. **Data collection** (`collect_data.py`): the agent is teleported to random positions in 30 AI2-THOR kitchen scenes (FloorPlan1–30) with randomized object placement. Bounding boxes are generated automatically from the simulator's instance detections and saved in YOLO format (598 images).
2. **Dataset management (Roboflow)**: uploaded, reviewed, split 70/20/10, and augmented (horizontal flip, ±20% brightness) → 1435 images.
3. **Training**: YOLOv8n, 50 epochs, 640 image size, on Google Colab (T4 GPU).
4. **Testing** (`test_yolo.py`): the trained model runs on live frames from AI2-THOR.

## Results (validation set)
| Metric | Value |
|---|---|
| mAP50 | 0.80 |
| mAP50-95 | 0.64 |
| Precision | 0.84 |
| Recall | 0.72 |

Per-class mAP50-95: Bread 0.78, Tomato 0.73, Bowl 0.72, Apple 0.71, Mug 0.64, Knife 0.23.
Knife is the weakest class (thin, small objects, fewer examples) — next step is collecting more knife images.

Training plots are in `results/`, and sample detections in `yolo_test/`.

## Files
- `collect_data.py` – dataset generation from AI2-THOR
- `test_thor.py` – AI2-THOR setup check
- `test_yolo.py` – run the trained model in the simulator
- `models/best.pt` – trained weights
- `results/` – training curves, confusion matrix
- `yolo_test/` – detection samples

## Setup
Ubuntu (VMware), Python 3.10 (conda):
pip install -r requirements.txt
