# License Plate Recognition (LPR) — YOLO + Transformer OCR

An end-to-end, two-stage deep learning pipeline for automatic license plate recognition (ALPR) on Indian vehicles, built as a college capstone project.

## Problem Statement

Manually verifying vehicle identity for toll collection, traffic monitoring, and law enforcement doesn't scale. This project automates it with a two-stage deep learning pipeline that **detects** license plates in images and **recognizes** the text on them.

## Architecture

```
Input Image → [Stage 1: YOLOv8 Detection] → Cropped Plate → [Stage 2: Transformer OCR] → Plate Text
```

1. **Detection** — YOLOv8, fine-tuned on an Indian license plate dataset, localizes the plate region.
2. **Recognition** — TrOCR (transformer-based OCR) and EasyOCR (CRNN baseline) read the plate text from the crop.
3. **Post-processing** — Regex validation against the standard Indian plate format (`SS DD LLL DDDD`).

## Project Structure

```
.
├── LPR_Capstone_Project.ipynb   # Full pipeline: data prep, training, inference, evaluation
├── data/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/ # YOLO format: class x_center y_center width height
└── README.md
```

## Workflow

1. **Data preprocessing & character segmentation** — grayscale, CLAHE contrast enhancement, adaptive thresholding, contour-based character segmentation.
2. **Stage 1 training** — fine-tune YOLOv8 on plate-detection data (mosaic/HSV augmentation, flip disabled to preserve text orientation).
3. **Stage 2 recognition** — TrOCR / EasyOCR on the cropped plate region.
4. **Evaluation** — mAP@0.5 / mAP@0.5:0.95 / IoU for detection; character-level and exact-match accuracy for recognition.
5. **Batch inference** — run the full pipeline over a folder of test images.

## Dataset

- [License Plate Detection Dataset (10,125 Images)](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images)
- [Indian License Plates with Labels](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels) (for Indian-specific plate formats)

Download and unzip into a `raw_dataset/` folder, then run the `split_raw_dataset()` cell in the notebook to auto-organize it into the YOLO-format train/val/test structure.

## Setup

```bash
pip install ultralytics easyocr transformers opencv-python-headless pillow matplotlib pandas scikit-learn
```

Then open `LPR_Capstone_Project.ipynb` and run cells top to bottom.

## Results

| Metric | Value |
|---|---|
| Detection mAP@0.5 | *fill in after training* |
| Detection mAP@0.5:0.95 | *fill in after training* |
| Character accuracy | *fill in after evaluation* |
| Exact plate match accuracy | *fill in after evaluation* |

## Use Cases

- Automated toll collection
- Traffic monitoring and analytics
- Vehicle identification for law enforcement

## Cases

- No web frontend/API is included — this is a self-contained notebook pipeline. It can be wrapped in a FastAPI/Flask service for deployment later.
- Plate numbers are personally identifiable information (PII) in most jurisdictions — handle logged data accordingly (encryption at rest, limited retention).

## Tech Stack
python · PyTorch · Ultralytics YOLOv8 · HuggingFace Transformers (TrOCR) · EasyOCR · OpenCV · Pandas
