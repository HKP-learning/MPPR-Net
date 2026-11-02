# MPPR-Net

This repository contains the implementation of **MPPR-Net** for aerial small-object detection.

The code is developed based on an Ultralytics-style detection framework and RT-DETR. This README mainly provides instructions for environment setup, dataset preparation, training, validation, and inference.

---

## 1. Environment

The experiments were conducted with the following environment:

- Python 3.x
- PyTorch 1.12.0
- CUDA 11.6
- NVIDIA GeForce RTX 3090 24 GB

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not included in `requirements.txt`, install a version compatible with your CUDA environment first.

Example:

```bash
pip install torch==1.12.0 torchvision==0.13.0
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Model Configuration

The MPPR-Net model configuration is located at:

```text
ultralytics/cfg/models/MPPR-Net.yaml
```

The main components implemented in the model include:

```text
LSRB-based compact backbone
TASFI
MS-HFRN
AMFE
```

---

## 3. Dataset Preparation

The current implementation supports standard YOLO-format detection datasets.

A typical dataset directory is organized as follows:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

Each annotation file should have the same filename as its corresponding image.

Example:

```text
images/train/000001.jpg
labels/train/000001.txt
```

The annotation format is:

```text
class_id x_center y_center width height
```

where all coordinates are normalized to `[0, 1]`.

An example dataset YAML file is:

```yaml
path: /path/to/dataset

train: images/train
val: images/val

names:
  0: class_0
  1: class_1
```

For VisDrone2019, the dataset configuration used in the experiments is:

```text
VisDrone2019.yaml
```

Please modify the dataset path according to your local environment.

---

## 4. Training

### VisDrone2019

A reference training command is:

```bash
yolo task=detect mode=train \
    model=ultralytics/cfg/models/MPPR-Net.yaml \
    data=VisDrone2019.yaml \
    epochs=200 \
    batch=4 \
    imgsz=640 \
    device=0 \
    optimizer=AdamW \
    lr0=0.0001 \
    weight_decay=0.0001 \
    seed=0 \
    deterministic=True \
    amp=False
```

The main training settings used for VisDrone2019 are:

```yaml
epochs: 200
batch: 4
imgsz: 640
optimizer: AdamW

lr0: 0.0001
weight_decay: 0.0001

seed: 0
deterministic: true
amp: false

workers: 8
```

The following data augmentation settings are used:

```yaml
mosaic: 1.0
mixup: 0.2

fliplr: 0.5
flipud: 0.0

translate: 0.1
scale: 0.5

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4

degrees: 0.0
shear: 0.0
perspective: 0.0
copy_paste: 0.0
```

Training outputs are saved under:

```text
runs/train/
```

---

## 5. Training on Other Datasets

The number of training epochs used in our experiments is:

| Dataset | Epochs |
|---|---:|
| VisDrone2019 | 200 |
| UAVDT | 200 |
| AI-TOD | 400 |
| TinyPerson | 400 |
| SeaPerson | 100 |

To train on another dataset, replace the dataset YAML file and number of epochs.

Example:

```bash
yolo task=detect mode=train \
    model=ultralytics/cfg/models/MPPR-Net.yaml \
    data=AI-TOD.yaml \
    epochs=400 \
    batch=4 \
    imgsz=640 \
    device=0 \
    optimizer=AdamW \
    lr0=0.0001 \
    weight_decay=0.0001 \
    seed=0 \
    deterministic=True \
    amp=False
```

---

## 6. Validation

After training, the best model is usually saved as:

```text
runs/train/exp/weights/best.pt
```

Run validation using:

```bash
yolo task=detect mode=val \
    model=runs/train/exp/weights/best.pt \
    data=VisDrone2019.yaml \
    imgsz=640 \
    device=0
```

The validation process reports metrics including:

```text
Precision
Recall
mAP50
mAP50-95
```

---

## 7. Inference

Run inference on an image directory:

```bash
yolo task=detect mode=predict \
    model=runs/train/exp/weights/best.pt \
    source=/path/to/images \
    imgsz=640 \
    device=0 \
    save=True
```

For a single image:

```bash
yolo task=detect mode=predict \
    model=runs/train/exp/weights/best.pt \
    source=/path/to/image.jpg \
    imgsz=640 \
    device=0
```

Prediction results are saved under:

```text
runs/detect/
```

---

## 8. Pretrained Weights

If pretrained MPPR-Net checkpoints are provided, place them under:

```text
weights/
```

Example:

```text
weights/
├── MPPR-Net-VisDrone.pt
├── MPPR-Net-UAVDT.pt
├── MPPR-Net-AITOD.pt
├── MPPR-Net-TinyPerson.pt
└── MPPR-Net-SeaPerson.pt
```

Validation example:

```bash
yolo task=detect mode=val \
    model=weights/MPPR-Net-VisDrone.pt \
    data=VisDrone2019.yaml \
    imgsz=640 \
    device=0
```

---

## 9. Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
yolo task=detect mode=train \
    model=runs/train/exp/weights/last.pt \
    resume=True
```

---

## 10. Notes

- Modify dataset paths in the corresponding YAML files before training.
- GPU memory usage depends on the input resolution and batch size.
- If CUDA out-of-memory errors occur, reduce `batch` first.
- The default input resolution used in most experiments is `640 × 640`.
- Automatic mixed precision is disabled in the reported configuration.
- Results may vary slightly across different GPUs, CUDA versions, and PyTorch versions.
- For reproducibility, the default random seed is set to `0` with deterministic execution enabled.

---

## 11. Example Commands

### Train

```bash
yolo task=detect mode=train \
    model=ultralytics/cfg/models/MPPR-Net.yaml \
    data=VisDrone2019.yaml \
    epochs=200 \
    batch=4 \
    imgsz=640 \
    device=0
```

### Validate

```bash
yolo task=detect mode=val \
    model=path/to/best.pt \
    data=VisDrone2019.yaml \
    imgsz=640 \
    device=0
```

### Predict

```bash
yolo task=detect mode=predict \
    model=path/to/best.pt \
    source=path/to/images \
    imgsz=640 \
    device=0
```

---

## 12. Contact

If you encounter problems when running the code, please open an issue in this repository.