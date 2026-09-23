# BLB Detection

A PyTorch implementation of **DeepLabV3+** for **binary semantic segmentation** — separating a
single foreground class (`byk`) from the background. It covers the full workflow: training,
inference, and ONNX export.

## Features

- DeepLabV3+ with ASPP and two switchable backbones: **MobileNetV2** and **Xception**
  (downsample factor 8 / 16)
- Two-phase training (freeze backbone, then unfreeze) for limited-memory GPUs
- Losses: cross-entropy, focal loss, dice loss
- Optimizers: SGD / Adam with cosine or step learning-rate decay
- Data augmentation: random scale/crop, flip, rotation, Gaussian blur, HSV jitter
- VOC-format dataset support
- mIoU / mPA / accuracy evaluation during training
- Inference modes: `predict`, `video`, `fps`, `dir_predict`, `export_onnx`

## Requirements

```text
torch, torchvision
numpy, opencv-python, Pillow, matplotlib, scipy, tqdm, tensorboard, h5py
# optional: onnx, onnxsim (ONNX export)
```

```bash
pip install -r requirements.txt
```

## Usage

### Dataset (VOC format)

```
VOCdevkit/VOC2007/
├── JPEGImages/             # .jpg input images
├── SegmentationClass/      # .png labels, each pixel value is its class id
└── ImageSets/Segmentation/ # train.txt / val.txt (one image name per line)
```

Label rule: **background = 0, foreground (`byk`) = 1** (do not use 255 for the foreground).

### Train

Edit the config block at the top of `train.py` (`num_classes`, `backbone`, `VOCdevkit_path`,
epochs, batch size, ...), then:

```bash
python train.py
```

### Predict

Edit `predict.py` (`mode`) and the inference config in `deeplab.py` (`model_path`,
`num_classes`, `backbone`), then:

```bash
python predict.py
```

### Export ONNX

Set `mode = "export_onnx"` in `predict.py` and run it.

## Notes

- `num_classes` = number of classes + 1 (background counts as a class). A binary task uses `2`.
- `num_classes`, `backbone`, and `downsample_factor` must match between training and inference,
  otherwise weight loading fails with a shape mismatch.
- `BatchNorm` requires `batch_size >= 2`.
