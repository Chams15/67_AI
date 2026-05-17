# YOLOv8 Gesture Training

Train the small YOLOv8 model on `yolo_dataset_final` with automatic checkpointing, post-training validation, JSON config support, and DirectML-safe defaults.

## Install

```powershell
pip install ultralytics
```

If you want to keep dependencies separate, you can also create a small training-only requirements file containing `ultralytics`.

If you plan to train on AMD GPU via DirectML, also install `torch-directml` in the same virtual environment.

## Run

```powershell
python train_yolov8_gestures.py
```

The script reads settings from `train_config.json` by default. Edit that file to change training settings without modifying the code.

Current config defaults are stored in [train_config.json](train_config.json). The script still accepts command-line overrides, but config values are loaded first.

You can still override any setting from the command line:

```powershell
python train_yolov8_gestures.py --epochs 150 --batch 2 --device auto
```

Recommended defaults for this machine:
- `imgsz=640`
- `batch=4` to `8` if you want a safer starting point on 8GB VRAM
- `workers=2` normally, but `workers=0` is forced automatically when DirectML is detected
- `save-period=5`
- `patience=25`

If you enable DirectML, the script will:
- detect `torch_directml` in the virtual environment,
- cast the model to the DirectML device before training,
- and force dataloader workers to `0` to avoid Windows crashes.

## Resume after a crash

If the run folder already has `weights/last.pt`, the script resumes automatically unless you pass `--fresh`.

```powershell
python train_yolov8_gestures.py --resume
```

## DirectML

To train on an AMD GPU through DirectML, set the device in `train_config.json` to `"dml"` or pass `--device dml`.

Example:

```powershell
python train_yolov8_gestures.py --device dml
```

When DirectML is active, the trainer automatically sets `workers=0`.

## Outputs

The script writes artifacts to:
- `runs/gesture_yolov8s/...`
- periodic checkpoints in `weights/`
- `metrics.json` after validation
