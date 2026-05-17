from __future__ import annotations

import importlib.util
from pathlib import Path
import argparse
import json
from datetime import datetime


DEFAULT_DATA = Path("yolo_dataset_final") / "data.yaml"
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_PROJECT = Path("runs") / "gesture_yolov8s"
DEFAULT_NAME = "gesture_yolov8s"
DEFAULT_CONFIG = Path("train_config.json")


DEFAULT_SETTINGS = {
    "data": str(DEFAULT_DATA),
    "model": DEFAULT_MODEL,
    "project": str(DEFAULT_PROJECT),
    "name": DEFAULT_NAME,
    "epochs": 100,
    "imgsz": 640,
    "batch": 4,
    "workers": 2,
    "device": "auto",
    "patience": 25,
    "save_period": 5,
    "seed": 67,
    "lr0": 0.01,
    "resume": False,
    "fresh": False,
    "cache": False,
    "amp": True,
    "close_mosaic": 10,
    "plots": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLOv8-small model on the gesture dataset with periodic checkpoints, "
            "evaluation metrics, and progress reporting."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config file.")
    parser.add_argument("--data", default=argparse.SUPPRESS, help="Path to YOLO data.yaml file.")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Base YOLOv8 checkpoint to start from.")
    parser.add_argument("--project", default=argparse.SUPPRESS, help="Folder where training runs are saved.")
    parser.add_argument("--name", default=argparse.SUPPRESS, help="Run name used inside the project folder.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=argparse.SUPPRESS, help="Input image size.")
    parser.add_argument("--batch", type=int, default=argparse.SUPPRESS, help="Batch size. Keep this small for 8GB VRAM.")
    parser.add_argument("--workers", type=int, default=argparse.SUPPRESS, help="Dataloader worker count.")
    parser.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help='Device to use: "auto", "cpu", "0", "0,1", etc.',
    )
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS, help="Early stopping patience.")
    parser.add_argument("--save-period", type=int, default=argparse.SUPPRESS, help="Save a checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for reproducibility.")
    parser.add_argument("--lr0", type=float, default=argparse.SUPPRESS, help="Initial learning rate.")
    parser.add_argument("--resume", action="store_true", default=argparse.SUPPRESS, help="Resume from the latest checkpoint if present.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Force a new run directory instead of resuming or reusing an existing run.",
    )
    parser.add_argument("--cache", action="store_true", default=argparse.SUPPRESS, help="Cache images in RAM for faster training.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=argparse.SUPPRESS, help="Disable mixed precision training.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=argparse.SUPPRESS, help="Enable mixed precision training.")
    parser.add_argument("--close-mosaic", type=int, default=argparse.SUPPRESS, help="Disable mosaic augmentation in the last N epochs.")
    parser.add_argument("--plots", action="store_true", default=argparse.SUPPRESS, help="Generate training/validation plots.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=argparse.SUPPRESS, help="Disable training/validation plots.")
    return parser


def load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config file is not valid JSON: {config_path}\n{exc}") from exc

    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")

    return loaded


def resolve_settings(config: dict, cli_args: argparse.Namespace) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    settings.update(config)

    cli_dict = vars(cli_args).copy()
    cli_dict.pop("config", None)

    settings.update(cli_dict)
    return settings


def torch_directml_is_available() -> bool:
    return importlib.util.find_spec("torch_directml") is not None


def detect_device(preferred: str) -> dict:
    device_info = {
        "backend": "cpu",
        "torch_device": None,
        "uses_directml": False,
    }

    if preferred != "auto":
        if preferred == "dml":
            if not torch_directml_is_available():
                raise SystemExit(
                    "DirectML was requested but torch_directml is not installed in this environment.\n"
                    "Install it first or change the device setting in train_config.json."
                )

            import torch_directml

            dml_device = torch_directml.device()
            device_info.update({"backend": "dml", "torch_device": dml_device, "uses_directml": True})
            return device_info

        device_info["backend"] = preferred
        return device_info

    try:
        import torch

        if torch_directml_is_available():
            import torch_directml

            dml_device = torch_directml.device()
            device_info.update({"backend": "dml", "torch_device": dml_device, "uses_directml": True})
            return device_info

        if torch.cuda.is_available():
            device_info["backend"] = "0"
            return device_info

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            device_info["backend"] = "mps"
            return device_info
    except Exception:
        pass

    return device_info


def apply_directml_model_cast(model, dml_device) -> None:
    if dml_device is None:
        return

    if hasattr(model, "model") and model.model is not None:
        model.model = model.model.to(dml_device)

    if hasattr(model, "device"):
        model.device = dml_device


def ultralytics_device_arg(device_info: dict) -> str | None:
    if device_info.get("uses_directml"):
        return None
    return device_info.get("backend", "cpu")


def ensure_run_name(project: Path, name: str, resume: bool, fresh: bool) -> tuple[str, bool]:
    """Return a safe run name and whether resume should be used."""
    run_dir = project / name
    last_ckpt = run_dir / "weights" / "last.pt"

    if fresh:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{name}_{stamp}", False

    if resume and last_ckpt.exists():
        return name, True

    if last_ckpt.exists():
        return name, True

    if run_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{name}_{stamp}", False

    return name, False


def load_ultralytics():
    try:
        from ultralytics import YOLO

        return YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it first with:\n"
            "  pip install ultralytics\n"
            "or install from train_requirements.txt if you create one."
        ) from exc


def install_progress_callbacks(model) -> None:
    if not hasattr(model, "add_callback"):
        return

    def on_train_start(trainer):
        epochs = getattr(trainer, "epochs", None)
        device = getattr(trainer, "device", None)
        print(f"Starting training: epochs={epochs}, device={device}")

    def on_train_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", 0))
        msg = f"Epoch {epoch}/{total} finished"
        loss = getattr(trainer, "tloss", None)
        if loss is not None:
            msg += f" | loss={loss}"
        print(msg)

    def on_fit_epoch_end(trainer):
        metrics = getattr(trainer, "metrics", None)
        if isinstance(metrics, dict) and metrics:
            summary = []
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
                if key in metrics:
                    summary.append(f"{key}={metrics[key]}")
            if summary:
                print("Validation summary: " + " | ".join(summary))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def write_metrics_json(metrics, out_path: Path) -> None:
    payload: dict = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        payload.update(results_dict)

    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("map", "map50", "map75", "mp", "mr"):
            if hasattr(box, attr):
                payload[f"box.{attr}"] = float(getattr(box, attr))

        maps = getattr(box, "maps", None)
        if maps is not None:
            payload["box.maps"] = [float(x) for x in maps]

    if not payload:
        payload["raw"] = str(metrics)

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json_config(config_path)
    settings = resolve_settings(config, args)

    data_path = Path(settings["data"])
    if not data_path.exists():
        raise SystemExit(f"Data config not found: {data_path}")

    project = Path(settings["project"])
    project.mkdir(parents=True, exist_ok=True)

    run_name, do_resume = ensure_run_name(project, settings["name"], settings["resume"], settings["fresh"])
    device_info = detect_device(str(settings["device"]))

    if device_info["uses_directml"]:
        if int(settings["workers"]) != 0:
            print("DirectML detected; forcing workers=0 to avoid dataloader crashes on Windows.")
        settings["workers"] = 0

    print("Training configuration:")
    print(f"  config: {config_path}")
    print(f"  data:   {data_path}")
    print(f"  model:  {settings['model']}")
    print(f"  run:    {project / run_name}")
    print(f"  device: {device_info['backend']}")
    print(f"  batch:  {settings['batch']}")
    print(f"  imgsz:  {settings['imgsz']}")
    print(f"  epochs: {settings['epochs']}")
    print(f"  resume: {do_resume}")

    YOLO = load_ultralytics()
    model = YOLO(settings["model"])
    apply_directml_model_cast(model, device_info["torch_device"])
    install_progress_callbacks(model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=settings["epochs"],
        imgsz=settings["imgsz"],
        batch=settings["batch"],
        workers=settings["workers"],
        project=str(project),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=settings["save_period"],
        patience=settings["patience"],
        seed=settings["seed"],
        lr0=settings["lr0"],
        cache=settings["cache"],
        amp=settings["amp"],
        close_mosaic=settings["close_mosaic"],
        plots=settings["plots"],
        verbose=True,
    )

    train_device = ultralytics_device_arg(device_info)
    if train_device is not None:
        train_kwargs["device"] = train_device

    if do_resume:
        train_kwargs["resume"] = True

    print("\nStarting training...\n")
    train_results = model.train(**train_kwargs)

    run_dir = project / run_name
    weights_dir = run_dir / "weights"
    last_ckpt = weights_dir / "last.pt"
    best_ckpt = weights_dir / "best.pt"

    print("\nTraining finished.")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Best checkpoint: {best_ckpt}")

    print("\nRunning evaluation on the validation split...\n")
    val_device = ultralytics_device_arg(device_info)
    if val_device is not None:
        val_results = model.val(
            data=str(data_path),
            imgsz=settings["imgsz"],
            batch=settings["batch"],
            device=val_device,
            split="val",
            plots=True,
            verbose=True,
        )
    else:
        print("DirectML path active: validation runs with the model already moved to the DirectML device.")
        val_results = model.val(
            data=str(data_path),
            imgsz=settings["imgsz"],
            batch=settings["batch"],
            split="val",
            plots=True,
            verbose=True,
        )

    metrics_path = run_dir / "metrics.json"
    write_metrics_json(val_results, metrics_path)

    print("\nValidation metrics:")
    results_dict = getattr(val_results, "results_dict", None)
    if isinstance(results_dict, dict) and results_dict:
        for key, value in results_dict.items():
            print(f"  {key}: {value}")
    else:
        box = getattr(val_results, "box", None)
        if box is not None:
            for attr in ("map", "map50", "map75", "mp", "mr"):
                if hasattr(box, attr):
                    print(f"  box.{attr}: {getattr(box, attr)}")

    print(f"\nSaved metrics JSON: {metrics_path}")
    print(f"Training artifacts stored in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
