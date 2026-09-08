from __future__ import annotations

import ast
from pathlib import Path
from uuid import uuid4

import pandas as pd
from PIL import Image, ImageOps

try:
    from .config import EAR_DETECTOR_WEIGHTS, UPLOAD_DIR, ensure_state_dirs
    from .state import utc_now
except ImportError:
    from config import EAR_DETECTOR_WEIGHTS, UPLOAD_DIR, ensure_state_dirs
    from state import utc_now


SUPPORTED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png"}
_EAR_DETECTOR = None
_TORCH = None
_T = None


def _normalise_bbox(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items()}
    return None


def crop_image_with_bbox(image: Image.Image, bbox):
    bbox = _normalise_bbox(bbox)
    if not isinstance(bbox, dict):
        return image.copy(), None

    width, height = image.size
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("w", width))
    h = float(bbox.get("h", height))

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.0:
        x *= width
        y *= height
        w *= width
        h *= height

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = max(x1 + 1, int(round(x + w)))
    y2 = max(y1 + 1, int(round(y + h)))
    x2 = min(x2, width)
    y2 = min(y2, height)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def _load_torchvision():
    global _TORCH, _T
    if _TORCH is None:
        import torch
        import torchvision.transforms as T

        _TORCH = torch
        _T = T
    return _TORCH, _T


def _body_transform():
    _, T = _load_torchvision()
    return T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _ear_transform():
    _, T = _load_torchvision()
    return T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _to_pil(tensor):
    _, T = _load_torchvision()
    return T.ToPILImage()(tensor)


def _load_ear_detector():
    global _EAR_DETECTOR
    if _EAR_DETECTOR is not None:
        return _EAR_DETECTOR
    torch, _ = _load_torchvision()
    if not EAR_DETECTOR_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing ear detector weights: {EAR_DETECTOR_WEIGHTS}")
    detector = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(EAR_DETECTOR_WEIGHTS),
    )
    detector.to("cuda" if torch.cuda.is_available() else "cpu")
    detector.eval()
    _EAR_DETECTOR = detector
    return _EAR_DETECTOR


def preprocess_body_image(raw_image: Image.Image, output_path: Path, bbox=None):
    image = ImageOps.exif_transpose(raw_image).convert("RGB")
    cropped, coords = crop_image_with_bbox(image, bbox)
    tensor = _body_transform()(cropped)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(tensor).save(output_path, format="JPEG")
    return output_path, cropped, coords


def _select_ear_candidates(predictions):
    detections = predictions.xyxy[0]
    candidates = []
    if detections is None or len(detections) == 0:
        return candidates
    detections = detections.cpu().numpy()
    names = predictions.names
    for x1, y1, x2, y2, conf, cls in detections:
        label = names[int(cls)]
        if label not in {"left", "right"}:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        candidates.append({
            "label": label,
            "conf": float(conf),
            "box": (x1, y1, x2, y2),
            "center_x": 0.5 * (x1 + x2),
        })
    return candidates


def _pick_best_pair(candidates):
    if not candidates:
        return None, None

    def pick_side(side):
        side_candidates = [c for c in candidates if c["label"] == side]
        if not side_candidates:
            return None
        best_conf = max(side_candidates, key=lambda c: c["conf"])["conf"]
        bests = [c for c in side_candidates if abs(c["conf"] - best_conf) < 1e-6]
        if side == "left":
            return min(bests, key=lambda c: c["center_x"])
        return max(bests, key=lambda c: c["center_x"])

    return pick_side("left"), pick_side("right")


def crop_ears(body_crop: Image.Image):
    torch, _ = _load_torchvision()
    detector = _load_ear_detector()
    body_crop = body_crop.convert("RGB")
    with torch.inference_mode():
        predictions = detector(body_crop)
    left_info, right_info = _pick_best_pair(_select_ear_candidates(predictions))
    return transform_ear_crops(body_crop, left_info, right_info)


def detect_ear_boxes(body_crop: Image.Image):
    torch, _ = _load_torchvision()
    detector = _load_ear_detector()
    body_crop = body_crop.convert("RGB")
    with torch.inference_mode():
        predictions = detector(body_crop)
    return _pick_best_pair(_select_ear_candidates(predictions))


def _clamped_box(info, width, height):
    if info is None:
        return None
    x1, y1, x2, y2 = info["box"]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def transform_ear_crops(body_crop: Image.Image, left_info, right_info):
    width, height = body_crop.size
    transform = _ear_transform()

    def crop_and_transform(info):
        box = _clamped_box(info, width, height)
        if box is None:
            return None
        return transform(body_crop.crop(box))

    return crop_and_transform(left_info), crop_and_transform(right_info)


def save_ear_crops(body_crop: Image.Image, left_path: Path, right_path: Path):
    left_tensor, right_tensor = crop_ears(body_crop)
    saved_left = ""
    saved_right = ""
    if left_tensor is not None:
        left_path.parent.mkdir(parents=True, exist_ok=True)
        _to_pil(left_tensor).save(left_path, format="JPEG")
        saved_left = str(left_path)
    if right_tensor is not None:
        right_path.parent.mkdir(parents=True, exist_ok=True)
        _to_pil(right_tensor).save(right_path, format="JPEG")
        saved_right = str(right_path)
    return saved_left, saved_right


def save_display_crops(raw_path: Path, output_dir: Path, bbox=None):
    """
    Save inspection-only RGB crops without ImageNet normalization.

    These files are not used for model inference. They mirror the body and ear crop
    geometry so the UI can show natural colors to an expert.
    """
    raw_image = ImageOps.exif_transpose(Image.open(raw_path)).convert("RGB")
    body_crop, coords = crop_image_with_bbox(raw_image, bbox)
    output_dir.mkdir(parents=True, exist_ok=True)

    body_path = output_dir / "body_rgb.jpg"
    left_path = output_dir / "left_ear_rgb.jpg"
    right_path = output_dir / "right_ear_rgb.jpg"
    body_crop.save(body_path, format="JPEG")

    try:
        left_info, right_info = detect_ear_boxes(body_crop)
    except Exception:
        return {
            "raw": str(raw_path),
            "body": str(body_path),
            "left_ear": "",
            "right_ear": "",
            "bbox": str(coords) if coords is not None else "",
        }

    width, height = body_crop.size

    def save_one(info, path):
        box = _clamped_box(info, width, height)
        if box is None:
            return ""
        body_crop.crop(box).save(path, format="JPEG")
        return str(path)

    return {
        "raw": str(raw_path),
        "body": str(body_path),
        "left_ear": save_one(left_info, left_path),
        "right_ear": save_one(right_info, right_path),
        "bbox": str(coords) if coords is not None else "",
    }


def preprocess_raw_image(raw_path: Path, output_dir: Path, bbox=None):
    raw_image = Image.open(raw_path)
    body_path = output_dir / "body.jpg"
    left_path = output_dir / "left_ear.jpg"
    right_path = output_dir / "right_ear.jpg"
    _, body_crop, coords = preprocess_body_image(raw_image, body_path, bbox=bbox)
    left_ear_path, right_ear_path = save_ear_crops(body_crop, left_path, right_path)
    return {
        "body_image_path": str(body_path),
        "left_ear_path": left_ear_path,
        "right_ear_path": right_ear_path,
        "bbox": str(coords) if coords is not None else "",
    }


def preprocess_upload(uploaded_file, state, bbox=None):
    """
    Store and preprocess an uploaded image using the Mara inference-time pipeline.

    The raw photo is preserved. The model-facing body crop and ear crops are derived
    artifacts under the UI state directory. If no bbox is supplied, the full image is
    resized as the body crop.
    """
    ensure_state_dirs()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError("Only .jpg, .jpeg, and .png uploads are supported.")

    query_id = f"upload:{uuid4().hex[:12]}"
    upload_dir = UPLOAD_DIR / query_id.replace(":", "_")
    upload_dir.mkdir(parents=True, exist_ok=True)

    raw_path = upload_dir / f"raw{suffix}"
    with raw_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    processed = preprocess_raw_image(raw_path, upload_dir, bbox=bbox)
    save_display_crops(raw_path, upload_dir / "display", bbox=bbox)

    uploads = state.read("uploads.csv")
    row = {
        "query_id": query_id,
        "timestamp": utc_now(),
        "original_filename": uploaded_file.name,
        "raw_image_path": str(raw_path),
        "body_image_path": processed["body_image_path"],
        "left_ear_path": processed["left_ear_path"],
        "right_ear_path": processed["right_ear_path"],
        "bbox": processed["bbox"],
        "status": "pending",
    }
    uploads = pd.concat([uploads, pd.DataFrame([row])], ignore_index=True)
    state.replace("uploads.csv", uploads)
    return row
