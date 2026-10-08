"""On-disk session layout.

    <session>/session.json          board, camera, robot, thresholds, verification
    <session>/samples/<NNN>/        image.png + sample.json (used for calibration)
    <session>/rejected/<NNN>/       samples deleted during collection or review (recoverable)
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SAMPLES = "samples"
REJECTED = "rejected"


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


class Session:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "session.json").exists():
            raise FileNotFoundError(f"{self.root} is not a calibration session (no session.json)")

    @classmethod
    def create(cls, root, meta: dict) -> "Session":
        root = Path(root)
        if (root / "session.json").exists():
            raise FileExistsError(f"{root} already exists; pass --resume to add samples")
        (root / SAMPLES).mkdir(parents=True, exist_ok=True)
        (root / REJECTED).mkdir(exist_ok=True)
        write_json(root / "session.json", meta)
        return cls(root)

    @property
    def meta(self) -> dict:
        return read_json(self.root / "session.json")

    def sample_dirs(self, rejected=False) -> list:
        d = self.root / (REJECTED if rejected else SAMPLES)
        return sorted(p for p in d.iterdir() if (p / "sample.json").exists()) if d.exists() else []

    def next_index(self) -> int:
        names = [p.name for p in self.sample_dirs() + self.sample_dirs(rejected=True)]
        return max((int(n) for n in names if n.isdigit()), default=-1) + 1

    def save_sample(self, index: int, image: np.ndarray, sample: dict) -> Path:
        d = self.root / SAMPLES / f"{index:03d}"
        d.mkdir(parents=True)
        if not cv2.imwrite(str(d / "image.png"), image):
            raise IOError(f"Failed to write {d / 'image.png'}")
        write_json(d / "sample.json", sample)
        return d

    def _move(self, sample_dir, dest: str) -> Path:
        target = self.root / dest / Path(sample_dir).name
        target.parent.mkdir(exist_ok=True)
        shutil.move(str(sample_dir), str(target))
        return target

    def reject(self, sample_dir) -> Path:
        return self._move(sample_dir, REJECTED)

    def restore(self, sample_dir) -> Path:
        return self._move(sample_dir, SAMPLES)


def load_sample(sample_dir) -> dict:
    s = read_json(Path(sample_dir) / "sample.json")
    s["_dir"] = Path(sample_dir)
    return s


def load_image(sample: dict) -> np.ndarray:
    img = cv2.imread(str(sample["_dir"] / sample.get("image_file", "image.png")), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image for sample {sample['_dir']}")
    return img


def sample_E_C(sample: dict):
    return (np.asarray(sample["robot"]["T_base_ee"], dtype=np.float64),
            np.asarray(sample["detection"]["T_camera_board"], dtype=np.float64))
