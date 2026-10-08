"""Shared ChArUco board definition used by generation, verification, detection and calibration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIB_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CALIB_DIR / "config.yaml"
DEFAULT_DATA_DIR = CALIB_DIR / "data"


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class BoardConfig:
    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    dictionary: str
    legacy_pattern: bool = False
    first_marker_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BoardConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def num_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def num_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def marker_ids(self) -> np.ndarray:
        return np.arange(self.first_marker_id, self.first_marker_id + self.num_markers, dtype=np.int32)

    @property
    def pattern_size_m(self) -> tuple:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    def corner_row_col(self, charuco_ids):
        ids = np.asarray(charuco_ids).ravel()
        return ids // (self.squares_x - 1), ids % (self.squares_x - 1)


def make_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(cfg: BoardConfig):
    if cfg.marker_m >= cfg.square_m:
        raise ValueError("marker_m must be smaller than square_m")
    dictionary = make_dictionary(cfg.dictionary)
    dict_size = dictionary.bytesList.shape[0]
    if cfg.marker_ids[-1] >= dict_size:
        raise ValueError(f"{cfg.dictionary} has {dict_size} markers; board needs IDs up to {cfg.marker_ids[-1]}")
    board = cv2.aruco.CharucoBoard((cfg.squares_x, cfg.squares_y), cfg.square_m, cfg.marker_m,
                                   dictionary, cfg.marker_ids)
    board.setLegacyPattern(cfg.legacy_pattern)
    return board


def make_detector(board):
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def board_corners_m(board) -> np.ndarray:
    """(N, 3) ChArUco corner positions in the board frame, indexed by corner ID."""
    return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)


def board_metadata(cfg: BoardConfig) -> dict:
    w, h = cfg.pattern_size_m
    return {
        **asdict(cfg),
        "num_markers": cfg.num_markers,
        "num_charuco_corners": cfg.num_corners,
        "marker_ids": cfg.marker_ids.tolist(),
        "pattern_size_mm": [round(w * 1000, 3), round(h * 1000, 3)],
        "board_frame": "OpenCV CharucoBoard: origin at the outer corner of square (0,0), "
                       "x along columns, y along rows, z into the board",
        "opencv_version": cv2.__version__,
    }


def same_board(a: dict, b: dict) -> bool:
    keys = ("squares_x", "squares_y", "square_m", "marker_m", "dictionary", "legacy_pattern", "first_marker_id")
    return all(a.get(k) == b.get(k) for k in keys)
