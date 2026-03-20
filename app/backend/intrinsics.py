"""Camera intrinsics classes for storing calibration matrices and distortion parameters."""

from typing import Optional, Dict, Any, List
import json
import numpy as np
import numpy.typing as npt
import os


class Intrinsics:
    """Stores camera intrinsic parameters: calibration matrix and distortion coefficients."""

    def __init__(
        self,
        mtx: npt.NDArray[np.float64],
        dist: npt.NDArray[np.float64],
        ret: Optional[float] = None,
        name: Optional[str] = None,
    ):
        self.ret = ret
        self.name = name
        self.mtx = np.array(mtx, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)

        if self.mtx.shape != (3, 3):
            raise ValueError(f"Calibration matrix must be 3x3, got {self.mtx.shape}")
        if len(self.dist.shape) > 1:
            self.dist = self.dist.flatten()

    @property
    def fx(self) -> float:
        return self.mtx[0, 0]

    @property
    def fy(self) -> float:
        return self.mtx[1, 1]

    @property
    def cx(self) -> float:
        return self.mtx[0, 2]

    @property
    def cy(self) -> float:
        return self.mtx[1, 2]

    @property
    def k1(self) -> float:
        return self.dist[0] if len(self.dist) > 0 else 0.0

    @property
    def k2(self) -> float:
        return self.dist[1] if len(self.dist) > 1 else 0.0

    @property
    def p1(self) -> float:
        return self.dist[2] if len(self.dist) > 2 else 0.0

    @property
    def p2(self) -> float:
        return self.dist[3] if len(self.dist) > 3 else 0.0

    @property
    def k3(self) -> float:
        return self.dist[4] if len(self.dist) > 4 else 0.0

    def __getitem__(self, key: str) -> npt.NDArray[np.float64]:
        key_lower = key.lower()
        if key_lower in ["mtx", "k"]:
            return self.mtx
        elif key_lower == "dist":
            return self.dist
        else:
            raise KeyError(f"Invalid key '{key}'. Use 'mtx'/'K' or 'dist'")

    def set(self, **kwargs) -> None:
        if "fx" in kwargs and kwargs["fx"] is not None:
            self.mtx[0, 0] = kwargs["fx"]
        if "fy" in kwargs and kwargs["fy"] is not None:
            self.mtx[1, 1] = kwargs["fy"]
        if "cx" in kwargs and kwargs["cx"] is not None:
            self.mtx[0, 2] = kwargs["cx"]
        if "cy" in kwargs and kwargs["cy"] is not None:
            self.mtx[1, 2] = kwargs["cy"]
        if "dist" in kwargs and kwargs["dist"] is not None:
            new_dist = np.array(kwargs["dist"], dtype=np.float64).flatten()
            orig_len = len(self.dist)
            if len(new_dist) < orig_len:
                padded = np.zeros(orig_len, dtype=np.float64)
                padded[: len(new_dist)] = new_dist
                self.dist = padded
            else:
                self.dist = new_dist[:orig_len]
        if "ret" in kwargs and kwargs["ret"] is not None:
            self.ret = kwargs["ret"]
        if "name" in kwargs and kwargs["name"] is not None:
            self.name = kwargs["name"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ret": self.ret,
            "mtx": self.mtx.tolist(),
            "dist": self.dist.tolist(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intrinsics":
        return cls(
            mtx=np.array(data.get("mtx") or data["K"]),
            dist=np.array(data["dist"]),
            ret=data.get("ret"),
            name=data.get("name"),
        )

    def copy(self) -> "Intrinsics":
        return Intrinsics(
            mtx=self.mtx.copy(),
            dist=self.dist.copy(),
            ret=self.ret,
            name=self.name,
        )


class IntrinsicsPair:
    """Stores a pair of camera intrinsics (thermal + wide/RGB)."""

    def __init__(self, thermal: Intrinsics, wide: Intrinsics):
        if thermal is None or wide is None:
            raise ValueError("Both thermal and wide intrinsics must be provided")
        self.thermal = thermal
        self.wide = wide

    def __getitem__(self, key: str) -> Intrinsics:
        key_lower = key.lower()
        if key_lower in ["thermal", "t"]:
            return self.thermal
        elif key_lower in ["wide", "w", "rgb"]:
            return self.wide
        else:
            raise KeyError(f"Invalid key '{key}'. Use 'thermal'/'T' or 'wide'/'W'/'rgb'")

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Thermal": self.thermal.to_dict(),
            "Wide": self.wide.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "IntrinsicsPair":
        thermal_data = data.get("Thermal") or data.get("thermal")
        wide_data = data.get("Wide") or data.get("wide") or data.get("RGB") or data.get("rgb")
        if thermal_data is None or wide_data is None:
            raise ValueError("Dictionary must contain both 'Thermal' and 'Wide'/'RGB' keys")
        thermal = Intrinsics.from_dict(thermal_data)
        thermal.name = thermal.name or "Thermal"
        wide = Intrinsics.from_dict(wide_data)
        wide.name = wide.name or "Wide"
        return cls(thermal, wide)

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "IntrinsicsPair":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> "IntrinsicsPair":
        return IntrinsicsPair(self.thermal.copy(), self.wide.copy())
