"""Camera intrinsics classes for storing calibration matrices and distortion parameters."""

from typing import Optional, Dict, Any, List
import json
import numpy as np
import numpy.typing as npt


class Intrinsics:
    """
    Class to store camera intrinsic parameters including calibration matrix and distortion coefficients.
    
    Attributes:
        ret: Calibration return value (RMS reprojection error)
        mtx: Camera calibration matrix (3x3)
        dist: Distortion coefficients array
        name: Optional name identifier for the camera
    """
    
    def __init__(self, 
                 ret: float, 
                 mtx: npt.NDArray[np.float64], 
                 dist: npt.NDArray[np.float64],
                 name: Optional[str] = None):
        """
        Initialize camera intrinsics.
        
        :param ret: RMS reprojection error from calibration
        :param mtx: 3x3 camera calibration matrix 
        :param dist: Distortion coefficients (typically 5 elements: k1, k2, p1, p2, k3)
        :param name: Optional name identifier for the camera
        :raises ValueError: If matrix dimensions are incorrect
        """
        self.ret = ret
        self.name = name
        
        # Convert to numpy arrays and validate dimensions
        self.mtx = np.array(mtx, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        
        if self.mtx.shape != (3, 3):
            raise ValueError(f"Calibration matrix must be 3x3, got {self.mtx.shape}")
        
        # Flatten distortion coefficients if needed
        if len(self.dist.shape) > 1:
            self.dist = self.dist.flatten()
    
    @property
    def fx(self) -> float:
        """Focal length in x direction."""
        return self.mtx[0, 0]
    
    @property
    def fy(self) -> float:
        """Focal length in y direction."""
        return self.mtx[1, 1]
    
    @property
    def cx(self) -> float:
        """Principal point x coordinate."""
        return self.mtx[0, 2]
    
    @property
    def cy(self) -> float:
        """Principal point y coordinate."""
        return self.mtx[1, 2]
    
    @property
    def k1(self) -> float:
        """First radial distortion coefficient."""
        return self.dist[0] if len(self.dist) > 0 else 0.0
    
    @property
    def k2(self) -> float:
        """Second radial distortion coefficient."""
        return self.dist[1] if len(self.dist) > 1 else 0.0
    
    @property
    def p1(self) -> float:
        """First tangential distortion coefficient."""
        return self.dist[2] if len(self.dist) > 2 else 0.0
    
    @property
    def p2(self) -> float:
        """Second tangential distortion coefficient."""
        return self.dist[3] if len(self.dist) > 3 else 0.0
    
    @property
    def k3(self) -> float:
        """Third radial distortion coefficient."""
        return self.dist[4] if len(self.dist) > 4 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert intrinsics to dictionary format.
        
        :return: Dictionary containing ret, mtx, and dist
        """
        return {
            'ret': self.ret,
            'mtx': self.mtx.tolist(),
            'dist': self.dist.tolist(),
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intrinsics':
        """
        Create Intrinsics object from dictionary.
        
        :param data: Dictionary containing calibration data
        :return: Intrinsics object
        """
        return cls(
            ret=data['ret'],
            mtx=np.array(data['mtx']),
            dist=np.array(data['dist']),
            name=data.get('name')
        )
    
    def save_json(self, filepath: str) -> None:
        """
        Save intrinsics to JSON file.
        
        :param filepath: Path to save the JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'Intrinsics':
        """
        Load intrinsics from JSON file.
        
        :param filepath: Path to the JSON file
        :return: Intrinsics object
        :raises FileNotFoundError: If file doesn't exist
        :raises json.JSONDecodeError: If file contains invalid JSON
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of intrinsics."""
        name_str = f" ({self.name})" if self.name else ""
        return (f"Intrinsics{name_str}:\n"
                f"  RMS Error: {self.ret:.6f}\n"
                f"  Focal Length: fx={self.fx:.2f}, fy={self.fy:.2f}\n"
                f"  Principal Point: cx={self.cx:.2f}, cy={self.cy:.2f}\n"
                f"  Distortion: k1={self.k1:.6f}, k2={self.k2:.6f}, p1={self.p1:.6f}, p2={self.p2:.6f}, k3={self.k3:.6f}")


class IntrinsicsPair:
    """
    Class to store a pair of camera intrinsics (typically thermal and wide cameras).
    
    Attributes:
        thermal: Intrinsics object for thermal camera
        wide: Intrinsics object for wide camera
    """
    
    def __init__(self, thermal: Intrinsics, wide: Intrinsics):
        """
        Initialize camera intrinsics pair.
        
        :param thermal: Intrinsics object for thermal camera
        :param wide: Intrinsics object for wide camera
        :raises ValueError: If either intrinsics object is None
        """
        if thermal is None or wide is None:
            raise ValueError("Both thermal and wide intrinsics must be provided")
            
        self.thermal = thermal
        self.wide = wide
    
    def __getitem__(self, key: str) -> Intrinsics:
        """
        Access intrinsics by key.
        
        :param key: 'thermal', 'Thermal', 'wide', 'Wide', 'T', or 'W'
        :return: Corresponding Intrinsics object
        :raises KeyError: If key is not recognized
        """
        key_lower = key.lower()
        if key_lower in ['thermal', 't']:
            return self.thermal
        elif key_lower in ['wide', 'w']:
            return self.wide
        else:
            raise KeyError(f"Invalid key '{key}'. Use 'thermal'/'T' or 'wide'/'W'")
    
    def items(self) -> List[tuple]:
        """Return list of (name, intrinsics) tuples."""
        return [('thermal', self.thermal), ('wide', self.wide)]
    
    def keys(self) -> List[str]:
        """Return list of camera names."""
        return ['thermal', 'wide']
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert intrinsics pair to dictionary format.
        
        :return: Dictionary containing both camera intrinsics
        """
        return {
            'Thermal': self.thermal.to_dict(),
            'Wide': self.wide.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> 'IntrinsicsPair':
        """
        Create IntrinsicsPair from dictionary.
        
        :param data: Dictionary containing calibration data for both cameras
        :return: IntrinsicsPair object
        """
        thermal_data = data.get('Thermal') or data.get('thermal')
        wide_data = data.get('Wide') or data.get('wide')
        
        if thermal_data is None or wide_data is None:
            raise ValueError("Dictionary must contain both 'Thermal' and 'Wide' keys")
        
        thermal = Intrinsics.from_dict(thermal_data)
        thermal.name = thermal.name or 'Thermal'
        
        wide = Intrinsics.from_dict(wide_data)
        wide.name = wide.name or 'Wide'
        
        return cls(thermal, wide)
    
    def save_json(self, filepath: str) -> None:
        """
        Save intrinsics pair to JSON file.
        
        :param filepath: Path to save the JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'IntrinsicsPair':
        """
        Load intrinsics pair from JSON file.
        
        :param filepath: Path to the JSON file
        :return: IntrinsicsPair object
        :raises FileNotFoundError: If file doesn't exist
        :raises json.JSONDecodeError: If file contains invalid JSON
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of intrinsics pair."""
        return f"IntrinsicsPair:\n{self.thermal}\n\n{self.wide}"
