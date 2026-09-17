"""Eye implementations. Heavy backends (flyvis, torch) stay optional."""

from fly_driver.eyes.cnn import SmallCnnEye
from fly_driver.eyes.factory import build_eye
from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.eyes.random_projection import RandomProjectionEye
from fly_driver.eyes.shuffled import ShuffledConnectomeEye

__all__ = [
    "FlyvisEye",
    "RandomProjectionEye",
    "ShuffledConnectomeEye",
    "SmallCnnEye",
    "build_eye",
]
