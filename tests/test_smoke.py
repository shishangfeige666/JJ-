"""基础可用性冒烟测试：依赖可导入、包可导入且带版本号。"""

import cv2
import numpy as np


def test_deps_importable():
    """numpy 与 opencv 已装好且能正常工作。"""
    assert np.zeros((2, 2)).shape == (2, 2)
    assert cv2.__version__


def test_package_importable():
    """jjchess 包可导入，并暴露 __version__。"""
    import jjchess

    assert jjchess.__version__ == "0.1.0"
