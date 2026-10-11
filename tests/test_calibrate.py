"""标定工具的纯函数测试。

`tools/` 不在 `jjchess` 包里（`main.py --calibrate` 用 `importlib` 按路径加载它），
这里照同样的方式加载，只测**不碰窗口**的部分。

重点是那个坐标换算：`WINDOW_NORMAL` 会把大图缩进屏幕显示，
而 `setMouseCallback` 给的是显示窗口坐标 —— 直接当原图像素用，标定出来的几何会
整体等比偏小，而且缩小后的几何照样能通过 `is_plausible()` 的全部判据。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from jjchess.geometry import BoardGeometry

_ROOT = Path(__file__).resolve().parent.parent


def _load_calibrate():
    path = _ROOT / "tools" / "calibrate.py"
    spec = importlib.util.spec_from_file_location("jjchess_calibrate_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calibrate = _load_calibrate()


def test_显示窗口被缩小过时坐标要换算回原图(monkeypatch):
    """本机客户区是 1440x2560，屏幕放不下，OpenCV 会缩到约 1/3 显示。

    不换算的话标定出来的 `dx` 就只有真值的三分之一左右。
    """
    monkeypatch.setattr(calibrate.cv2, "getWindowImageRect", lambda _n: (0, 0, 480, 853))
    x, y = calibrate._to_image_coords("w", 100, 200, 1440, 2560)
    assert (x, y) == (300, 600)


def test_显示窗口没有缩小时坐标原样返回(monkeypatch):
    monkeypatch.setattr(calibrate.cv2, "getWindowImageRect", lambda _n: (0, 0, 1440, 2560))
    assert calibrate._to_image_coords("w", 700, 900, 1440, 2560) == (700, 900)


def test_问不到显示区域时按原图算而不是崩掉(monkeypatch):
    """窗口刚被关掉之类，`getWindowImageRect` 可能拿到空矩形。"""

    def _boom(_name):
        raise calibrate.cv2.error("窗口没了")

    monkeypatch.setattr(calibrate.cv2, "getWindowImageRect", _boom)
    assert calibrate._to_image_coords("w", 700, 900, 1440, 2560) == (700, 900)


def test_缩小后的几何能通过校验所以必须靠换算挡住():
    """把这条不变式钉住：`is_plausible()` 挡不住「整体等比缩小」的假几何。

    这正是「必须换算回原图像素」的理由 —— 校验指望不上，只能从坐标源头修。
    """
    real = BoardGeometry((101.5, 609.3), 154.57, 148.06, 0.85 * 148.06)
    shrunk = BoardGeometry((33.8, 203.1), 154.57 / 3, 148.06 / 3, 0.85 * 148.06 / 3)
    assert real.is_plausible() and shrunk.is_plausible()
    assert shrunk.dx == pytest.approx(real.dx / 3)
