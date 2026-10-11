"""窗口发现与抓图测试。

抓图那条真机用例只在 JJ象棋 真开着的时候跑；其余几条纯靠合成图像就能验。

[RF-1] 游戏没开 / 标题变了 → 必须是带中文说明的 `WindowNotFound`。
[RF-2] 最小化或被遮挡时 `PrintWindow` 可能返回全白帧。**绝不能把全白帧当成
        「32 子全空」**，所以空白帧检测必须是这里钉死的第一道闸。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess.window import WindowNotFound, find_window, is_blank_frame

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> np.ndarray | None:
    """读 fixture 图片。用 `np.fromfile + imdecode` 而不是 `cv2.imread` —— 后者
    在非 ASCII 路径上会静默返回 None。"""
    path = FIXTURES / name
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def _game_running() -> bool:
    try:
        find_window("JJ象棋", timeout_s=0.0)
    except (WindowNotFound, OSError):
        return False
    return True


def test_找不到窗口抛中文异常():
    with pytest.raises(WindowNotFound) as ei:
        find_window("绝对不存在的窗口标题xyz", timeout_s=0.3)
    assert "找到" in str(ei.value) or "未" in str(ei.value)  # [RF-1]


def test_空白帧检测_纯白为真():
    assert is_blank_frame(np.full((2560, 1440, 3), 255, np.uint8)) is True


def test_空白帧检测_纯黑为真():
    assert is_blank_frame(np.zeros((2560, 1440, 3), np.uint8)) is True


def test_空白帧检测_真实画面为假():
    img = np.random.randint(0, 255, (400, 300, 3), np.uint8)  # [RF-2]
    assert is_blank_frame(img) is False


def test_空白帧检测_近全白也算空白():
    # 压缩/缩放边缘常留几个杂点，不能要求 100% 纯色。
    img = np.full((600, 400, 3), 255, np.uint8)
    img[0, 0] = (200, 200, 200)
    assert is_blank_frame(img) is True


def test_空白帧检测_真机抓到的全黑帧():
    """真机回归：PrintWindow 会**零星**返回整帧全黑。

    2026-10-11 用 `tools/record_frames.py --watch` 每 0.5 s 抓一次真机窗口，
    在游戏平稳停着没动的情况下抓到了平均亮度 0.00 的整帧全黑，下一帧就恢复正常。
    这张 `blank_frame.png` 就是那一帧的原样存档 —— 不是合成出来的。

    危害在于：空白帧上 90 个格子全部"看起来是空的"。要是把它当成正常画面传下去，
    识别层会认为棋盘上 32 个子全没了，进而判定对局结束或者走一步莫名其妙的棋。
    """
    img = _load_fixture("blank_frame.png")
    assert img is not None, "缺少 tests/fixtures/blank_frame.png"
    assert img.mean() < 1.0  # 确实是全黑
    assert is_blank_frame(img) is True


@pytest.mark.skipif(not _game_running(), reason="JJ象棋未运行")
def test_抓真窗口():
    w = find_window("JJ象棋", timeout_s=2.0)
    img = w.grab()
    assert img.shape == (w.client_size[1], w.client_size[0], 3)
    assert img.dtype == np.uint8
    assert not is_blank_frame(img)
    assert w.is_alive()
    assert w.size_changed(*w.client_size) is False
    assert w.size_changed(w.client_size[0] + 7, w.client_size[1]) is True
