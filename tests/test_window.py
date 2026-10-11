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

from jjchess.window import (
    TORN_DARK_RATIO,
    WindowNotFound,
    find_window,
    frame_usable,
    is_blank_frame,
    is_torn_frame,
)

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


def test_残缺帧检测_真机抓到的半张黑帧():
    """真机回归：PrintWindow 会返回「右下角整块全黑」的残缺帧。

    2026-10-11 用 `tools/record_frames.py --watch` 抓到 `mid_162214.png`：
    棋盘左上部分完全正常，从中间斜着切下去全是黑的（位图区未初始化）。

    它不是空白帧 —— 平均亮度 88.66，`is_blank_frame` 判 `False`。但黑区里的格子
    拉普拉斯方差≈0，会被识别层判成「空」，于是整盘看起来像红方凭空少了七八个子。
    所以必须有第二道闸。
    """
    img = _load_fixture("torn_frame.png")
    assert img is not None, "缺少 tests/fixtures/torn_frame.png"
    assert not is_blank_frame(img)  # 空白判据拦不住它
    assert is_torn_frame(img) is True
    assert frame_usable(img) is False


def test_残缺帧判据不误伤好帧():
    """好帧的近黑像素占比约 1.1%，离 10% 的阈值有 9 倍余量。"""
    for name in ("lobby.png", "opening.png"):
        img = _load_fixture(name)
        assert img is not None, f"缺少 tests/fixtures/{name}"
        dark = float((np.asarray(img).astype(np.float32).mean(axis=2) < 10.0).mean())
        assert dark < TORN_DARK_RATIO / 2.0, f"{name} 近黑占比 {dark:.4f}，余量不足"
        assert is_torn_frame(img) is False
        assert frame_usable(img) is True


def test_残缺帧判据_合成半黑图():
    """合成一张左半正常、右半全黑的图 —— 判据要按面积占比而不是位置。"""
    rng = np.random.default_rng(20261011)
    noise = rng.integers(60, 200, (400, 400, 3), dtype=np.uint8)
    img = noise.copy()
    img[:, 200:] = 0  # 50% 黑
    assert is_torn_frame(img) is True
    img = noise.copy()
    img[:, 380:] = 0  # 5% 黑，安全区
    assert is_torn_frame(img) is False


def test_空白帧与残缺帧都不可用():
    assert frame_usable(None) is False
    assert frame_usable(np.full((100, 100, 3), 255, np.uint8)) is False
    assert frame_usable(np.zeros((100, 100, 3), np.uint8)) is False


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
