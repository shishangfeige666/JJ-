"""窗口发现与抓图测试。

抓图那条真机用例只在 JJ象棋 真开着的时候跑；其余几条纯靠合成图像就能验。

[RF-1] 游戏没开 / 标题变了 → 必须是带中文说明的 `WindowNotFound`。
[RF-2] 最小化或被遮挡时 `PrintWindow` 可能返回全白帧。**绝不能把全白帧当成
        「32 子全空」**，所以空白帧检测必须是这里钉死的第一道闸。
"""

from __future__ import annotations

import numpy as np
import pytest

from jjchess.window import WindowNotFound, find_window, is_blank_frame


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
