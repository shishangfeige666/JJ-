"""棋盘自动定位的测试。

三类输入各管一件事：

- `lobby.png`（真机大厅帧）—— **画面里根本没有棋盘**，必须早判定失败（[RF-3]）；
- 合成网格图 —— 不依赖真机 fixture 也能验证算法本身；
- `opening.png`（真机开局帧）—— 端到端确认投影峰值法在真实渲染上能work。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess.vision.locator import BoardNotFound, find_lines, locate

FIXTURES = Path(__file__).parent / "fixtures"

# 合成棋盘的几何：故意让 dx 与 dy 略有差别，比正方形更能暴露换算错误。
FAKE_W, FAKE_H = 900, 1100
FAKE_X0, FAKE_DX = 100, 85
FAKE_Y0, FAKE_DY = 180, 77


def _render_fake_board() -> np.ndarray:
    """浅木色底 + 深色 9 纵线 / 10 横线 的等距合成棋盘。"""
    img = np.full((FAKE_H, FAKE_W, 3), (196, 214, 232), np.uint8)  # 浅木色（BGR）
    dark = (90, 110, 130)

    for c in range(9):
        x = round(FAKE_X0 + c * FAKE_DX)
        cv2.line(img, (x, FAKE_Y0), (x, FAKE_Y0 + 9 * FAKE_DY), dark, 2)
    for r in range(10):
        y = round(FAKE_Y0 + r * FAKE_DY)
        cv2.line(img, (FAKE_X0, y), (FAKE_X0 + 8 * FAKE_DX, y), dark, 2)
    return img


@pytest.fixture(scope="module")
def fake_board() -> np.ndarray:
    return _render_fake_board()


def test_大厅帧判定无棋盘():
    img = cv2.imread(str(FIXTURES / "lobby.png"))  # [RF-3]
    assert img is not None, "缺少 tests/fixtures/lobby.png"
    with pytest.raises(BoardNotFound):
        locate(img)


def test_合成棋盘能定位(fake_board):
    g = locate(fake_board)
    assert g.is_plausible()
    # 定位结果必须真的落在画出来的那些线上，而不是随便编一组等距数字。
    assert g.origin[0] == pytest.approx(FAKE_X0, abs=3)
    assert g.origin[1] == pytest.approx(FAKE_Y0, abs=3)
    assert g.dx == pytest.approx(FAKE_DX, abs=2)
    assert g.dy == pytest.approx(FAKE_DY, abs=2)


def test_等距拟合能剔除异常峰():
    prof = np.zeros(400, np.float32)
    for x in [10, 50, 90, 130, 170, 210, 250, 290, 330]:
        prof[x] = 100
    prof[95] = 80  # 一个假峰
    assert np.allclose(find_lines(prof, 9), [10, 50, 90, 130, 170, 210, 250, 290, 330], atol=3)


def test_异常峰比真峰还强时也能剔除():
    """[RF-3]/鲁棒性：假峰更强时不能「取最高的 9 个」了事，得靠等距拟合救回来。"""
    prof = np.zeros(400, np.float32)
    for x in [10, 50, 90, 130, 170, 210, 250, 290, 330]:
        prof[x] = 100
    prof[95] = 200  # 比所有真峰都强
    assert np.allclose(find_lines(prof, 9), [10, 50, 90, 130, 170, 210, 250, 290, 330], atol=3)


def test_峰数不足时拒绝硬凑():
    prof = np.zeros(400, np.float32)
    for x in [10, 50, 90, 130, 170]:  # 只要 9 条却只有 5 条
        prof[x] = 100
    with pytest.raises(BoardNotFound):
        find_lines(prof, 9)


def test_真实开局帧能定位():
    path = FIXTURES / "opening.png"
    img = cv2.imread(str(path))
    assert img is not None, f"缺少 {path}"
    g = locate(img)
    assert g.is_plausible()
    assert 60 < g.cell_px < 200
