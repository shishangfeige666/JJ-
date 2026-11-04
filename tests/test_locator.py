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
    # 全仓统一用 `np.fromfile` + `imdecode`：`cv2.imread` 在非 ASCII 路径上会静默失败。
    img = cv2.imdecode(np.fromfile(str(FIXTURES / "lobby.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
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
    """真机开局帧的几何**逐个数字钉死**。

    原来只断言 `60 < cell_px < 200` —— 真值 125.85，而 60~200 允许 ±60% 的误差，
    任何「尺度对但整体缩小」的假解都落在里面，等于这条用例没有区分力。
    合成帧那条用的是 `abs=2/abs=3`，真机帧没有理由更松。
    fixture 是固定 PNG，`locate` 是确定性的。
    """
    path = FIXTURES / "opening.png"
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"缺少 {path}"
    g = locate(img)
    assert g.is_plausible()
    assert g.origin[0] == pytest.approx(101.52, abs=3)
    assert g.origin[1] == pytest.approx(609.32, abs=3)
    assert g.dx == pytest.approx(154.57, abs=2)
    assert g.dy == pytest.approx(148.06, abs=2)
    assert g.cell_px == pytest.approx(125.85, abs=2)


def test_棋盘外那条特别强的线不该抬高标准():
    """棋盘外面（木框、侧栏）常有比网格线还强的边，不能拿它当「线强度」的基准。

    原来是 `拟合线中位数 ÷ 全部候选里最强的那条`。真机 2026-11-04 实测：内线 619~1050
    （中位数 685），候选里有一条 1575 的棋盘外竖线 —— 门槛 0.45×1575≈709，同一盘棋
    只要棋子压住几段线，比值就从 0.65 掉到 0.44，`locate()` 于是翻脸说「没有棋盘」。
    基准换成**拟合到的这些线自己的最大值**之后，棋盘外那条再强也不影响判断。
    """
    prof = np.zeros(400, np.float32)
    for x in [10, 50, 90, 130, 170, 210, 250, 290, 330]:
        prof[x] = 150
    prof[130] = 130  # 被棋子压住一段，弱一点
    prof[395] = 400  # 棋盘外的强边（真机上是木框）
    assert np.allclose(find_lines(prof, 9), [10, 50, 90, 130, 170, 210, 250, 290, 330], atol=3)


def test_拟到的线强弱悬殊时仍然拒绝():
    """换掉基准不等于不设防：拟到的线里强弱差一大截，照样不当棋盘。"""
    prof = np.zeros(400, np.float32)
    for i, x in enumerate([10, 50, 90, 130, 170, 210, 250, 290, 330]):
        prof[x] = 100 if i == 0 else 40
    with pytest.raises(BoardNotFound):
        find_lines(prof, 9)


def test_棋子压住几段线的真机帧也能定位():
    """2026-11-04 真机现场帧：`locate()` 曾在这里失败，把还在下的一局判成「棋盘消失」。

    那一帧内线强度的中位数是 685，候选里最强的是棋盘外一条 1575 的竖线，门槛 709 ——
    差 3% 就翻脸（`pre_live.png` 是同一盘棋更早的一帧，当时还能定位）。换掉基准之后，
    这一帧定出来的几何与 `pre_live.png` 完全一致。

    `weak_lines.png` 是从 `_probe/` 里挑出来改名的（`tests/fixtures/mid_*.png`
    在 .gitignore 里，是滚动产物）。
    """
    path = FIXTURES / "weak_lines.png"
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"缺少 {path}"
    g = locate(img)
    assert g.is_plausible()
    assert g.origin[0] == pytest.approx(118.31, abs=3)
    assert g.origin[1] == pytest.approx(629.30, abs=3)
    assert g.dx == pytest.approx(150.38, abs=2)
    assert g.dy == pytest.approx(143.60, abs=2)
