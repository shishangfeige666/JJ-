"""拟人化鼠标测试。

坐标换算是这一节最容易写错的地方，所以用例把期望值全部**手算写死**，不走 `cell_to_px`：

    板坐标 (rank, file) --ScreenMapper.to_screen--> 屏幕格 (row, col) --cell_to_px--> 屏幕像素

执红：`row = 9 - rank`、`col = file`；执黑：`row = rank`、`col = 8 - file`。
测试几何用 `origin=(100, 60), dx=80, dy=80`，于是 `cell_to_px(row, col) = (100 + 80*col, 60 + 80*row)`。

**所有用例都不碰真的 Win32 写操作** —— `dry_run=True` 天然不碰，需要走真实分支的两条
把 `jjchess.control` 里的两个薄封装 `_set_cursor_pos` / `_send_mouse_input` 换掉。
唯一的例外是最后那条真机用例：它只挪光标再挪回去，不做任何点击。
"""

from __future__ import annotations

import time

import pytest

from jjchess import control
from jjchess.control import Mouse
from jjchess.geometry import BLACK, BoardGeometry, ScreenMapper


def _geom() -> BoardGeometry:
    return BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)


class _FakeTime:
    """替掉 `control.time`，把拟人化停顿记下来而不真的等。"""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _FakeInput:
    """替掉 `_set_cursor_pos` / `_send_mouse_input`，记录调用顺序。"""

    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []
        self.buttons: list[str] = []

    def set_cursor_pos(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        return True

    def send_mouse_input(self, press: bool) -> int:
        self.buttons.append("按下" if press else "抬起")
        return 1


@pytest.fixture
def fake(monkeypatch) -> tuple[_FakeInput, _FakeTime]:
    inp, clock = _FakeInput(), _FakeTime()
    monkeypatch.setattr(control, "_set_cursor_pos", inp.set_cursor_pos)
    monkeypatch.setattr(control, "_send_mouse_input", inp.send_mouse_input)
    monkeypatch.setattr(control, "time", clock)
    return inp, clock


def test_dry_run不真点(monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover - 被调用就是失败
        raise AssertionError("dry_run 模式下不允许调用任何 Win32 写操作")

    monkeypatch.setattr(control, "_set_cursor_pos", _boom)
    monkeypatch.setattr(control, "_send_mouse_input", _boom)

    m = Mouse(dry_run=True)
    m.play_move("e0e1", _geom())
    assert len(m.clicks) == 2


def test_dry_run不等待():
    """彩排要快点把整局走完看坐标对不对，不该真的按拟人化节奏睡。"""
    m = Mouse(dry_run=True)
    t0 = time.perf_counter()
    for _ in range(20):
        m.play_move("e0e1", _geom())
    assert time.perf_counter() - t0 < 1.0
    assert len(m.clicks) == 40


def test_落点抖动在范围内():
    m = Mouse(dry_run=True, jitter_px=6)
    for _ in range(50):
        (x1, y1), (x2, y2) = m.play_move("e0e1", _geom())
        # e0 = 板坐标 (rank 0, file 4) -> 屏幕 (row 9, col 4) -> (420, 780)
        assert abs(x1 - 420.0) <= 6
        assert abs(y1 - 780.0) <= 6
        # e1 = (rank 1, file 4) -> 屏幕 (row 8, col 4) -> (420, 700)
        assert abs(x2 - 420.0) <= 6
        assert abs(y2 - 700.0) <= 6


def test_点击坐标对应正确交叉点():
    m = Mouse(dry_run=True, jitter_px=0)
    (x1, y1), (x2, y2) = m.play_move("h2e2", _geom())
    # h2 = (rank 2, file 7) -> 屏幕 (row 7, col 7) -> (660, 620)
    assert (x1, y1) == (660.0, 620.0)
    # e2 = (rank 2, file 4) -> 屏幕 (row 7, col 4) -> (420, 620)
    assert (x2, y2) == (420.0, 620.0)


def test_执黑时坐标上下左右都翻转():
    m = Mouse(dry_run=True, jitter_px=0, my_side=BLACK)
    (x1, y1), (x2, y2) = m.play_move("h2e2", _geom())
    # 执黑：row = rank, col = 8 - file
    # h2 = (rank 2, file 7) -> 屏幕 (row 2, col 1) -> (180, 220)
    assert (x1, y1) == (180.0, 220.0)
    # e2 = (rank 2, file 4) -> 屏幕 (row 2, col 4) -> (420, 220)
    assert (x2, y2) == (420.0, 220.0)


def test_非法着法拒绝():
    with pytest.raises(ValueError):
        Mouse(dry_run=True).play_move("zzzz", _geom())


def test_非法执色拒绝():
    with pytest.raises(ValueError):
        Mouse(dry_run=True, my_side=0)


@pytest.mark.parametrize("bad", [(float("nan"), 0.0), (0.0, float("inf"))])
def test_非有限坐标拒绝(bad):
    with pytest.raises(ValueError):
        Mouse(dry_run=True).click_screen(*bad)


def test_真点击会移动光标并按下抬起(fake):
    inp, clock = fake
    m = Mouse(dry_run=False, jitter_px=0)
    m.play_move("h2e2", _geom())

    # 两次定位到两个交叉点，坐标取整
    assert inp.moves == [(660, 620), (420, 620)]
    # 每点一次 = 按下 + 抬起
    assert inp.buttons == ["按下", "抬起", "按下", "抬起"]
    assert m.clicks == [(660.0, 620.0), (420.0, 620.0)]
    # 拟人化停顿：按下→抬起 60~140ms，两点之间 180~350ms
    assert len(clock.sleeps) == 3
    assert 0.06 <= clock.sleeps[0] <= 0.14
    assert 0.18 <= clock.sleeps[1] <= 0.35
    assert 0.06 <= clock.sleeps[2] <= 0.14


def test_click_client按客户区原点换算(fake):
    """`geom` 的 origin 是屏幕坐标，但客户区原点不在屏幕 (0,0)。

    本机实测：JJ象棋 窗口客户区原点在屏幕 `(-1440, -279)`。面板/标定工具拿到的是
    客户区坐标，所以必须有一个显式的加法，不能想当然。
    """
    inp, _ = fake
    m = Mouse(dry_run=False, jitter_px=0)
    m.click_client(100.0, 200.0, origin=(-1440, -279))

    assert inp.moves == [(-1340, -79)]
    assert m.clicks == [(-1340.0, -79.0)]


def test_映射与boardgeometry一致():
    """`play_move` 的换算必须和 `ScreenMapper` + `cell_to_px` 完全一致 ——
    这里用同一套 API 复算一遍，防止以后有人"优化"成手写算术时改错方向。"""
    geom = _geom()
    m = Mouse(dry_run=True, jitter_px=0, my_side=BLACK)
    for iccs in ("a0a1", "i9i8", "e0e1", "h2e2"):
        (x1, y1), (x2, y2) = m.play_move(iccs, geom)
        mapper = ScreenMapper(BLACK)
        src_rank, src_file = int(iccs[1]), ord(iccs[0]) - ord("a")
        dst_rank, dst_file = int(iccs[3]), ord(iccs[2]) - ord("a")
        assert (x1, y1) == geom.cell_to_px(*mapper.to_screen(src_rank, src_file))
        assert (x2, y2) == geom.cell_to_px(*mapper.to_screen(dst_rank, dst_file))
