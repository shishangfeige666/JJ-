"""`jjchess.ui` 的测试：合成图是纯函数，热键用可注入的按下状态。

面板本身依赖 `cv2.imshow`，无头环境跑不了，所以**所有断言都打在 `compose()` 的
返回值上**，`Panel` 只测「关掉显示时仍返回合成图」与「重复 close 不炸」。
"""

from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess.config import DEFAULTS
from jjchess.geometry import RED, BoardGeometry
from jjchess.rules.board import START_FEN, Board
from jjchess.session import State
from jjchess.ui.hotkeys import _KEYS, FakeHotkeys, Hotkeys
from jjchess.ui.panel import Panel, compose
from jjchess.vision.recognizer import Recognition

WIDTH, HEIGHT = 1500, 900
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OPENING_PNG = FIXTURES / "opening.png"

_GEOM = BoardGeometry(origin=(100.0, 100.0), dx=80.0, dy=80.0, cell_px=64.0)


def _load_fixture(name: str) -> np.ndarray:
    """`cv2.imread` 在非 ASCII 路径上会静默返回 None，一律走 imdecode。"""
    data = np.fromfile(str(FIXTURES / name), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"读不出 fixture：{name}"
    return img


def _fake_session(**overrides) -> types.SimpleNamespace:
    base = {
        "state": State.IDLE,
        "message": "",
        "board": Board.from_fen(START_FEN),
        "last_analysis": None,
        "my_side": RED,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _fake_recognition() -> Recognition:
    conf = np.full((10, 9), 0.95, dtype=np.float32)
    conf[5, 3] = 0.65  # 中间档：黄框
    conf[4, 1] = 0.40  # 低档：红框
    return Recognition(
        board=Board.from_fen(START_FEN),
        confidence=conf,
        unknown=[(4, 1)],
    )


# ============================================================ compose


def test_合成图尺寸固定():
    out = compose(None, None, None, _fake_session(), WIDTH, HEIGHT)
    assert out.shape == (HEIGHT, WIDTH, 3)


def test_自定义尺寸也照做():
    out = compose(None, None, None, _fake_session(), 800, 480)
    assert out.shape == (480, 800, 3)


def test_有帧时左半区非空():
    frame = _load_fixture("opening.png")
    from jjchess.vision.locator import locate

    geom = locate(frame)
    out = compose(frame, geom, _fake_recognition(), _fake_session(), WIDTH, HEIGHT)
    assert out[:, :900].std() > 5, "左半应该是缩放后的棋盘，不是纯色"


def test_右半含关键信息文字():
    s = _fake_session(state=State.THINKING, message="THINKING (human delay 1.3s)")
    out = compose(None, None, None, s, WIDTH, HEIGHT)
    assert out[:, 900:].std() > 5, "右半应该画了文字"


def test_无帧时不崩溃():
    out = compose(None, None, None, _fake_session(), WIDTH, HEIGHT)
    assert out is not None and out.shape == (HEIGHT, WIDTH, 3)


def test_有帧无几何时不崩溃():
    frame = _load_fixture("opening.png")
    out = compose(frame, None, _fake_recognition(), _fake_session(), WIDTH, HEIGHT)
    assert out.shape == (HEIGHT, WIDTH, 3)


def test_有几何无识别时不崩溃():
    out = compose(None, _GEOM, None, _fake_session(), WIDTH, HEIGHT)
    assert out.shape == (HEIGHT, WIDTH, 3)


def test_识别叠加会改变左半画面():
    """有 `rec` 与没 `rec` 必须画出不同的左半 —— 证明确实叠了标注。"""
    frame = _load_fixture("opening.png")
    from jjchess.vision.locator import locate

    geom = locate(frame)
    plain = compose(frame, geom, None, _fake_session(), WIDTH, HEIGHT)
    marked = compose(frame, geom, _fake_recognition(), _fake_session(), WIDTH, HEIGHT)
    assert not np.array_equal(plain[:, :900], marked[:, :900])

    from jjchess.ui.panel import status_lines

    assert not any("LOW" in line for line in status_lines(_fake_session(), None))
    assert any("LOW" in line for line in status_lines(_fake_session(), _fake_recognition()))


@pytest.mark.parametrize(
    ("conf", "expect"),
    [
        (0.95, "green"),
        (0.75, "green"),
        (0.70, "yellow"),
        (0.60, "yellow"),
        (0.59, "red"),
        (0.0, "red"),
    ],
)
def test_置信度配色分级(conf: float, expect: str):
    from jjchess.ui.panel import conf_color

    got = conf_color(conf)
    palette = {"green": (80, 220, 80), "yellow": (60, 200, 230), "red": (60, 60, 230)}
    assert got == palette[expect], f"置信度 {conf} 应该是 {expect}"


def test_配色阈值取自配置():
    from jjchess.ui.panel import conf_color

    assert conf_color(DEFAULTS.conf_good + 0.01) == (80, 220, 80)
    assert conf_color(DEFAULTS.conf_min - 0.01) == (60, 60, 230)


def test_状态文字包含状态与消息():
    from jjchess.ui.panel import status_lines

    s = _fake_session(state=State.PAUSED, message="看不懂局面变化（第 3 次）")
    text = "\n".join(status_lines(s))
    assert "PAUSED" in text
    assert "attempt 3" in text, "中文消息要翻成可读英文，不能只剩问号"


def test_状态文字是纯ASCII():
    """`cv2.putText` 画不了中文，画出来是一串问号 —— 必须在进面板前就转掉。"""
    from jjchess.ui.panel import status_lines

    s = _fake_session(state=State.PAUSED, message="轮到我方但局面异常静止，请检查")
    raw = "\n".join(status_lines(s))
    assert raw.isascii(), f"面板文字必须全是 ASCII，实际：{raw!r}"


# ============================================================ hotkeys


def test_热键轮询去重():
    hk = FakeHotkeys(["pause", "pause", "quit"])
    assert hk.poll() == {"pause"}
    assert hk.poll() == {"pause"}
    assert hk.poll() == {"quit"}


def test_假热键脚本用完后返回空集():
    hk = FakeHotkeys(["step"])
    assert hk.poll() == {"step"}
    for _ in range(3):
        assert hk.poll() == set()


def test_假热键接受集合与None():
    hk = FakeHotkeys([{"pause", "quit"}, None])
    assert hk.poll() == {"pause", "quit"}
    assert hk.poll() == set()


def _hotkeys_from(frames: list[set[str]]) -> Hotkeys:
    """按帧脚本驱动真实 `Hotkeys` 的边沿检测（不真的按键）。"""
    vk_names = {vk: name for name, vk in _KEYS.items()}
    cursors = dict.fromkeys(_KEYS, 0)
    series = {name: [name in f for f in frames] for name in _KEYS}

    def is_down(vk: int) -> bool:
        name = vk_names[vk]
        i = cursors[name]
        cursors[name] += 1
        return series[name][i] if i < len(series[name]) else False

    return Hotkeys(is_down=is_down)


def test_真实热键按住不重复触发():
    hk = _hotkeys_from([{"pause"}, {"pause"}, {"pause"}, set(), set()])
    assert hk.poll() == {"pause"}
    assert hk.poll() == set(), "一直按住不该反复触发"
    assert hk.poll() == set()


def test_真实热键松开再按会再触发():
    hk = _hotkeys_from([set(), {"pause"}, set(), {"pause"}])
    assert hk.poll() == set()
    assert hk.poll() == {"pause"}
    assert hk.poll() == set()
    assert hk.poll() == {"pause"}


def test_真实热键能同时报多个键():
    hk = _hotkeys_from([{"pause", "quit"}])
    assert hk.poll() == {"pause", "quit"}


# ============================================================ Panel


def test_面板关闭显示时仍返回合成图():
    p = Panel(enabled=False, width=WIDTH, height=HEIGHT)
    out = p.render(None, None, None, _fake_session())
    assert out.shape == (HEIGHT, WIDTH, 3)
    assert p.frames == 1
    out = p.render(None, None, None, _fake_session())
    assert p.frames == 2


def test_面板重复关闭不炸():
    p = Panel(enabled=False, width=WIDTH, height=HEIGHT)
    p.close()
    p.close()
    p.close()
