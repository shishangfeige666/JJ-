"""CLI 与主循环辅助函数的测试。

`main()` 的完整循环要真窗口 + 真引擎，放在 Task 19 的端到端演练里跑；
这里把「参数解析」「执色判定」「坏帧重抓」「记谱路径」这些**不依赖窗口**的
部分单独测干净 —— 它们恰恰是最容易写错的地方。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess.config import DEFAULT_HUMAN_DELAY, parse_human_delay
from jjchess.geometry import BLACK, RED, BoardGeometry
from jjchess.main import (
    build_parser,
    count_pieces,
    detect_side,
    grab_usable,
    main,
    record_path,
    resolve_side,
)
from jjchess.window import WindowNotFound

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OPENING_PNG = FIXTURES / "opening.png"

_FAKE_GEOM = BoardGeometry(origin=(100.0, 100.0), dx=80.0, dy=80.0, cell_px=64.0)


def _load_fixture(name: str) -> np.ndarray:
    data = np.fromfile(str(FIXTURES / name), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"读不出 fixture：{name}"
    return img


# ============================================================ 参数解析


def test_help_不崩溃(capsys):
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["--help"])
    assert ei.value.code == 0
    assert "--human-delay" in capsys.readouterr().out


def test_默认值():
    a = build_parser().parse_args([])
    assert a.side == "auto"
    assert a.movetime == 1200
    assert a.threads == 8
    assert a.hash == 512
    assert a.dry_run is False
    assert a.no_panel is False
    assert a.no_record is False
    assert a.debug_dir is None
    assert a.human_delay is None, "不传 --human-delay 就是不拟人化"


def test_human_delay裸开关():
    a = build_parser().parse_args(["--human-delay"])
    assert a.human_delay == ""
    assert parse_human_delay(a.human_delay) == DEFAULT_HUMAN_DELAY


def test_human_delay带区间():
    a = build_parser().parse_args(["--human-delay", "1-4"])
    assert parse_human_delay(a.human_delay) == (1.0, 4.0)


def test_非法side被拒():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--side", "green"])


def test_非法human_delay被拒(capsys):
    assert main(["--human-delay", "4-1"]) == 2
    assert "--human-delay" in capsys.readouterr().err


def test_命令行没出现的开关不覆盖配置文件():
    from jjchess.main import build_overrides

    args = build_parser().parse_args([])
    ov = build_overrides(args, [])
    assert ov["movetime_ms"] is None
    assert ov["threads"] is None
    assert ov["hash_mb"] is None
    assert ov["record"] is None
    assert ov["debug_dir"] is None
    assert ov["human_delay"] is None


def test_命令行出现的开关才覆盖():
    from jjchess.main import build_overrides

    raw = ["--movetime", "3000", "--threads=4", "--hash", "128", "--no-record", "--debug-dir", "d"]
    args = build_parser().parse_args(raw)
    ov = build_overrides(args, raw)
    assert ov["movetime_ms"] == 3000
    assert ov["threads"] == 4
    assert ov["hash_mb"] == 128
    assert ov["record"] is False
    assert ov["debug_dir"] == Path("d")


def test_配置文件在命令行没写时能生效(tmp_path):
    """这条是 `build_overrides` 存在的全部理由。

    几个开关的 argparse 默认值（1200/8/512）与 `DEFAULTS` 相同，如果无脑把
    `args` 全丢给 `load_config`，用户改了配置文件却不见效 —— 极难排查。
    """
    import json

    from jjchess.config import load_config
    from jjchess.main import build_overrides

    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(
        json.dumps({"engine": {"movetime_ms": 3000, "threads": 2}}), encoding="utf-8"
    )

    raw = ["--config", str(cfg_file)]
    args = build_parser().parse_args(raw)
    cfg = load_config(cfg_file, build_overrides(args, raw))
    assert cfg.movetime_ms == 3000
    assert cfg.threads == 2


# ============================================================ 执色判定


def test_找不到窗口时返回非零并打印中文提示(monkeypatch, capsys):
    def _boom(*args, **kwargs):
        raise WindowNotFound("未找到标题为「JJ象棋」的窗口，请先打开微信小游戏")

    monkeypatch.setattr("jjchess.main.find_window", _boom)
    assert main(["--no-panel"]) == 2
    assert "未找到" in capsys.readouterr().err


def test_calibrate转交标定工具不找窗口(monkeypatch):
    called: list[list[str]] = []

    def _fake(args):
        called.append(list(args))
        return 0

    def _boom(*args, **kwargs):
        raise AssertionError("--calibrate 不该去找游戏窗口")

    monkeypatch.setattr("jjchess.main.find_window", _boom)
    monkeypatch.setattr("jjchess.main._run_calibrate", _fake)
    assert main(["--calibrate"]) == 0
    assert called, "应该把控制权交给标定工具"


def _synthetic_bottom_row(
    ink_bgr: tuple[int, int, int], rows: int = 10, cols: int = 9
) -> np.ndarray:
    """画一张只有最下面一排有子的合成图，用来测执色判定。"""
    geom = _FAKE_GEOM
    img = np.full((1000, 900, 3), (120, 170, 215), np.uint8)  # 木色底
    for col in range(cols):
        x, y = geom.cell_to_px(rows - 1, col)
        cv2.circle(img, (int(x), int(y)), 26, (150, 195, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 26, ink_bgr, 2, cv2.LINE_AA)
        # 一个「十」字形的笔画，模拟棋子上的字
        cv2.line(img, (int(x) - 14, int(y)), (int(x) + 14, int(y)), ink_bgr, 5, cv2.LINE_AA)
        cv2.line(img, (int(x), int(y) - 14), (int(x), int(y) + 14), ink_bgr, 5, cv2.LINE_AA)
    return img


def test_合成图_红字在下判红方():
    img = _synthetic_bottom_row((40, 40, 200))
    assert detect_side(img, _FAKE_GEOM) == RED


def test_合成图_黑字在下判黑方():
    img = _synthetic_bottom_row((30, 30, 30))
    assert detect_side(img, _FAKE_GEOM) == BLACK


def test_空盘判红方():
    """一格子都没有（比如大厅画面）时给个安全的默认值，不能抛异常。"""
    img = np.full((1000, 900, 3), (120, 170, 215), np.uint8)
    assert detect_side(img, _FAKE_GEOM) == RED


@pytest.mark.skipif(not OPENING_PNG.is_file(), reason="缺少真机开局帧")
def test_真机开局帧判红方():
    img = _load_fixture("opening.png")
    from jjchess.vision.locator import locate

    geom = locate(img)
    assert detect_side(img, geom) == RED


@pytest.mark.skipif(not OPENING_PNG.is_file(), reason="缺少真机开局帧")
def test_上下翻面后判黑方():
    """把真机帧上下翻过来，最下面一排就变成黑方的子 —— 应该判成执黑。

    这一条同时验证了「游戏总是把我方画在下方」这个假设的另一半：
    翻转后的几何也要跟着翻，否则格子根本对不上。
    """
    img = _load_fixture("opening.png")
    from jjchess.vision.locator import locate

    geom = locate(img)
    height = img.shape[0]
    flipped = cv2.flip(img, 0)
    flipped_geom = BoardGeometry(
        origin=(geom.origin[0], height - 1 - geom.origin[1] - 9 * geom.dy),
        dx=geom.dx,
        dy=geom.dy,
        cell_px=geom.cell_px,
    )
    assert detect_side(flipped, flipped_geom) == BLACK


def test_显式指定执色时忽略画面():
    assert resolve_side("red", None, None) == RED
    assert resolve_side("black", None, None) == BLACK


def test_auto时才看图():
    img = _synthetic_bottom_row((30, 30, 30))
    assert resolve_side("auto", img, _FAKE_GEOM) == BLACK


# ============================================================ 开局判定


@pytest.mark.skipif(not OPENING_PNG.is_file(), reason="缺少真机开局帧")
def test_开局帧数出32个棋子():
    img = _load_fixture("opening.png")
    from jjchess.vision.locator import locate

    assert count_pieces(img, locate(img)) == 32


@pytest.mark.skipif(not (FIXTURES / "lobby.png").is_file(), reason="缺少大厅帧")
def test_大厅帧数不出32个棋子():
    img = _load_fixture("lobby.png")
    from jjchess.vision.locator import BoardNotFound, locate

    try:
        geom = locate(img)
    except BoardNotFound:
        return  # 定位阶段就被拒了，更理想
    assert count_pieces(img, geom) != 32


# ============================================================ 坏帧重抓


class _FakeWindow:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        return self.frames[min(self.grabs - 1, len(self.frames) - 1)]


def _blank() -> np.ndarray:
    return np.zeros((300, 200, 3), np.uint8)


def _good(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)


def test_坏帧会被跳过并重抓():
    window = _FakeWindow([_blank(), _blank(), _good()])
    img = grab_usable(window, attempts=5, delay=0.0)
    assert img is not None
    assert window.grabs == 3


def test_连续坏帧最终返回None():
    window = _FakeWindow([_blank()])
    assert grab_usable(window, attempts=3, delay=0.0) is None
    assert window.grabs == 3, "重试次数要按 attempts 来，不能无限重抓"


def test_好帧一次就够():
    window = _FakeWindow([_good()])
    assert grab_usable(window, attempts=3, delay=0.0) is not None
    assert window.grabs == 1


# ============================================================ 记谱路径


def test_记谱路径默认写到records():
    stamp = dt.datetime(2026, 10, 11, 15, 30, 22)
    path = record_path(None, stamp)
    assert path == Path("records") / "game-20261011-153022.txt"


def test_记谱路径跟随debug_dir(tmp_path):
    stamp = dt.datetime(2026, 10, 11, 15, 30, 22)
    path = record_path(tmp_path, stamp)
    assert path == tmp_path / "game-20261011-153022.txt"
