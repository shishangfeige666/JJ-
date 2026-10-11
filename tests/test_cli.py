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

from jjchess.config import DEFAULT_HUMAN_DELAY, Config, parse_human_delay
from jjchess.engine.uci import Analysis
from jjchess.geometry import BLACK, RED, BoardGeometry, ScreenMapper
from jjchess.main import (
    EXIT_NO_FRAME,
    EXIT_OK,
    FrameSettle,
    _next_round,
    _play,
    _retry_unknown,
    build_parser,
    count_pieces,
    detect_side,
    dump_frame,
    geom_close,
    grab_usable,
    log_state,
    main,
    record_path,
    resolve_side,
    screen_geometry,
    trace_frame,
)
from jjchess.rules.board import START_FEN, Board
from jjchess.session import State
from jjchess.vision.recognizer import Recognition
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


# ============================================================ 状态日志


class _FakeSession:
    def __init__(self, state, message: str = "") -> None:
        self.state = state
        self.message = message


def test_状态变化会打印一行(capsys):
    """`--no-panel` 时这是唯一能看出会话卡在哪儿的窗口。"""
    s = _FakeSession(State.THINKING, "正在思考")
    assert log_state(s, None) is State.THINKING
    assert "[THINKING]  正在思考" in capsys.readouterr().out


def test_状态没变就不打印(capsys):
    s = _FakeSession(State.MY_TURN)
    log_state(s, State.MY_TURN)
    assert capsys.readouterr().out == ""


def test_没有消息时不留下多余空格(capsys):
    s = _FakeSession(State.PLAYING)
    log_state(s, State.THINKING)
    assert capsys.readouterr().out.strip() == "[PLAYING]"


# ============================================================ 客户区 → 屏幕


def test_屏幕几何把客户区原点加回去():
    """不换算的话点击会落到另一块显示器上 —— 而且干跑看不出来。

    本机 JJ象棋 的客户区原点在 `(-1440, -279)`（竖屏副显示器）。
    """
    screen = screen_geometry(_FAKE_GEOM, (-1440, -279))
    assert screen is not None
    assert screen.origin == (-1340.0, -179.0)
    # 间距与棋子直径不受平移影响
    assert (screen.dx, screen.dy, screen.cell_px) == (80.0, 80.0, 64.0)
    assert screen.cell_to_px(7, 7) == (100.0 + 7 * 80 - 1440, 100.0 + 7 * 80 - 279)


def test_屏幕几何对None保持None():
    assert screen_geometry(None, (-1440, -279)) is None


# ============================================================ 现场取证


class _FakeRec:
    def __init__(self, fen: str) -> None:
        self.board = Board.from_fen(fen)


def test_画面FEN变了就打一行(capsys):
    """模型和画面分歧时，这一行是唯一能对上账的证据。"""
    rec = _FakeRec(START_FEN)
    assert trace_frame(rec, None) == START_FEN
    assert f"[画面] {START_FEN}" in capsys.readouterr().out


def test_画面FEN没变就不打印(capsys):
    rec = _FakeRec(START_FEN)
    trace_frame(rec, START_FEN)
    assert capsys.readouterr().out == ""


def test_画面为空时保持上一次的FEN(capsys):
    assert trace_frame(None, START_FEN) == START_FEN
    assert capsys.readouterr().out == ""


def test_没有debugdir时不存图(tmp_path):
    cfg = Config(debug_dir=None)
    dump_frame(np.zeros((4, 4, 3), np.uint8), cfg, "paused")
    assert list(tmp_path.iterdir()) == []


def test_暂停时把画面存成png(tmp_path):
    """VERIFY 失败那一刻的画面必须留下来，否则事后只能靠猜。"""
    cfg = Config(debug_dir=tmp_path)
    img = np.full((6, 8, 3), 7, np.uint8)
    dump_frame(img, cfg, "paused")
    files = list(tmp_path.glob("frame-*-paused.png"))
    assert len(files) == 1
    back = cv2.imdecode(np.fromfile(str(files[0]), np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == (6, 8, 3)
    assert int(back[0, 0, 0]) == 7


# ============================================================ 画面停稳再上报


def test_第一帧不算停稳():
    """刚看到的局面不能立刻当真 —— 说不定棋子还在滑。"""
    gate = FrameSettle(seconds=0.3)
    assert gate.feed("A", 100.0) is False


def test_同一个局面持续够久才算停稳():
    gate = FrameSettle(seconds=0.3)
    gate.feed("A", 100.0)
    assert gate.feed("A", 100.29) is False
    assert gate.feed("A", 100.31) is True


def test_过渡帧会把计时重置():
    """这正是真机踩到的坑：动画中途的每一帧都是「新局面」。"""
    gate = FrameSettle(seconds=0.3)
    gate.feed("A", 100.0)
    gate.feed("B", 100.4)  # 棋子滑到一半
    gate.feed("C", 100.8)  # 还在滑
    assert gate.feed("C", 100.9) is False
    assert gate.feed("C", 101.11) is True


def test_停稳之后新局面的第一帧仍然不算停稳():
    gate = FrameSettle(seconds=0.3)
    gate.feed("A", 100.0)
    assert gate.feed("A", 100.4) is True
    assert gate.feed("B", 100.5) is False


def test_屏幕几何原点为零时与客户区几何等价():
    assert screen_geometry(_FAKE_GEOM, (0, 0)) == _FAKE_GEOM


# ============================================================ 主循环收尾


class _LoopWindow:
    """`_play` 主循环要的最小窗口：抓图 + 尺寸 + 客户区原点。"""

    hwnd = 1
    title = "JJ象棋"
    client_origin = (0, 0)
    client_size = (200, 300)

    def grab(self) -> np.ndarray:
        return _good()

    def refresh(self) -> None:
        pass

    def size_changed(self, width: int, height: int) -> bool:
        return False


class _NoKeys:
    """热键桩：一次都不按。"""

    def poll(self) -> dict[str, bool]:
        return {}


class _ResizeWindow(_LoopWindow):
    """抓够 `after` 帧之后开始报「窗口尺寸变了」。

    真机上把 `geom` 清成 `None` 的是结算弹窗（`locate()` 抛 `BoardNotFound`），
    这里用 `size_changed` 走同一条清空路径 —— 两者都会把循环逼进那个早退分支。
    """

    def __init__(self, after: int) -> None:
        self.after = after
        self.checks = 0

    def size_changed(self, width: int, height: int) -> bool:
        self.checks += 1
        return self.checks > self.after


class _StubEngine:
    """引擎桩：永远给同一步合法着法。主循环测试不关心棋力，只关心它不崩。"""

    def analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis:
        return Analysis(bestmove="h2e2", score_cp=0, depth=1)


def _loop_recognition() -> Recognition:
    return Recognition(
        board=Board.from_fen(START_FEN),
        confidence=np.ones((10, 9), np.float32),
        unknown=[],
    )


def test_终局后主循环收尾退出(tmp_path, monkeypatch, capsys):
    """spec 的终局检测写的是「判定对局结束，**停止**并打印对局记录」。

    真机上少了这个 break：对局结束后进程一直空转打日志，`_finish()` 永远不跑，
    记谱文件永远不落盘。这条测试把「画面还在、但已经认不出棋盘」的结算现场
    搬进主循环，断言循环会自己收摊返回 `EXIT_OK`。

    没有 break 时循环不会停 —— 所以让 `grab_usable` 在抓够次数之后开始返回 `None`，
    让它以 `EXIT_NO_FRAME` 收场，测试是失败而不是卡死。
    """
    geom = _FAKE_GEOM
    rec = _loop_recognition()

    calls = {"recognize": 0, "grab": 0}

    def fake_recognize(*_args, **_kwargs):
        calls["recognize"] += 1
        return rec if calls["recognize"] == 1 else None

    def fake_grab(*_args, **_kwargs):
        calls["grab"] += 1
        return _good() if calls["grab"] <= 40 else None

    monkeypatch.setattr("jjchess.main.recognize", fake_recognize)
    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.locate", lambda _img: geom)
    monkeypatch.setattr("jjchess.main.Hotkeys", _NoKeys)
    monkeypatch.setattr("jjchess.main.LOOP_SLEEP", 0.0)
    monkeypatch.setattr("jjchess.main.MAX_BAD_STREAK", 3)
    # 6 秒的棋盘失联额度在这条测试里没有意义，拨到 0 免得白等。
    monkeypatch.setattr("jjchess.session.BOARD_LOST_TIMEOUT_S", 0.0)
    monkeypatch.setattr(
        "jjchess.main._wait_for_board",
        lambda *_a, **_k: (None, geom, rec, ScreenMapper(RED), None, RED),
    )

    args = build_parser().parse_args(["--no-panel", "--dry-run", "--debug-dir", str(tmp_path)])
    cfg = Config(debug_dir=tmp_path)

    assert _play(args, cfg, _LoopWindow(), None) == EXIT_OK
    assert "对局结束" in capsys.readouterr().out


def test_棋盘定位不到时终局也要收尾(tmp_path, monkeypatch, capsys):
    """收尾判断必须放在循环**开头**，否则会被早退分支绕过去。

    这是真机踩出来的：对局结束时画面切到结算弹窗，`locate()` 连续两次都抛
    `BoardNotFound`，于是走的是「`geom is None` → 定位不到 → `session.step()` →
    `continue`」那条早退分支。收尾判断原来写在循环末尾，`continue` 直接跳过它，
    进程就在那里空转 —— 状态不再变化，终端一行都不打，记谱永远不落盘。
    放到开头之后，无论从哪条路走到终局，下一轮都会收尾。
    """
    from jjchess.vision.locator import BoardNotFound

    geom = _FAKE_GEOM
    rec = _loop_recognition()
    calls = {"grab": 0}

    def fake_grab(*_args, **_kwargs):
        calls["grab"] += 1
        return _good() if calls["grab"] <= 40 else None

    def boom(_img):
        raise BoardNotFound("画面里没有棋盘")

    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.recognize", lambda *_a, **_k: rec)
    monkeypatch.setattr("jjchess.main.locate", boom)
    # 关掉定期复查，专测「窗口尺寸变了 → geom 清空 → 定位不到」这条早退路径。
    monkeypatch.setattr("jjchess.main.RELOCATE_EVERY_S", 1e9)
    monkeypatch.setattr("jjchess.main.Hotkeys", _NoKeys)
    monkeypatch.setattr("jjchess.main.LOOP_SLEEP", 0.0)
    monkeypatch.setattr("jjchess.main.MAX_BAD_STREAK", 3)
    monkeypatch.setattr("jjchess.session.BOARD_LOST_TIMEOUT_S", 0.0)
    monkeypatch.setattr(
        "jjchess.main._wait_for_board",
        lambda *_a, **_k: (None, geom, rec, ScreenMapper(RED), None, RED),
    )

    args = build_parser().parse_args(["--no-panel", "--dry-run", "--debug-dir", str(tmp_path)])
    cfg = Config(debug_dir=tmp_path)

    # 头一帧正常：`latest["board"]` 初值就是开局局面，会话第一轮就会走出 IDLE 落到
    # 「轮到我方」。之后立刻开始报「尺寸变了」，把循环逼进那条早退分支。
    #
    # 这里必须卡在**刚走出 IDLE 那一轮**，不能再往后放：真机上是「对方回合」时棋盘
    # 消失的，而如果让会话先跑进 `PLAYING`，`geom` 已经没了、点不了鼠标，它会先按
    # 「走子拿不到几何」自动暂停 —— 那是另一条路，测不到这个缺陷。
    assert _play(args, cfg, _ResizeWindow(after=1), _StubEngine()) == EXIT_OK
    out = capsys.readouterr().out
    assert "对局结束" in out


# ================================================================ 多轮对局


class _StubMouse:
    """鼠标桩：记录点了哪里，绝不真的动系统光标。"""

    def __init__(self, dry_run: bool = False, my_side: int = RED) -> None:
        self.dry_run = dry_run
        self.my_side = my_side
        self.clicks: list[tuple[int, int]] = []

    def click_client(self, x: int, y: int, origin) -> None:
        self.clicks.append((x, y))

    def move_to(self, *_a, **_k) -> None:
        pass

    def play_move(self, *_a, **_k) -> None:
        pass


class _QuickSettle:
    """画面稳定闸门桩：立刻放行。真闸门要等 0.5 秒，测试里没这个时间。"""

    def feed(self, _fen: str, _now: float) -> bool:
        return True

    def reset(self) -> None:
        pass


def test_认出结算画面的再来一局按钮():
    """绿色「再来一局」按钮要认得出来 —— 认不出就没法连着下，认错了更糟。

    判定只看底部那条带的颜色。真机实测：结算画面 G=185.0 / B=94.9（差 90.1），
    对局中同一处 G=82.4 / B=70.6（差 11.8），大厅是 G=92.4 / B=54.6。
    """
    from jjchess.main import is_result_screen

    def load(name: str):
        return cv2.imdecode(np.fromfile(FIXTURES / name, np.uint8), cv2.IMREAD_COLOR)

    assert is_result_screen(load("result.png")), "结算画面应该认得出来"
    for name in ("opening.png", "mid_01.png", "lobby.png"):
        assert not is_result_screen(load(name)), f"{name} 不是结算画面"
    assert not is_result_screen(None)

    # 再钉一遍数值：绿色通道和「绿减蓝」两个门槛都得真起作用。
    blank = np.zeros((2560, 1440, 3), np.uint8)
    y = int(0.910 * 2560)
    band = (slice(y - 12, y + 12), slice(432, 1008))
    blank[band] = (70, 82, 81)  # 对局中的木头色
    assert not is_result_screen(blank)
    blank[band] = (90, 172, 164)  # 结算画面的按钮绿
    assert is_result_screen(blank)
    blank[band] = (90, 130, 200)  # 够绿但绿蓝差不够（比如偏蓝的其它按钮）
    assert not is_result_screen(blank)


def _patch_round_loop(monkeypatch, geom, rec, tmp_path, extra: list[str]):
    """把主循环的一局做成「一帧有棋盘，然后一直认不到」的桩。"""
    ctl = {"give": 1, "grab": 0}
    starts: list[int] = []

    def fake_recognize(*_a, **_k):
        if ctl["give"] > 0:
            ctl["give"] -= 1
            return rec
        return None

    def fake_grab(*_a, **_k):
        ctl["grab"] += 1
        return _good() if ctl["grab"] <= 200 else None

    monkeypatch.setattr("jjchess.main.recognize", fake_recognize)
    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.locate", lambda _img: geom)
    monkeypatch.setattr("jjchess.main.Hotkeys", _NoKeys)
    monkeypatch.setattr("jjchess.main.LOOP_SLEEP", 0.0)
    monkeypatch.setattr("jjchess.main.MAX_BAD_STREAK", 3)
    monkeypatch.setattr("jjchess.main.FrameSettle", _QuickSettle)
    monkeypatch.setattr("jjchess.session.BOARD_LOST_TIMEOUT_S", 0.0)
    monkeypatch.setattr(
        "jjchess.main._wait_for_board",
        lambda *_a, **_k: (None, geom, rec, ScreenMapper(RED), None, RED),
    )
    monkeypatch.setattr("jjchess.main.Mouse", _StubMouse)
    args = build_parser().parse_args(["--no-panel", "--debug-dir", str(tmp_path), *extra])
    return args, ctl, starts


def test_只下一局时不碰再来一局(tmp_path, monkeypatch, capsys):
    """`--rounds 1` 是旧行为：下完一局收工，绝不去点结算画面。"""
    geom = _FAKE_GEOM
    rec = _loop_recognition()
    args, _ctl, _starts = _patch_round_loop(monkeypatch, geom, rec, tmp_path, ["--rounds", "1"])

    called: list[int] = []
    monkeypatch.setattr("jjchess.main._next_round", lambda *_a, **_k: called.append(1) or geom)

    assert _play(args, Config(debug_dir=tmp_path), _LoopWindow(), _StubEngine()) == EXIT_OK
    assert called == [], "--rounds 1 不该去开新局"
    out = capsys.readouterr().out
    assert "对局结束" in out
    assert "已下满 1 局" in out


def test_默认下完一局自动开下一局(tmp_path, monkeypatch, capsys):
    """默认（`--rounds 0`）要连着下：终局之后去点「再来一局」。"""
    geom = _FAKE_GEOM
    rec = _loop_recognition()
    args, _ctl, _starts = _patch_round_loop(monkeypatch, geom, rec, tmp_path, [])

    called: list[int] = []
    # 第二局开不起来（返回 None）→ 循环收工。这条测试只关心「有没有去开」。
    monkeypatch.setattr("jjchess.main._next_round", lambda *_a, **_k: called.append(1))

    assert _play(args, Config(debug_dir=tmp_path), _LoopWindow(), _StubEngine()) == EXIT_OK
    assert called == [1], "终局之后应该去开新局"


def test_再来一局后第二局用新的会话与记谱(tmp_path, monkeypatch, capsys):
    """连着下时，第二局必须是全新的会话 + 全新的记谱。

    复用会话的话，它会带着上一局的模型去认新一局的开局；复用记谱的话，
    第二局的着法会追加进第一局的文件里。
    """
    geom = _FAKE_GEOM
    rec = _loop_recognition()
    args, ctl, _starts = _patch_round_loop(monkeypatch, geom, rec, tmp_path, ["--rounds", "2"])
    seen: list[object] = []

    def fake_next_round(*_a, **_k):
        seen.append(1)
        ctl["give"] = 1  # 新一局的开局能看见一帧
        return geom

    monkeypatch.setattr("jjchess.main._next_round", fake_next_round)

    assert _play(args, Config(debug_dir=tmp_path), _LoopWindow(), _StubEngine()) == EXIT_OK
    out = capsys.readouterr().out
    assert seen == [1], "应该只开一次新局（下满 2 局就收工）"
    assert "第 2 局开始" in out
    assert "已下满 2 局" in out
    assert out.count("，正在收尾…") == 2, "两局都要各自收一次尾"


def test_等不到结算画面就收工(tmp_path, monkeypatch, capsys):
    """点不到「再来一局」时**宁可收工也不能盲点** —— 那个位置在对局中是棋盘。"""
    mouse = _StubMouse()
    monkeypatch.setattr("jjchess.main.grab_usable", lambda *_a, **_k: _good())
    monkeypatch.setattr("jjchess.main.RESULT_WAIT_S", 0.0)

    assert _next_round(_LoopWindow(), _NoKeys(), mouse) is None
    assert mouse.clicks == [], "认不出结算画面就不该点下去"
    assert "等不到" in capsys.readouterr().err


def test_彩排模式不自动开新局(tmp_path, monkeypatch, capsys):
    """彩排本来就没真点鼠标，新一局不会开始，别在那里空等 20 秒。"""
    mouse = _StubMouse(dry_run=True)
    monkeypatch.setattr("jjchess.main.grab_usable", lambda *_a, **_k: _good())

    assert _next_round(_LoopWindow(), _NoKeys(), mouse) is None
    assert mouse.clicks == []
    assert "彩排模式" in capsys.readouterr().out


def test_认出结算画面后点再来一局并等新棋盘(tmp_path, monkeypatch, capsys):
    """完整的一轮：认出按钮 → 点它 → 等新棋盘出现 → 把新几何交回去。"""
    from jjchess.main import PLAY_AGAIN_POINT

    geom = _FAKE_GEOM
    result = cv2.imdecode(np.fromfile(FIXTURES / "result.png", np.uint8), cv2.IMREAD_COLOR)
    mouse = _StubMouse()
    frames = {"n": 0}

    def fake_grab(*_a, **_k):
        frames["n"] += 1
        # 前两帧是结算画面，之后新一局的棋盘出来了。
        return result if frames["n"] <= 2 else _good()

    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.locate", lambda _img: geom)

    got = _next_round(_LoopWindow(), _NoKeys(), mouse)
    assert got is geom
    assert len(mouse.clicks) == 1
    x, y = mouse.clicks[0]
    assert (x, y) == (
        int(PLAY_AGAIN_POINT[0] * result.shape[1]),
        int(PLAY_AGAIN_POINT[1] * result.shape[0]),
    )
    out = capsys.readouterr().out
    assert "已点「再来一局」" in out
    assert "新一局棋盘定位" in out


def test_空的装饰棋盘不算找到棋盘(tmp_path, monkeypatch, capsys):
    """`locate()` 会在「正在匹配」那类页面的空棋盘上拟出网格。

    真机实测：点完「再来一局」游戏先过一屏「正在匹配」，那上面一块**空的**装饰棋盘
    被 `locate()` 认了出来（格距 102.7，真棋盘 125.8）。认下来的话第二局会一直卡在
    IDLE 里等一个永远等不到的开局 —— 真机就是这么卡住的。
    """
    geom = _FAKE_GEOM
    result = cv2.imdecode(np.fromfile(FIXTURES / "result.png", np.uint8), cv2.IMREAD_COLOR)
    mouse = _StubMouse()
    frames = {"n": 0}

    def fake_grab(*_a, **_k):
        frames["n"] += 1
        return result if frames["n"] <= 1 else _good()

    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.locate", lambda _img: geom)
    monkeypatch.setattr("jjchess.main.count_pieces", lambda *_a, **_k: 0)
    monkeypatch.setattr("jjchess.main.NEW_BOARD_WAIT_S", 0.0)

    assert _next_round(_LoopWindow(), _NoKeys(), mouse) is None
    assert "等不到棋盘" in capsys.readouterr().err


def test_主循环里空棋盘也不认(tmp_path, monkeypatch, capsys):
    """主循环同样：定位成功但盘上一颗子都没有，就当没找到棋盘，别拿去识别。"""
    geom = _FAKE_GEOM
    rec = _loop_recognition()
    calls = {"recognize": 0, "grab": 0}

    def fake_recognize(*_a, **_k):
        calls["recognize"] += 1
        return rec

    def fake_grab(*_a, **_k):
        calls["grab"] += 1
        return _good() if calls["grab"] <= 8 else None

    monkeypatch.setattr("jjchess.main.recognize", fake_recognize)
    monkeypatch.setattr("jjchess.main.grab_usable", fake_grab)
    monkeypatch.setattr("jjchess.main.locate", lambda _img: geom)
    monkeypatch.setattr("jjchess.main.count_pieces", lambda *_a, **_k: 0)
    monkeypatch.setattr("jjchess.main.Hotkeys", _NoKeys)
    monkeypatch.setattr("jjchess.main.LOOP_SLEEP", 0.0)
    monkeypatch.setattr("jjchess.main.MAX_BAD_STREAK", 3)
    monkeypatch.setattr(
        "jjchess.main._wait_for_board",
        # 首帧就没有几何 —— 逼主循环自己走一遍「定位 → 数子 → 认不认」。
        lambda *_a, **_k: (None, None, rec, ScreenMapper(RED), None, RED),
    )
    args = build_parser().parse_args(["--no-panel", "--rounds", "1", "--debug-dir", str(tmp_path)])

    assert _play(args, Config(debug_dir=tmp_path), _LoopWindow(), _StubEngine()) == EXIT_NO_FRAME
    assert calls["recognize"] == 0, "空棋盘不该拿去识别"
    assert "棋盘重新定位" not in capsys.readouterr().out


# ------------------------------------------------------- 几何漂移与认不出的格子
#
# 真机第二次卡死（停了十几分钟不动）的两个成因，各配一条测试钉住。


def test_几何只偏一点点就换过去():
    """换局那一下拟合出来的网格可能偏几像素，复查时应当换过去自愈。

    真机实测：第 2 局的几何是 `origin=(101.61, 602.69), dx=154.56`，正常应当是
    `origin≈(101.5, 609.3)` —— 纵向偏了 6 像素。
    """
    old = BoardGeometry(origin=(101.607, 602.686), dx=154.558, dy=148.012, cell_px=125.810)
    better = BoardGeometry(origin=(101.523, 609.322), dx=154.573, dy=148.059, cell_px=125.850)
    assert geom_close(better, old)


def test_装饰棋盘的几何不算同一个棋盘():
    """「正在匹配」页面上那块空棋盘拟合出来的格距只有 121，差得太多，绝不能换过去。"""
    real = BoardGeometry(origin=(101.523, 609.322), dx=154.573, dy=148.059, cell_px=125.850)
    fake = BoardGeometry(origin=(241.30, 380.44), dx=121.35, dy=120.77, cell_px=102.65)
    assert not geom_close(fake, real)


def test_认不出来的格子会换个几何再认一遍(monkeypatch):
    """`recognize` 把认不出的格子当**空**，换个几何认干净了就必须换过去。

    真机现场：几何偏 6 像素时红方底线那格（黑車刚落上去）置信度只有 **0.366** 被判
    `unknown`，于是「画面上多了一个子」在会话眼里变成「什么都没变」，`_grid_diff == 0`
    直接早退，脚本停在那儿一动不动。用当时那份几何重跑识别，得到的正是模型里那个盘子。
    """
    from jjchess.vision.locator import BoardNotFound  # noqa: F401  （桩里会抛）
    from jjchess.vision.recognizer import Recognition

    stale = BoardGeometry(origin=(101.607, 602.686), dx=154.558, dy=148.012, cell_px=125.810)
    fresh = BoardGeometry(origin=(100.328, 608.907), dx=154.730, dy=148.140, cell_px=125.980)
    dirty = Recognition(
        board=Board.from_fen(START_FEN), confidence=np.ones((10, 9), np.float32), unknown=[(9, 1)]
    )
    clean = Recognition(
        board=Board.from_fen(START_FEN), confidence=np.ones((10, 9), np.float32), unknown=[]
    )

    monkeypatch.setattr("jjchess.main.locate", lambda _img: fresh)
    monkeypatch.setattr("jjchess.main.count_pieces", lambda *_a, **_k: 16)
    monkeypatch.setattr("jjchess.main.recognize", lambda *_a, **_k: clean)

    got = _retry_unknown(_good(), stale, None, ScreenMapper(RED), Config(), dirty)
    assert got is clean, "换个几何认干净了就该用新的那份"


def test_换了几何还是认不出来就原样返回(monkeypatch, capsys):
    """认不干净时不能把这一帧丢掉 —— 丢掉会被会话读成「棋盘没了」，10 秒后误判终局。"""
    from jjchess.vision.recognizer import Recognition

    stale = _FAKE_GEOM
    dirty = Recognition(
        board=Board.from_fen(START_FEN), confidence=np.ones((10, 9), np.float32), unknown=[(9, 1)]
    )
    still = Recognition(
        board=Board.from_fen(START_FEN), confidence=np.ones((10, 9), np.float32), unknown=[(9, 1)]
    )

    monkeypatch.setattr("jjchess.main.locate", lambda _img: stale)
    monkeypatch.setattr("jjchess.main.count_pieces", lambda *_a, **_k: 16)
    monkeypatch.setattr("jjchess.main.recognize", lambda *_a, **_k: still)
    monkeypatch.setattr("jjchess.main._unknown_warn_at", 0.0)

    got = _retry_unknown(_good(), stale, None, ScreenMapper(RED), Config(), dirty)
    assert got is dirty
    assert "格认不出来" in capsys.readouterr().err
