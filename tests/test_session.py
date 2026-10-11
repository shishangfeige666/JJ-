"""`jjchess.session` 的单元测试：合法着法解释器、可中断睡眠、状态机鲁棒性。

[RF-2] 抓图失败（全白帧 / 残缺帧）时 `board_provider()` 返回 `None`。
       会话**绝不能**因此改动 `self.board` —— 那会被当成「棋盘上子没了」。
       注意：两种坏帧都在抓图层就被 `window.frame_usable` 挡下并折算成 `None`，
       所以这里一条用例同时覆盖它们。
[RF-4] 局面长时间不变：**轮到我方**时必须判 `PAUSED`（等人来看），
       只有**非我方回合**（对方长考/已离场）才能判 `GAME_OVER`。
"""

from __future__ import annotations

import datetime
import threading
import time

import numpy as np
import pytest

from jjchess.config import DEFAULTS, Config
from jjchess.control import Mouse
from jjchess.engine.uci import Analysis
from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.record import GameRecorder
from jjchess.rules.board import START_FEN, Board
from jjchess.rules.moves import legal_moves, make
from jjchess.session import (
    BOARD_LOST_TIMEOUT_S,
    MAX_UNEXPLAINED,
    VERIFY_TIMEOUT_S,
    Session,
    SessionDeps,
    State,
    interpret,
    interruptible_sleep,
)

_FAKE_GEOM = BoardGeometry(origin=(100.0, 100.0), dx=80.0, dy=80.0, cell_px=64.0)


def _recorder() -> GameRecorder:
    return GameRecorder(RED, START_FEN, datetime.datetime(2026, 10, 11, 12, 0, 0))


# ============================================================ 合法着法解释器


def test_完全一致返回None():
    b = Board.from_fen(START_FEN)
    assert interpret(b, b.copy()) is None


def test_识别出对手着法():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    assert interpret(prev, cur) == "h2e2"


def test_单格误识别被纠正():
    """[核心纠错] 真实着法是 h2e2，但 a0 的红车被识别成空 —— 仍要还原出 h2e2。"""
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    cur.grid[0, 0] = 0
    assert interpret(prev, cur) == "h2e2"


def test_多格误识别在容差内仍能纠正():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    cur.grid[0, 0] = 0
    cur.grid[9, 0] = 0
    assert interpret(prev, cur, tolerance=2) == "h2e2"


def test_超出容差无法解释返回None():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    cur.grid[0, 0] = 0
    cur.grid[0, 1] = 0
    cur.grid[0, 2] = 0
    cur.grid[0, 3] = 0
    assert interpret(prev, cur) is None


def test_不修改入参():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    want_prev = prev.grid.tolist()
    want_cur = cur.grid.tolist()
    interpret(prev, cur)
    assert prev.grid.tolist() == want_prev
    assert cur.grid.tolist() == want_cur


def test_只接受合法着法():
    """`interpret` 只在 `legal_moves` 里搜，所以自将的伪合法着法永远解释不出来。"""
    prev = Board.from_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    assert prev.side == RED
    # 红车在 e1 挡着将帅；挪到 a1 就会让将帅照面（自将），不是合法着法。
    cur = prev.copy()
    rook = int(cur.grid[1, 4])
    cur.grid[1, 4] = 0
    cur.grid[1, 0] = rook
    assert "e1a1" not in legal_moves(prev)
    assert interpret(prev, cur) is None


# ============================================================ 可中断睡眠


def test_睡眠可被提前中断():
    flag = {"v": False}
    t0 = time.monotonic()
    threading.Timer(0.2, lambda: flag.__setitem__("v", True)).start()
    ok = interruptible_sleep(3.0, lambda: flag["v"], slice_s=0.05)
    assert ok is False and time.monotonic() - t0 < 1.0


def test_睡眠睡满返回True():
    t0 = time.monotonic()
    assert interruptible_sleep(0.3, lambda: False, slice_s=0.05) is True
    assert 0.25 < time.monotonic() - t0 < 1.0


def test_睡眠零秒立即返回():
    flag = {"v": False}
    assert interruptible_sleep(0.0, lambda: flag["v"]) is True


def test_睡眠每一小片都会跑一次tick():
    """`tick` 是让 F8 / ESC 在拟人延迟期间**真正即时**的关键。

    `stop()` 读的是 `_pause_requested` / `should_stop` 两个标志，而设置它们的是
    主循环里的热键处理 —— 主循环此刻正阻塞在这个函数里，没人去读键盘。
    不给 `tick` 的话 `stop()` 永远看不到新按键，只能睡满（真机上按 F8 等了 3.4 秒）。
    """
    ticks = {"n": 0}
    ok = interruptible_sleep(0.3, lambda: False, slice_s=0.05, tick=lambda: ticks.__setitem__("n", ticks["n"] + 1))
    assert ok is True
    # 0.3 秒切 0.05 秒的片，至少要跑好几次；给个宽松下界，别把实现钉死。
    assert ticks["n"] >= 5


def test_靠tick设置的标志能打断睡眠():
    """热键处理挂在 `tick` 上时，睡眠必须当片就被打断，而不是等睡满。"""
    flag = {"v": False}

    def tick() -> None:
        flag["v"] = True

    t0 = time.monotonic()
    assert interruptible_sleep(3.0, lambda: flag["v"], slice_s=0.05, tick=tick) is False
    assert time.monotonic() - t0 < 1.0


def test_零秒睡眠也会跑一次tick():
    """`seconds <= 0` 的短路分支同样要让热键有机会被读到。"""
    ticks = {"n": 0}
    assert interruptible_sleep(0.0, lambda: False, tick=lambda: ticks.__setitem__("n", 1)) is True
    assert ticks["n"] == 1


# ============================================================ 会话脚手架


class _FakeEngine:
    """固定返回一个着法的假引擎。"""

    def __init__(self, bestmove: str = "h2e2", ok: bool = True) -> None:
        self.bestmove = bestmove if ok else ""
        self.calls: list[tuple[str, int]] = []

    def analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis:
        self.calls.append((fen, movetime_ms))
        if not self.bestmove:
            return Analysis()
        return Analysis(bestmove=self.bestmove, score_cp=10, depth=12, pv=[self.bestmove])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _Provider:
    """可变的局面来源：测试里手动控制"这一帧看到了什么"。"""

    def __init__(self, board: Board | None) -> None:
        self.board = board

    def __call__(self) -> Board | None:
        return None if self.board is None else self.board.copy()


class _Clock:
    """可拨快的单调时钟。

    走子校验是按**时间**给额度的（棋子滑过去的动画要时间，真机踩过 40ms 就判失败），
    所以测「超时失败」不能靠真等 2.5 秒 —— 把时钟拨过去。
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _make_session(
    state: State = State.MY_TURN,
    *,
    provider=None,
    engine=None,
    config: Config | None = None,
    geom=_FAKE_GEOM,
    my_side: int = RED,
    clock: _Clock | None = None,
    delay_tick=None,
    recorder: GameRecorder | None = None,
) -> Session:
    deps = SessionDeps(
        board_provider=provider if provider is not None else (lambda: None),
        engine=engine if engine is not None else _FakeEngine(),
        mouse=Mouse(dry_run=True, my_side=my_side),
        mapper=ScreenMapper(my_side),
        recorder=recorder,
        config=config if config is not None else DEFAULTS,
        geometry_provider=(lambda: geom),
        now=clock if clock is not None else time.monotonic,
        delay_tick=delay_tick,
    )
    s = Session(deps)
    s.board = Board.from_fen(START_FEN)
    # 回合归属要跟状态自洽：OPPONENT_TURN 时该黑方走，否则 interpret 会在红方的
    # 着法表里找黑方的着法，永远找不到。
    s.board.side = -my_side if state is State.OPPONENT_TURN else my_side
    s.state = state
    s._idle_since = deps.now()
    return s


def _run_until(s: Session, want: State, limit: int = 30) -> None:
    """推进状态机直到进入 `want`（或超过步数上限）。"""
    for _ in range(limit):
        if s.state is want:
            return
        if s.state in (State.PAUSED, State.GAME_OVER):
            break
        s.step()
    assert s.state is want, f"期望进入 {want}，实际停在 {s.state}：{s.message}"


# ============================================================ [RF-4] 终局判定


def test_轮到我方时局面不变_不进终局():
    s = _make_session(State.MY_TURN)
    s._idle_since = time.monotonic() - 1000
    s.step()
    assert s.state is not State.GAME_OVER
    assert s.state is State.PAUSED  # 改成暂停，等人来看
    assert "静止" in s.message or "异常" in s.message


def test_非我方回合静止超时_进终局():
    s = _make_session(State.OPPONENT_TURN)
    s._idle_since = time.monotonic() - 1000
    s.step()
    assert s.state is State.GAME_OVER


def test_刚开局的静止不算终局():
    s = _make_session(State.OPPONENT_TURN)  # _idle_since 是当前时刻
    s.step()
    assert s.state is State.OPPONENT_TURN


# ============================================================ [RF-2] 坏帧


def test_全白帧不改变已确认局面():
    s = _make_session()  # board_provider 恒返回 None
    before = s.board
    for _ in range(10):
        s.step()
    assert s.board is before


def test_残缺帧不改变已确认局面():
    """残缺帧与全白帧一样，在抓图层就被折算成 `None`，会话看到的没有区别。"""
    s = _make_session(State.OPPONENT_TURN)
    before = s.board
    for _ in range(10):
        s.step()
    assert s.board is before


def test_抓图失败时不会凭空记一笔对方着法():
    """抓图失败时 `interpret` 根本不该被调用 —— 没有输入就什么都没有。"""
    seen: list[str] = []

    class _Rec:
        def add(self, board_before, mv, analysis=None, think_ms=None):
            seen.append(mv)

    s = _make_session(State.OPPONENT_TURN, provider=lambda: None)
    s.deps.recorder = _Rec()
    for _ in range(10):
        s.step()
    assert seen == []
    assert s.state is State.OPPONENT_TURN


# ============================================================ 状态机主流程


def test_轮到我方会思考并点击():
    engine = _FakeEngine("h2e2")
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, engine=engine)

    s.step()
    assert s.state is State.THINKING
    s.step()  # 思考（引擎调用）→ PLAYING
    assert s.state is State.PLAYING
    assert s.last_analysis is not None and s.last_analysis.bestmove == "h2e2"
    assert s.deps.mouse.clicks == [], "还没到 PLAYING 的下一步，不该点"

    s.step()  # 点击
    assert s.state is State.VERIFY
    assert s.deps.mouse.clicks, "dry_run 也要留下点击记录"
    assert engine.calls and engine.calls[0][1] == DEFAULTS.movetime_ms


def test_走子后局面校验通过则轮到对方():
    start = Board.from_fen(START_FEN)
    expected = start.copy()
    make(expected, "h2e2")
    p = _Provider(start)
    s = _make_session(State.MY_TURN, provider=p)

    _run_until(s, State.VERIFY)
    p.board = expected  # 画面跟上了
    s.step()
    assert s.state is State.OPPONENT_TURN
    assert s.board is not None and s.board.grid.tolist() == expected.grid.tolist()


def test_棋子还在半路上时不算走成了():
    """落点还是空的（棋子正飞过去）只能说明「还没走成」，不能拿容忍度把它认成走成了。

    真机踩过：点完鼠标后抓到的帧正好在棋子滑行途中，落点是空的，
    `_grid_diff` 只差 1 格就被 `tolerance_cells = 1` 放行，于是把一次
    **还没生效**的点击记成走子成功，整盘棋从此错位。
    """
    start = Board.from_fen(START_FEN)
    expected = start.copy()
    make(expected, "h2e2")
    flying = expected.copy()
    flying.grid[2, 4] = 0  # 红炮已经离开 h2，但还没落到 e2

    p = _Provider(start)
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = False

    _run_until(s, State.VERIFY)
    p.board = flying
    s.step()
    assert s.state is State.VERIFY, "棋子还在半路上，不能算走成了"

    p.board = expected  # 落定了
    s.step()
    assert s.state is State.OPPONENT_TURN
    assert s.board is not None and s.board.grid.tolist() == expected.grid.tolist()


def test_走子后局面校验失败会暂停():
    start = Board.from_fen(START_FEN)
    other = start.copy()
    make(other, "b2e2")
    p = _Provider(start)
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = False  # 这条测的是「真点了但没生效」，不是彩排

    _run_until(s, State.VERIFY)
    p.board = other  # 画面停在别处，永远对不上
    s.step()
    assert s.state is State.VERIFY, "刚点完还在动画里，不能立刻判失败"
    clock.advance(VERIFY_TIMEOUT_S + 0.1)
    s.step()
    assert s.state is State.PAUSED
    assert "校验失败" in s.message


def test_彩排模式校验失败时给出彩排提示():
    """彩排本来就不点鼠标，局面不会变是意料之中，提示要说清楚而不是报故障。"""
    p = _Provider(Board.from_fen(START_FEN))
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = True

    _run_until(s, State.VERIFY)
    clock.advance(VERIFY_TIMEOUT_S + 0.1)
    s.step()
    assert s.state is State.PAUSED
    assert "彩排" in s.message
    assert "校验失败" not in s.message


def test_抓图失败时校验原地等待不改局面():
    """坏帧不是「局面没变」：抓不到画面就什么都别动。

    推的秒数要**大于** `VERIFY_TIMEOUT_S`（否则测不到「超时也不算数」）
    且**小于** `BOARD_LOST_TIMEOUT_S`（否则会走「棋盘没了 → 终局」那条路）。
    """
    p = _Provider(Board.from_fen(START_FEN))
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = False

    _run_until(s, State.VERIFY)
    p.board = None
    clock.advance(VERIFY_TIMEOUT_S + 1.0)
    s.step()
    assert s.state is State.VERIFY, "抓不到画面不能当成校验失败"
    assert s.board is not None and s.board.to_fen() == START_FEN


def test_棋盘一直读不到就判定对局结束():
    """真机现场：双車绝杀之后游戏切到结算画面，棋盘整个消失。

    `board_provider()` 从此一直给 `None`，各状态里的「什么都不改，继续等」会永远
    等下去。之前就是被 `VERIFY_TIMEOUT_S` 抢先把人引到「点击后局面校验失败」去了。
    """
    p = _Provider(Board.from_fen(START_FEN))
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = False

    _run_until(s, State.VERIFY)
    p.board = None

    clock.advance(BOARD_LOST_TIMEOUT_S - 0.1)
    s.step()
    assert s.state is State.VERIFY, "还没到额度就收摊太急了"

    # 失联计时是从**第一次读到 `None`** 那一刻起算的，所以这里要再走满一个额度。
    clock.advance(BOARD_LOST_TIMEOUT_S)
    s.step()
    assert s.state is State.GAME_OVER
    assert "看不到棋盘" in s.message


def test_棋盘失联后恢复则重新计时():
    """窗口被挡住几秒又露出来，不该被算成对局结束。"""
    p = _Provider(Board.from_fen(START_FEN))
    clock = _Clock()
    s = _make_session(State.OPPONENT_TURN, provider=p, clock=clock)

    p.board = None
    clock.advance(BOARD_LOST_TIMEOUT_S - 0.5)
    s.step()
    p.board = Board.from_fen(START_FEN)
    p.board.side = -RED
    s.step()
    assert s.state is State.OPPONENT_TURN, "读到棋盘就该把失联计时清零"

    p.board = None
    clock.advance(BOARD_LOST_TIMEOUT_S - 0.5)
    s.step()
    assert s.state is State.OPPONENT_TURN, "第二次失联要重新计时，不能接着上次算"


def test_对方还着已经落下来时两步一起收():
    """人机对战里对方想得很快：校验时画面可能已经是「我方走完 + 对方还着」。"""
    start = Board.from_fen(START_FEN)
    after_mine = start.copy()
    make(after_mine, "h2e2")
    after_both = after_mine.copy()
    after_both.side = -RED
    make(after_both, "h9g7")

    p = _Provider(start)
    clock = _Clock()
    s = _make_session(State.MY_TURN, provider=p, clock=clock)
    s.deps.mouse.dry_run = False

    _run_until(s, State.VERIFY)
    p.board = after_both
    s.step()
    assert s.state is State.MY_TURN, "对方已经还着，收完两步该轮到我方"
    assert s.board is not None and s.board.to_fen() == after_both.to_fen()


# ============================================================ 自动暂停后的恢复点


def test_几何缺失自动暂停后恢复重试同一步():
    """几何拿不到只是点不了，不是这步走坏了；F8 恢复后该重试点击而不是重新想。"""
    holder = {"geom": _FAKE_GEOM}
    s = _make_session(State.MY_TURN, provider=_Provider(Board.from_fen(START_FEN)))
    s.deps.geometry_provider = lambda: holder["geom"]

    _run_until(s, State.PLAYING)
    holder["geom"] = None
    s.step()
    assert s.state is State.PAUSED
    assert "几何" in s.message

    holder["geom"] = _FAKE_GEOM
    s.resume()
    assert s.state is State.PLAYING
    s.step()
    assert s.state is State.VERIFY  # 同一步真的点下去了
    assert s.deps.mouse.clicks  # 两次点击（起子 + 落子）


def _junk_board(drop: int) -> Board:
    """红方底线凭空少了 `drop + 1` 个子的局面 —— 没有任何黑方着法能走成这样。

    注意 `Board.grid` 的 `rank 0` 是**红方底线**：FEN 的 rank 0（黑方底线）落在 `grid[9]`。
    """
    board = Board.from_fen(START_FEN)
    board.side = -RED
    for col in range(drop + 1):
        board.grid[0, col] = 0
    return board


def _model_board() -> Board:
    """会话里那盘「模型认为正确」的局面（黑方走）。"""
    board = Board.from_fen(START_FEN)
    board.side = -RED
    return board


def test_同一个看不懂的局面看很多次也只算一次():
    """真机回归：主循环 0.1~0.2 秒一轮，`board_provider()` 在新局面停稳之前会一直返回
    **同一个**已停稳的盘子。照着调用次数数的话，一个瞬时坏帧（吃子动画把某一格闪没了）
    会被连数三次，`MAX_UNEXPLAINED = 3` 当场误触发暂停。"""
    s = _make_session(State.OPPONENT_TURN, provider=_Provider(_junk_board(1)))
    for _ in range(20):
        s.step()
    assert s.state is State.OPPONENT_TURN
    assert s.message == "看不懂局面变化（第 1 次）"


def test_画面回到预期后看不懂的计数清零():
    """坏一帧、好一帧、再坏一帧…… 攒不到三次。攒着算的话，一局棋里的零星抖动
    迟早会把对局误暂停。"""
    provider = _Provider(_junk_board(1))
    s = _make_session(State.OPPONENT_TURN, provider=provider)
    for _ in range(5):
        provider.board = _junk_board(2)
        s.step()
        provider.board = _model_board()
        s.step()
    assert s.state is State.OPPONENT_TURN
    assert not s.message.startswith("无法解释")


def test_对方回合看不懂变化后恢复仍停在对方回合():
    """自动暂停必须记住该回到哪一格；旧实现会退回默认的 MY_TURN。"""
    provider = _Provider(_junk_board(1))
    s = _make_session(State.OPPONENT_TURN, provider=provider)
    for i in range(3):
        provider.board = _junk_board(i + 1)
        s.step()
    assert s.state is State.PAUSED
    assert "看不懂" in s.message or "无法解释" in s.message
    s.resume()
    assert s.state is State.OPPONENT_TURN


def test_对方走子被识别后轮到我方():
    start = Board.from_fen(START_FEN)
    after = start.copy()
    make(after, "h9g7")
    s = _make_session(State.OPPONENT_TURN, provider=_Provider(after))
    s.step()
    assert s.state is State.MY_TURN
    assert s.board is not None and s.board.side == RED


def test_对方走子带一格误识别仍被纠正():
    start = Board.from_fen(START_FEN)
    seen = start.copy()
    make(seen, "h9g7")
    seen.grid[9, 8] = 0  # 右下角红车被识别成空
    s = _make_session(State.OPPONENT_TURN, provider=_Provider(seen))
    s.step()
    assert s.state is State.MY_TURN
    want = start.copy()
    make(want, "h9g7")
    assert s.board.grid.tolist() == want.grid.tolist()  # 采纳的是修正后的局面


def test_落点认不出来时宁可不解释也不能瞎猜():
    """真机踩过的坑：黑炮 d9 吃仕 d0（我方俥 d7 正好当炮架），抓到的帧恰好在吃子动画中途 ——
    d9 空了、d0 还没显出来。老实现拿容忍度 1 去配「炮 d9 进 d8」（d8 也空着，只差一格），
    于是把模型改成「d8 有颗黑炮」，从此和画面对不上，几步之后误报「点击后局面校验失败」。

    一颗子走完一定**在落点上**，所以落点对不上的候选着法一律不认。
    """
    prev = Board.from_fen("3ck4/9/3R5/9/9/4P4/9/9/9/3AK4 b - - 0 1")
    flying = prev.copy()
    flying.grid[9, 3] = 0  # 黑炮离开 d9，还没落到 d0
    assert interpret(prev, flying, 1) is None

    landed = prev.copy()
    make(landed, "d9d0")
    assert interpret(prev, landed, 1) == "d9d0"


def test_无法解释的变化三次后暂停():
    provider = _Provider(_junk_board(1))
    s = _make_session(State.OPPONENT_TURN, provider=provider)
    for i in range(3):
        provider.board = _junk_board(i + 1)
        s.step()
    assert s.state is State.PAUSED
    assert "无法解释" in s.message


def test_引擎没返回着法不会乱走():
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, engine=_FakeEngine(ok=False))
    for _ in range(10):
        s.step()
    assert s.deps.mouse.clicks == []
    assert s.state is State.PAUSED


def test_引擎给出非法着法不放行():
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, engine=_FakeEngine("a0a9"))
    for _ in range(5):
        s.step()
    assert s.deps.mouse.clicks == []
    assert s.state is State.PAUSED
    assert "不合法" in s.message


def test_拿不到几何时拒绝盲点():
    s = _make_session(State.MY_TURN, provider=_Provider(Board.from_fen(START_FEN)), geom=None)
    for _ in range(5):
        s.step()
    assert s.deps.mouse.clicks == []
    assert s.state is State.PAUSED


def test_开局局面进入对局():
    s = _make_session(State.IDLE, provider=_Provider(Board.from_fen(START_FEN)))
    s.step()
    assert s.state is State.MY_TURN
    assert s.board is not None


def test_执黑时开局先等对方走():
    s = _make_session(State.IDLE, provider=_Provider(Board.from_fen(START_FEN)), my_side=-1)
    s.step()
    assert s.state is State.OPPONENT_TURN


def test_大厅画面不进对局():
    """大厅里没有棋盘，识别层返回的基本是空 —— 棋子数不够就不该判定开局。"""
    empty = Board.from_fen(START_FEN)
    empty.grid[:] = 0
    empty.grid[0, 4] = 1
    empty.grid[9, 4] = -1
    s = _make_session(State.IDLE, provider=_Provider(empty))
    for _ in range(5):
        s.step()
    assert s.state is State.IDLE


# ============================================================ 暂停 / 停止


def test_暂停后step不做任何事():
    s = _make_session(State.MY_TURN)
    s.step()
    assert s.state is State.THINKING
    s.pause()
    assert s.state is State.PAUSED
    s.step()
    s.step()
    assert s.state is State.PAUSED
    assert s.deps.mouse.clicks == []


def test_恢复后回到暂停前的状态():
    s = _make_session(State.MY_TURN)
    s.step()  # -> THINKING
    s.pause()
    s.resume()
    assert s.state is State.THINKING


def test_请求停止后step不做任何事():
    s = _make_session(State.MY_TURN)
    s.request_stop()
    assert s.should_stop is True
    s.step()
    assert s.state is State.MY_TURN


def test_单步在暂停中也能推进一步():
    """F9 的语义：暂停着也能往前走一步，走完仍然停住。"""
    s = _make_session(State.MY_TURN)
    s.pause()
    assert s.state is State.PAUSED
    s.step_once()
    assert s.state is State.THINKING, "单步应该真的推进了一格"
    s.step()
    assert s.state is State.PAUSED, "单步之后不许继续跑"


def test_单步不会把暂停解掉():
    s = _make_session(State.MY_TURN)
    s.pause()
    s.step_once()
    assert s._pause_requested is True
    s.step()
    assert s.state is State.PAUSED


def test_未暂停时单步就是普通一步():
    s = _make_session(State.MY_TURN)
    s.step_once()
    assert s.state is State.THINKING


# ============================================================ 拟人化延迟（S7）


def test_人类延迟期间被中断就不走子():
    cfg = Config(**{**DEFAULTS.__dict__, "human_delay": (3.0, 3.0)})
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, config=cfg)
    s.step()  # -> THINKING
    threading.Timer(0.2, s.request_stop).start()
    t0 = time.monotonic()
    s.step()  # 思考完进入延迟，被中断
    assert time.monotonic() - t0 < 1.5, "延迟必须切成小片，不能一次睡满 3 秒"
    assert s.deps.mouse.clicks == []
    assert s.state is State.MY_TURN


def test_人类延迟睡满才走子():
    cfg = Config(**{**DEFAULTS.__dict__, "human_delay": (0.2, 0.2)})
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, config=cfg)
    _run_until(s, State.VERIFY)
    assert s.deps.mouse.clicks


def test_拟人延迟期间delay_tick会被反复调用并能借它暂停():
    """主循环就是靠 `delay_tick` 在延迟里读热键的。

    真机实测：`--human-delay 3-5` 那局在延迟中间按 F8，延迟照样睡满 4.7 秒，
    3.4 秒之后才响应 —— 因为主循环阻塞在 `session.step()` 里，没人去读键盘。
    """
    cfg = Config(**{**DEFAULTS.__dict__, "human_delay": (3.0, 3.0)})
    p = _Provider(Board.from_fen(START_FEN))
    ticks = {"n": 0}

    def tick() -> None:
        ticks["n"] += 1
        if ticks["n"] >= 3:
            s.pause()  # 模拟「主循环在这一片里读到了 F8」

    s = _make_session(State.MY_TURN, provider=p, config=cfg, delay_tick=tick)
    s.step()  # -> THINKING
    t0 = time.monotonic()
    s.step()  # 思考完进入延迟，第 3 片就被上面那个 tick 打断
    assert time.monotonic() - t0 < 1.5, "delay_tick 得真的被调用，不能等睡满 3 秒"
    assert ticks["n"] >= 3
    assert s.state is State.MY_TURN
    assert s.deps.mouse.clicks == []


def test_延迟消息带秒数():
    cfg = Config(**{**DEFAULTS.__dict__, "human_delay": (0.1, 0.1)})
    p = _Provider(Board.from_fen(START_FEN))
    s = _make_session(State.MY_TURN, provider=p, config=cfg)
    s.step()  # -> THINKING

    seen: list[str] = []
    original = s.deps.mouse.play_move

    def _spy(mv, geom):
        seen.append(s.message)
        return original(mv, geom)

    s.deps.mouse.play_move = _spy  # type: ignore[method-assign]
    _run_until(s, State.VERIFY)
    assert seen and "human delay" in seen[0]


# ============================================================ 不该崩的输入


@pytest.mark.parametrize("tolerance", [0, 1, 2])
def test_容差取值都能工作(tolerance: int):
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    got = interpret(prev, cur, tolerance=tolerance)
    assert got is not None and got in legal_moves(prev)


@pytest.mark.parametrize("tolerance", [0, 1])
def test_小容差时唯一解就是真实着法(tolerance: int):
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    assert interpret(prev, cur, tolerance=tolerance) == "h2e2"


def test_容差放大后可能有多个解():
    """`tolerance=2` 时「炮二平五」与「炮八平五」都只差两格，取先生成的那一步。

    这是容差换纠错能力的必然代价：容差越大越能吸收误识别，也越容易选错同型着法。
    `tolerance_cells` 默认 1 就是这个折中。
    """
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    make(cur, "h2e2")
    got = interpret(prev, cur, tolerance=2)
    assert got in {"h2e2", "b2e2"}


def test_解释器不依赖网格可变性():
    """`Board.grid` 是 numpy 数组，比较必须按元素比 —— 用 `is` 会永远不等。"""
    a = Board.from_fen(START_FEN)
    b = a.copy()
    assert a.grid.tolist() == b.grid.tolist()
    assert not np.shares_memory(a.grid, b.grid)


# ============================================================ 代码评审修复
#
# 下面这几条都是整分支评审（三个新鲜上下文评审员）报出来的真缺陷，每条先红后绿。


def test_轮到我方时画面已落下我方着法要按我方采纳():
    """`interpret` 在 MY_TURN 只可能解释出**我方**的着法，不能当成对方走的。

    MY_TURN 的不变式是 `self.board.side == my_side`，而 `interpret` 只在
    `legal_moves(prev)` 里搜 —— 所以「轮到我方时局面却变了」的真实含义只有一个：
    上一次点击其实打出去了，只是画面确认超时把它当成了没走成。

    早先这里按「对方真的走了」处理（`opponent=True`，状态留在 MY_TURN），后果是
    引擎拿着**轮到黑方**的 FEN 替对方算棋、鼠标替对方走子、记谱里多出一手根本不是
    我方决策的着法。真机上的表现是「暂停恢复之后机器人走出一手莫名其妙的棋」。
    """
    provider = _Provider(_model_board())
    rec = _recorder()
    s = _make_session(State.MY_TURN, provider=provider, recorder=rec)
    landed = _model_board()
    make(landed, "h2e2")  # 我方这一手其实已经落下去了
    provider.board = landed

    s.step()

    assert s.state is State.OPPONENT_TURN, "我方着法落下之后该轮到对方"
    assert s.board.side == -RED, "局面要推进到「我方走完」，轮到黑方"
    assert s.message == "走 h2e2"
    assert [m.iccs for m in rec.moves] == ["h2e2"]


def test_恢复之后不会凭旧账立刻又暂停():
    """`resume()` 必须清掉「看不懂局面」的记账，否则 F8 是个死键。

    自动暂停的语义是「人看一眼、修好了再按 F8 继续」，但 `_unexplained` 是按
    「看不懂的局面」累加的：暂停期间画面没变，恢复后第一帧还是那个坏局面，
    同一个 FEN 不重复计数也就永远清不掉 —— 每一次 `step()` 都立刻回到暂停，
    消息一模一样，只有 ESC 能出去。
    """
    provider = _Provider(_model_board())
    s = _make_session(State.OPPONENT_TURN, provider=provider)
    for drop in (1, 2, 3):
        provider.board = _junk_board(drop)
        s.step()
    assert s.state is State.PAUSED
    assert s._unexplained == MAX_UNEXPLAINED

    s.resume()

    assert s._unexplained == 0, "恢复时要把看不懂的记账清零"
    assert s._last_unexplained is None
    s.step()
    assert s.state is not State.PAUSED, "恢复后又凭旧账立刻暂停了，F8 等于死键"
    assert s._unexplained == 1
    assert "第 1 次" in s.message


def test_恢复之后不会因为暂停期间的时间被判终局():
    """`resume()` 也要刷新静止计时：暂停期间没人走棋，旧时间戳会让恢复当场收摊。"""
    clock = _Clock()
    s = _make_session(
        State.OPPONENT_TURN, provider=_Provider(_model_board()), clock=clock
    )
    clock.advance(1.0)
    s.pause()
    clock.advance(60.0)  # 人看了很久才按 F8
    s.resume()

    assert s._idle_since == clock.t, "静止计时要从恢复那一刻重新起算"
    s.step()
    assert s.state is not State.GAME_OVER


def test_轮到我方时看不懂局面照样算账():
    """MY_TURN 的「画面变了但解释不出来」要跟对方回合一样计数。

    早先这里既不计数也不暂停（与 `_step_opponent_turn` 三振机制不对称）：
    模型已经追不上画面了还照旧点击，下一步点击就打在错误的格子上。
    """
    provider = _Provider(_model_board())
    s = _make_session(State.MY_TURN, provider=provider)
    for drop in (1, 2, 3):
        provider.board = _junk_board(drop)
        s.step()
    assert s.state is State.PAUSED
    assert s.message == "看不懂局面变化，已暂停（请检查识别）"


def test_思考时异常静止要真的停下来():
    """`_step_thinking` 不能丢掉 `_check_idle()` 的返回值。

    随后那句 `self.state = State.PLAYING` 会把刚设好的 PAUSED 盖回去 ——
    面板和日志上写着「轮到我方但局面异常静止，请检查」，实际却在继续点击。
    """
    clock = _Clock()
    s = _make_session(
        State.THINKING, provider=_Provider(_model_board()), clock=clock
    )
    s._idle_since = clock.t - 1000

    s.step()

    assert s.state is State.PAUSED
    assert s._pending_move is None, "没停下来就会带着这一手去点击"
    assert "静止" in s.message or "异常" in s.message


def test_校验失败重试不会把同一手记两遍():
    """记谱点放在**确认走成之后**。

    点击可能根本没生效（校验超时 → 回到 MY_TURN 重新决策）。提前在点击那一瞬写记谱，
    就会把一步没走成的着法留在记谱里；重试成功后再写一次，同一回合就出现两手，
    ICCS 行也成了非法着法序列。`GameRecorder` 没有回滚接口，所以只能等确认。
    """
    clock = _Clock()
    provider = _Provider(_model_board())
    rec = _recorder()
    s = _make_session(State.MY_TURN, provider=provider, clock=clock, recorder=rec)

    _run_until(s, State.VERIFY)
    assert rec.moves == [], "还没确认走成就不该记谱"

    clock.advance(VERIFY_TIMEOUT_S + 0.1)  # 画面一直没变 → 超时
    s.step()
    assert s.state is State.PAUSED
    assert rec.moves == [], "没走成的一手绝不能进记谱"

    s.resume()
    _run_until(s, State.VERIFY)
    landed = s._board_before.copy()
    make(landed, s._pending_move)
    provider.board = landed
    s.step()

    assert s.state is State.OPPONENT_TURN
    assert [m.iccs for m in rec.moves] == ["h2e2"], "重试成功后同一手只该记一次"
