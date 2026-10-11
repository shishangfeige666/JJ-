"""`jjchess.session` 的单元测试：合法着法解释器、可中断睡眠、状态机鲁棒性。

[RF-2] 抓图失败（全白帧 / 残缺帧）时 `board_provider()` 返回 `None`。
       会话**绝不能**因此改动 `self.board` —— 那会被当成「棋盘上子没了」。
       注意：两种坏帧都在抓图层就被 `window.frame_usable` 挡下并折算成 `None`，
       所以这里一条用例同时覆盖它们。
[RF-4] 局面长时间不变：**轮到我方**时必须判 `PAUSED`（等人来看），
       只有**非我方回合**（对方长考/已离场）才能判 `GAME_OVER`。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from jjchess.config import DEFAULTS, Config
from jjchess.control import Mouse
from jjchess.engine.uci import Analysis
from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.rules.board import START_FEN, Board
from jjchess.rules.moves import legal_moves, make
from jjchess.session import (
    BOARD_LOST_TIMEOUT_S,
    VERIFY_TIMEOUT_S,
    Session,
    SessionDeps,
    State,
    interpret,
    interruptible_sleep,
)

_FAKE_GEOM = BoardGeometry(origin=(100.0, 100.0), dx=80.0, dy=80.0, cell_px=64.0)


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
) -> Session:
    deps = SessionDeps(
        board_provider=provider if provider is not None else (lambda: None),
        engine=engine if engine is not None else _FakeEngine(),
        mouse=Mouse(dry_run=True, my_side=my_side),
        mapper=ScreenMapper(my_side),
        recorder=None,
        config=config if config is not None else DEFAULTS,
        geometry_provider=(lambda: geom),
        now=clock if clock is not None else time.monotonic,
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


def test_对方回合看不懂变化后恢复仍停在对方回合():
    """自动暂停必须记住该回到哪一格；旧实现会退回默认的 MY_TURN。"""
    weird = Board.from_fen(START_FEN)
    weird.set(6, 0, 0)  # 凭空少两个红兵：没有任何黑方着法能走成这样
    weird.set(6, 2, 0)
    weird.side = -RED
    s = _make_session(State.OPPONENT_TURN, provider=_Provider(weird))
    for _ in range(5):
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


def test_无法解释的变化三次后暂停():
    junk = Board.from_fen(START_FEN)
    junk.grid[0, 0] = 0
    junk.grid[0, 1] = 0
    junk.grid[0, 2] = 0
    junk.grid[0, 3] = 0
    s = _make_session(State.OPPONENT_TURN, provider=_Provider(junk))
    for _ in range(3):
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
