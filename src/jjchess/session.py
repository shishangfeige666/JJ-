"""对局状态机 —— 整套脚本的鲁棒性核心。

一句话概括这个模块要扛住的事：**画面识别一定会出错，但不能让错误变成非法着法。**

两道防线
--------
1. **合法着法解释器** `interpret(prev, cur)`：不直接相信识别出的局面，而是反过来问
   「上一局面走过哪一步会变成现在这样？」—— 只在 `legal_moves(prev)` 里搜，
   允许至多 `tolerance` 个格子对不上（识别偶尔把某个子看成空格）。
   于是「识别出的非法局面」永远进不了引擎，皮卡鱼的严格校验也就永远不会被触发。
2. **坏帧闸门**：抓图失败时 `board_provider()` 返回 `None`，会话**原样保留**自己的局面认知。
   全白帧与残缺帧（`window.frame_usable`）都在抓图层被折算成 `None`，
   否则「32 子全空」会被当成对方一口气吃光了所有子。

状态迁移
--------
    IDLE ──开局局面──> MY_TURN / OPPONENT_TURN
    MY_TURN ──> THINKING ──> PLAYING ──> VERIFY ──> OPPONENT_TURN
                    │                          └──校验失败──> PAUSED
                    └──引擎连续失败──────────> PAUSED
    OPPONENT_TURN ──解释出对方着法──> MY_TURN
                  ├──连续 3 次解释不出──> PAUSED
                  └──静止超时──> GAME_OVER          （[RF-4] 对方长考/已离场）
    MY_TURN 静止超时 ──> PAUSED                        （[RF-4] 轮到我方却不动 = 异常）

`step()` 是**非阻塞**的一次推进（唯一例外是 `--human-delay` 的拟人停顿，
它被切成 ≤50 ms 的小片，所以 F8 / ESC 仍然即时生效）。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from jjchess.config import Config
from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.rules.moves import legal_moves, make

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注
    from jjchess.control import Mouse
    from jjchess.engine.uci import Analysis
    from jjchess.record import GameRecorder
    from jjchess.rules.board import Board

__all__ = [
    "MAX_UNEXPLAINED",
    "MAX_VERIFY_TRIES",
    "Session",
    "SessionDeps",
    "State",
    "interpret",
    "interruptible_sleep",
]

#: `interpret` 连续失败多少次就认输暂停。
MAX_UNEXPLAINED = 3
#: 走子后局面校验失败后还允许重试几次。
MAX_VERIFY_TRIES = 1
#: 开局判定的棋子数下限（满盘 32 子，少几个也算，但大厅的空盘一定不够）。
_MIN_OPENING_PIECES = 28


class State(str, Enum):
    """会话状态。`str` 混入是为了面板/日志能直接打印。"""

    IDLE = "IDLE"
    MY_TURN = "MY_TURN"
    THINKING = "THINKING"
    PLAYING = "PLAYING"
    VERIFY = "VERIFY"
    OPPONENT_TURN = "OPPONENT_TURN"
    PAUSED = "PAUSED"
    GAME_OVER = "GAME_OVER"


def interruptible_sleep(seconds: float, stop: Callable[[], bool], slice_s: float = 0.05) -> bool:
    """睡 `seconds` 秒，但随时响应 `stop()`。

    返回 `True` 表示睡满了，`False` 表示被 `stop()` 打断。

    ⚠️ **不能写成一次 `time.sleep(seconds)`** —— `--human-delay` 最长 2 秒，
    那样按 F8 / ESC 就要等两秒才有反应（spec 的成功标准 S6 / S7）。
    """
    if seconds <= 0:
        return not stop()
    deadline = time.monotonic() + seconds
    while True:
        if stop():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(slice_s, remaining))


def _grid_diff(a: Board, b: Board) -> int:
    """两个局面有多少格不一样。"""
    return int(np.count_nonzero(a.grid != b.grid))


def interpret(prev: Board, cur: Board, tolerance: int = 1) -> str | None:
    """从 `prev` 到 `cur` 的那一步是什么；解释不出来返回 `None`。

    **纯函数，不改动入参。** 只在 `legal_moves(prev)` 里搜，所以返回值一定是合法着法 ——
    这正是 spec 说的"拒绝把非法局面喂给皮卡鱼"。允许 `tolerance` 个格子对不上，
    用来吸收「某个子被识别成空格」这类零星错误。
    """
    if _grid_diff(prev, cur) == 0:
        return None
    for mv in legal_moves(prev):
        trial = prev.copy()
        make(trial, mv)
        if _grid_diff(trial, cur) <= tolerance:
            return mv
    return None


@dataclass
class SessionDeps:
    """会话需要的外部能力，全部以可替换对象的形式注入（便于测试）。"""

    board_provider: Callable[[], Board | None]
    engine: object
    mouse: Mouse
    mapper: ScreenMapper
    recorder: GameRecorder | None
    config: Config
    #: 棋盘几何也来自「当前这一帧」，所以同样用 provider 而不是固定值。
    geometry_provider: Callable[[], BoardGeometry | None] | None = None


@dataclass
class Session:
    """对局状态机。每次 `step()` 推进一步。"""

    deps: SessionDeps
    state: State = State.IDLE
    board: Board | None = None
    last_analysis: Analysis | None = None
    message: str = ""
    should_stop: bool = False

    _pause_requested: bool = False
    _state_before_pause: State = State.MY_TURN
    _idle_since: float = field(default_factory=time.monotonic)
    _unexplained: int = 0
    _verify_tries: int = 0
    _pending_move: str | None = None
    _board_before: Board | None = None
    _think_ms: int | None = None

    # ---------------------------------------------------------------- 属性

    @property
    def my_side(self) -> int:
        return self.deps.mapper.my_side

    @property
    def my_turn(self) -> bool:
        return self.state in (State.MY_TURN, State.THINKING, State.PLAYING, State.VERIFY)

    # ---------------------------------------------------------------- 控制

    def pause(self) -> None:
        """请求暂停。下一次 `step()` 起生效，且**不丢**当前局面认知。"""
        if self.state is not State.PAUSED:
            self._state_before_pause = self.state
        self._pause_requested = True
        self.state = State.PAUSED
        self.message = "已暂停（按 F8 继续）"

    def resume(self) -> None:
        """从暂停中恢复，回到暂停前的状态。"""
        if self.state is not State.PAUSED:
            return
        self._pause_requested = False
        back = self._state_before_pause
        self.state = State.MY_TURN if back is State.PAUSED else back
        self.message = "继续"

    def request_stop(self) -> None:
        """请求整个主循环退出（ESC）。"""
        self.should_stop = True

    # ---------------------------------------------------------------- 主循环

    def step(self) -> None:
        """推进一步。非阻塞（拟人延迟除外，它切成了小片）。"""
        if self.should_stop:
            return
        if self._pause_requested:
            self.state = State.PAUSED
            self.message = "已暂停（按 F8 继续）"
            return

        if self.state is State.IDLE:
            self._step_idle()
        elif self.state is State.MY_TURN:
            self._step_my_turn()
        elif self.state is State.THINKING:
            self._step_thinking()
        elif self.state is State.PLAYING:
            self._step_playing()
        elif self.state is State.VERIFY:
            self._step_verify()
        elif self.state is State.OPPONENT_TURN:
            self._step_opponent_turn()
        # PAUSED / GAME_OVER：什么都不做，等外部指令。

    # ---------------------------------------------------------------- 各状态

    def _step_idle(self) -> None:
        cur = self.deps.board_provider()
        if cur is None:
            return
        if not _looks_like_opening(cur):
            self.message = "等待对局开始…"
            return
        board = cur.copy()
        board.side = RED  # 象棋红先
        self.board = board
        self._idle_since = time.monotonic()
        self._unexplained = 0
        self._verify_tries = 0
        self.message = "检测到开局局面"
        self.state = State.MY_TURN if self.my_side == RED else State.OPPONENT_TURN

    def _step_my_turn(self) -> None:
        cur = self.deps.board_provider()
        if cur is not None and self.board is not None and _grid_diff(cur, self.board) != 0:
            # 轮到我方时局面却变了：可能是识别抖动，也可能对方真的走了。
            mv = interpret(self.board, cur, self.deps.config.tolerance_cells)
            if mv is not None:
                self._adopt(mv, opponent=True)
        if self._check_idle():
            return
        self.state = State.THINKING

    def _step_thinking(self) -> None:
        self._check_idle()
        board = self.board
        if board is None:  # pragma: no cover - 不该发生
            self.state = State.IDLE
            return
        started = time.monotonic()
        analysis = self.deps.engine.analyse(  # type: ignore[attr-defined]
            board.to_fen(), self.deps.config.movetime_ms
        )
        self._think_ms = max(1, int((time.monotonic() - started) * 1000))
        self.last_analysis = analysis

        if not analysis.ok:
            self._unexplained += 1
            self.message = f"引擎没有返回着法（第 {self._unexplained} 次）"
            if self._unexplained >= MAX_UNEXPLAINED:
                self.state = State.PAUSED
                self.message = "引擎连续多次没有返回着法，已暂停"
            else:
                self.state = State.MY_TURN
            return

        mv = analysis.bestmove
        if mv not in set(legal_moves(board)):
            self.state = State.PAUSED
            self.message = f"引擎给出的着法 {mv} 在当前局面下不合法，已暂停"
            return
        self._unexplained = 0

        delays = self.deps.config.human_delay
        if delays:
            seconds = random.uniform(*delays)
            self.message = f"THINKING (human delay {seconds:.1f}s)"
            if not interruptible_sleep(seconds, lambda: self._pause_requested or self.should_stop):
                self.state = State.MY_TURN  # 被打断就不走子，重新决策
                return

        self._pending_move = mv
        self._board_before = board.copy()
        self.state = State.PLAYING

    def _step_playing(self) -> None:
        mv = self._pending_move
        before = self._board_before
        if mv is None or before is None:  # pragma: no cover - 不该发生
            self.state = State.MY_TURN
            return
        geom = self.deps.geometry_provider() if self.deps.geometry_provider else None
        if geom is None:
            self.state = State.PAUSED
            self.message = "拿不到棋盘几何，无法点击，已暂停"
            return

        self.deps.mouse.play_move(mv, geom)
        if self.deps.recorder is not None:
            self.deps.recorder.add(before, mv, self.last_analysis, self._think_ms)
        self._verify_tries = 0
        self.message = f"已走 {mv}，等待画面确认"
        self.state = State.VERIFY

    def _step_verify(self) -> None:
        mv = self._pending_move
        before = self._board_before
        if mv is None or before is None:  # pragma: no cover - 不该发生
            self.state = State.MY_TURN
            return
        expected = before.copy()
        make(expected, mv)

        cur = self.deps.board_provider()
        if cur is not None and _grid_diff(cur, expected) == 0:
            self.board = expected
            self._idle_since = time.monotonic()
            self._pending_move = None
            self._board_before = None
            self._verify_tries = 0
            self.message = "走子已确认"
            self.state = State.OPPONENT_TURN
            return

        self._verify_tries += 1
        if self._verify_tries > MAX_VERIFY_TRIES:
            self.state = State.PAUSED
            self.message = "点击后局面校验失败，已暂停"

    def _step_opponent_turn(self) -> None:
        # 静止计时先判：抓图一直失败（窗口最小化/被关掉）本身也是一种"对方没动"，
        # 拖过 `gameover_idle_sec` 就该收摊，而不是无限等下去。
        if self._check_idle():
            return
        cur = self.deps.board_provider()
        if cur is None:
            # [RF-2] 抓图失败：**什么都不改**。绝不能把「没抓到」当成「棋盘空了」。
            return
        if self.board is not None and _grid_diff(cur, self.board) == 0:
            return

        mv = interpret(self.board, cur, self.deps.config.tolerance_cells)  # type: ignore[arg-type]
        if mv is None:
            self._unexplained += 1
            self.message = f"看不懂局面变化（第 {self._unexplained} 次）"
            if self._unexplained >= MAX_UNEXPLAINED:
                self.state = State.PAUSED
                self.message = "无法解释局面变化，已暂停（请检查识别）"
            return

        self._adopt(mv, opponent=True)

    # ---------------------------------------------------------------- 工具

    def _adopt(self, mv: str, *, opponent: bool) -> None:
        """采纳一步（对方走的），把局面推到**修正后**的样子并轮到我方。"""
        before = self.board
        if before is None:  # pragma: no cover - 不该发生
            self.state = State.IDLE
            return
        delta = before.copy()
        make(delta, mv)
        if self.deps.recorder is not None:
            self.deps.recorder.add(before, mv, None, None)
        self.board = delta
        self._idle_since = time.monotonic()
        self._unexplained = 0
        self.message = f"对方走 {mv}" if opponent else f"走 {mv}"
        self.state = State.MY_TURN

    def _check_idle(self) -> bool:
        """局面静止过久。返回 `True` 表示已经接管了状态，调用方必须返回。"""
        limit = self.deps.config.gameover_idle_sec
        if time.monotonic() - self._idle_since <= limit:
            return False
        if self.state is State.OPPONENT_TURN:
            # [RF-4] 对方长考、掉线或已经走人 —— 这一局结束了。
            self.state = State.GAME_OVER
            self.message = "局面长时间不变，判定对局结束"
            return True
        # [RF-4] 轮到我方却长时间不动 = 有东西不对劲，停下来等人看，别草率收摊。
        self.state = State.PAUSED
        self.message = "轮到我方但局面异常静止，请检查（按 F8 继续）"
        return True


def _looks_like_opening(board: Board) -> bool:
    """棋子够多、双方将帅都在 —— 这才像一局棋，而不是大厅菜单。"""
    values = board.grid
    if int(np.count_nonzero(values)) < _MIN_OPENING_PIECES:
        return False
    red_king = 1  # PIECE_CHARS 里的 K
    black_king = -1
    return bool((values == red_king).any() and (values == black_king).any())
