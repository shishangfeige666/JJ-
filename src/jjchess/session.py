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
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from jjchess.config import Config
from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.rules.moves import legal_moves, make, parse_move

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注
    from jjchess.control import Mouse
    from jjchess.engine.uci import Analysis
    from jjchess.record import GameRecorder
    from jjchess.rules.board import Board

__all__ = [
    "BOARD_LOST_TIMEOUT_S",
    "MAX_UNEXPLAINED",
    "VERIFY_TIMEOUT_S",
    "Session",
    "SessionDeps",
    "State",
    "interpret",
    "interruptible_sleep",
]

#: `interpret` 连续失败多少次就认输暂停。
MAX_UNEXPLAINED = 3
#: 点完鼠标之后，画面最多容许「还没变过来」多久（秒）。
#:
#: 原来是「允许重试 1 次」，但主循环 20ms 一轮，两次校验之间只隔 40ms ——
#: 游戏里棋子是**滑过去**的，40ms 连动画都没放完。真机实测就这样误判成
#: 「点击后局面校验失败」而暂停。改成按时间给额度，动画多慢都不会误伤。
#:
#: 5 秒是量出来的：吃子那一下游戏会放一段「棋子闪烁 + 被吃子消失」的动画，
#: 逐帧抓到的局面在两三个变体之间来回跳，实测要 2~3 秒才停稳，
#: 再留 0.3 秒给画面稳定闸门，2.5 秒正好卡在边缘上（真机踩过）。
VERIFY_TIMEOUT_S = 5.0
#: 棋盘「读不到」多久就判定对局结束（秒）。
#:
#: 真机实测：我方用双車做成绝杀之后，游戏立刻切到结算画面（大红的「再来一局」），
#: 棋盘整个消失 —— 定位层抛 `BoardNotFound`，`board_provider()` 一直给 `None`。
#: 这时候不该报「点击后局面校验失败」把人引偏，而是老老实实收摊。
#: 比 `gameover_idle_sec` 短，是因为「棋盘没了」比「棋盘没动」更确定。
BOARD_LOST_TIMEOUT_S = 6.0
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


def interruptible_sleep(
    seconds: float,
    stop: Callable[[], bool],
    slice_s: float = 0.05,
    tick: Callable[[], None] | None = None,
) -> bool:
    """睡 `seconds` 秒，但随时响应 `stop()`。

    返回 `True` 表示睡满了，`False` 表示被 `stop()` 打断。

    `tick` 在每一小片里先跑一次 —— **这是让 F8 / ESC 在拟人延迟期间真正「即时」的关键**。
    `stop()` 读的是 `_pause_requested` / `should_stop` 两个标志，而设置它们的是主循环里的
    热键处理（`_handle_hotkeys`），主循环此刻正阻塞在这个函数里面 —— 没人去读键盘，
    `stop()` 就永远看不到新按键，只能睡满才轮得到热键。真机上按 F8 等了 3.4 秒才暂停
    （`--human-delay 3-5`，延迟 4.7 秒），就是这么来的。把热键处理挂到 `tick` 上才对得上
    spec 的成功标准 S7：「延迟期间 F8/ESC 仍即时响应」。

    ⚠️ **不能写成一次 `time.sleep(seconds)`** —— 那样连 `stop()` 都没机会看。
    """
    if seconds <= 0:
        if tick is not None:
            tick()
        return not stop()
    deadline = time.monotonic() + seconds
    while True:
        if tick is not None:
            tick()
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

    **但落点必须对得上。** 一颗子走完就一定在落点上，落点是空的只可能说明「这一步
    还没走成」（抓早了）或者「真着法其实是吃子，还差一格没显出来」。少了这条规矩，
    容忍度就成了模型漂移的通道 —— 真机上黑炮 d9 吃仕 d0，抓到的帧正好在吃子动画
    中途（d9 空了、d0 还没显出来），`tolerance = 1` 会让「炮 d9 进 d8」以「d8 也空着、
    只差一格」为由匹配上，模型从此凭空多一颗 d8 的黑炮，几步之后误报
    「点击后局面校验失败」。
    """
    if _grid_diff(prev, cur) == 0:
        return None
    for mv in legal_moves(prev):
        trial = prev.copy()
        make(trial, mv)
        dst_rank, dst_file = parse_move(mv)[1]
        if trial.grid[dst_rank, dst_file] != cur.grid[dst_rank, dst_file]:
            continue
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
    #: 单调时钟。走子校验要按时间给额度，测试得能把它拨快。
    now: Callable[[], float] = time.monotonic
    #: 拟人延迟期间每一小片调一次，用来读热键（见 `interruptible_sleep` 的 `tick`）。
    #: 不给也不会崩 —— 只是延迟期间按 F8/ESC 要等延迟走完才生效。
    delay_tick: Callable[[], None] | None = None


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
    _idle_since: float = 0.0
    _unexplained: int = 0
    #: 上一个「记过账」的看不懂局面。同一个局面只算一次 —— 见 `_step_opponent_turn`。
    _last_unexplained: str | None = None
    _play_at: float = 0.0
    _board_lost_since: float | None = None
    _pending_move: str | None = None
    _board_before: Board | None = None
    _think_ms: int | None = None

    def __post_init__(self) -> None:
        # `_idle_since` 的初值要用**注入的**时钟，不能是模块导入那一刻的真实时间。
        self._idle_since = self.deps.now()

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
        # 恢复时必须把两本账清零，否则 F8 会是个死键。
        #
        # 「请检查识别」这类自动暂停的语义就是「人看一眼、修好了再按 F8 继续」，
        # 但 `_unexplained` 是按「看不懂的局面」累加的：暂停期间画面没变，
        # 恢复后第一帧还是那个坏局面，同一个 FEN 不重复计数（见 `_count_unexplained`）
        # 也就永远清不掉 —— 每一次 `step()` 都立刻回到暂停，消息一模一样，
        # 只有 ESC 能出去。静止计时同理：暂停期间没人走棋，`_idle_since` 停在暂停前
        # 那一刻，恢复后当场被判「局面长时间不变」而收摊。
        self._idle_since = self.deps.now()
        self._unexplained = 0
        self._last_unexplained = None
        self.message = "继续"

    def request_stop(self) -> None:
        """请求整个主循环退出（ESC）。"""
        self.should_stop = True

    def _auto_pause(self, resume_state: State, message: str) -> None:
        """自己发现不对劲而停下来（不是用户按 F8）。

        必须同时记下「按 F8 之后该回到哪一格」：早先只有 `pause()` 会写
        `_state_before_pause`，自动暂停会让它停在上一次显式暂停时的旧值，
        按 F8 就恢复到跟当前情况无关的状态（比如对方回合的故障恢复成我方回合）。
        """
        self.state = State.PAUSED
        self._state_before_pause = resume_state
        self.message = message

    def step_once(self) -> None:
        """单步推进（F9）：暂停中也能往前走一步，走完仍然停住。

        注意是「走完停住」而不是「走完继续跑」—— F9 是给暂停态用的放大镜，
        松开 F9 不该顺带把暂停解掉。
        """
        if self.state is State.PAUSED:
            self._pause_requested = False
            self.state = self._state_before_pause
        self.step()
        if self.state is not State.GAME_OVER:
            self._pause_requested = True

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
        self._idle_since = self.deps.now()
        self._unexplained = 0
        self._last_unexplained = None
        self.message = "检测到开局局面"
        self.state = State.MY_TURN if self.my_side == RED else State.OPPONENT_TURN

    def _step_my_turn(self) -> None:
        cur = self.deps.board_provider()
        if self._note_board(cur):
            return
        if cur is not None and self.board is not None and _grid_diff(cur, self.board) != 0:
            mv = interpret(self.board, cur, self.deps.config.tolerance_cells)
            if mv is not None:
                # MY_TURN 的不变式是 `self.board.side == my_side`（`_step_idle`、
                # `_adopt`、`_step_verify` 三处共同维持），而 `interpret` 只在
                # `legal_moves(prev)` 里搜 —— 所以这里**只可能**解释出我方的着法。
                #
                # 「轮到我方时局面却变了」的真实含义只有一个：上一次点击其实打出去了，
                # 只是画面确认超时把它当成没走成（`_step_verify` 的第 3 条出路），
                # 或者点击成功但紧接着就被暂停了。这一步确实落下了。
                #
                # 早先这里按「对方真的走了」处理（`opponent=True`），后果是：引擎拿着
                # 轮到黑方的 FEN 替**对方**算棋、鼠标替**对方**走子、记谱里多出一手根本
                # 不是我方决策的着法。要在构造上可能，除非 `interpret` 能返回对方的着法。
                #
                # 补收时也要记谱：这一步从没走到过 `_step_verify` 的确认分支，
                # 不记就漏了一手。
                self._record_move(self.board, mv)
                self._adopt(mv, opponent=False)
                return
            self._count_unexplained(
                cur, State.MY_TURN, "看不懂局面变化，已暂停（请检查识别）"
            )
            return
        if self._check_idle():
            return
        self.state = State.THINKING

    def _step_thinking(self) -> None:
        # 返回值不能丢：`_check_idle()` 会把状态改成 PAUSED（轮到我方却异常静止），
        # 而下面第 367 行的 `self.state = State.PLAYING` 会把它盖回 PLAYING ——
        # 面板和日志上写着「请检查」，实际却在继续点击。真机评审实测复现过。
        if self._check_idle():
            return
        board = self.board
        if board is None:  # pragma: no cover - 不该发生
            self.state = State.IDLE
            return
        started = self.deps.now()
        analysis = self.deps.engine.analyse(  # type: ignore[attr-defined]
            board.to_fen(), self.deps.config.movetime_ms
        )
        self._think_ms = max(1, int((self.deps.now() - started) * 1000))
        self.last_analysis = analysis

        if not analysis.ok:
            self._unexplained += 1
            self.message = f"引擎没有返回着法（第 {self._unexplained} 次）"
            if self._unexplained >= MAX_UNEXPLAINED:
                self._auto_pause(State.MY_TURN, "引擎连续多次没有返回着法，已暂停")
            else:
                self.state = State.MY_TURN
            return

        mv = analysis.bestmove
        if mv not in set(legal_moves(board)):
            self._auto_pause(State.MY_TURN, f"引擎给出的着法 {mv} 在当前局面下不合法，已暂停")
            return
        self._unexplained = 0
        self._last_unexplained = None

        delays = self.deps.config.human_delay
        if delays:
            seconds = random.uniform(*delays)
            self.message = f"THINKING (human delay {seconds:.1f}s)"
            if not interruptible_sleep(
                seconds,
                lambda: self._pause_requested or self.should_stop,
                tick=self.deps.delay_tick,
            ):
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
            # 回到 PLAYING：F8 恢复后重试同一步点击，而不是丢掉这一步重新想。
            self._auto_pause(State.PLAYING, "拿不到棋盘几何，无法点击，已暂停")
            return

        self.deps.mouse.play_move(mv, geom)
        # 记谱**不在这里**写：点击可能根本没生效（`_step_verify` 超时后回到 `MY_TURN`
        # 重新决策），提前写就会把一步没走成的着法留在记谱里；重试成功后再写一次，
        # 同一回合就出现两手。`GameRecorder` 没有回滚接口，所以只能等确认。
        self._play_at = self.deps.now()
        self.message = f"已走 {mv}，等待画面确认"
        self.state = State.VERIFY

    def _step_verify(self) -> None:
        """等画面变成「我方这步走完」的样子。

        三条出路，按顺序判：

        1. 画面正好等于预期（容许 `tolerance_cells` 格识别抖动）→ 确认，轮到对方；
        2. 画面等于「我方走完**且对方已还着**」→ 两步一起收（对方想得太快时会发生）；
        3. 都没等到、且已经超时 → 认为这一步没生效，回到 `MY_TURN` 重新决策。

        第 1 条也要求**落点上有我方的子**：棋子飞在半路上时落点还是空的，那不叫「走成了」。
        少了这条，一次没生效的点击会被当成走子成功，后面整盘棋都跟着错位。

        抓图失败（`None`）时**什么都不改**，继续等 —— 坏帧不是「局面没变」。
        但一直读不到棋盘（对局已经结束、画面切到结算）就得收摊，见 `BOARD_LOST_TIMEOUT_S`。
        """
        mv = self._pending_move
        before = self._board_before
        if mv is None or before is None:  # pragma: no cover - 不该发生
            self.state = State.MY_TURN
            return
        expected = before.copy()
        make(expected, mv)
        dst_rank, dst_file = parse_move(mv)[1]

        cur = self.deps.board_provider()
        if self._note_board(cur):
            return
        if cur is None:
            return

        tol = self.deps.config.tolerance_cells
        landed = expected.grid[dst_rank, dst_file] == cur.grid[dst_rank, dst_file]
        if landed and _grid_diff(cur, expected) <= tol:
            # 允许 `tolerance_cells` 格对不上：吃子那一瞬间抓到的帧可能正好少一格，
            # 为它把整步判失败太亏。对方真还了一着至少动 2 格（起点 + 终点），
            # 所以 `tol = 1` 不会把「对方已经还着」误判成「我方走成了」。
            self._record_move(before, mv)
            self.board = expected
            self._idle_since = self.deps.now()
            self._pending_move = None
            self._board_before = None
            self.message = "走子已确认"
            self.state = State.OPPONENT_TURN
            return

        if _grid_diff(cur, before) != 0:
            # 画面确实动了，只是比我方这步多走了一步 —— 对方的还着已经落下来了。
            reply = interpret(expected, cur, tol)
            if reply is not None:
                self._record_move(before, mv)
                self.board = expected
                self._pending_move = None
                self._board_before = None
                self._adopt(reply, opponent=True)
                return

        if self.deps.now() - self._play_at < VERIFY_TIMEOUT_S:
            # 棋子是滑过去的，动画没放完就判失败会误伤（真机踩过）。
            return

        if getattr(self.deps.mouse, "dry_run", False):
            # 彩排时本来就没点鼠标，局面当然不会变 —— 这不算故障，别吓唬人。
            msg = "彩排：没真的点鼠标，局面不会变，已暂停（按 F10 关闭彩排）"
        else:
            msg = "点击后局面校验失败，已暂停"
        # 画面没变成预期 → 这一步当没走过，恢复后重新决策（`self.board` 也没动）。
        self._auto_pause(State.MY_TURN, msg)

    def _step_opponent_turn(self) -> None:
        # 静止计时先判：抓图一直失败（窗口最小化/被关掉）本身也是一种"对方没动"，
        # 拖过 `gameover_idle_sec` 就该收摊，而不是无限等下去。
        if self._check_idle():
            return
        cur = self.deps.board_provider()
        if self._note_board(cur):
            return
        if cur is None:
            # [RF-2] 抓图失败：**什么都不改**。绝不能把「没抓到」当成「棋盘空了」。
            return
        if self.board is not None and _grid_diff(cur, self.board) == 0:
            # 画面回到预期，说明刚才那个看不懂的局面是**瞬时的**（吃子动画把某一格
            # 闪没了、过渡帧抓早了一拍），不能攒着下次一起算账。
            self._unexplained = 0
            self._last_unexplained = None
            return

        mv = interpret(self.board, cur, self.deps.config.tolerance_cells)  # type: ignore[arg-type]
        if mv is None:
            self._count_unexplained(
                cur, State.OPPONENT_TURN, "无法解释局面变化，已暂停（请检查识别）"
            )
            return

        self._adopt(mv, opponent=True)

    # ---------------------------------------------------------------- 工具

    def _count_unexplained(self, cur: Board, resume_state: State, message: str) -> None:
        """记一笔「画面变了但解释不出来」，够 `MAX_UNEXPLAINED` 次就停下来。

        一个坏局面只记一次账：`board_provider()` 在新局面停稳之前会一直返回
        **同一个**已停稳的盘子，主循环一轮 0.1~0.2 秒，照着数次数的话一个坏帧
        会被连数三次，`MAX_UNEXPLAINED = 3` 立刻误触发暂停（真机实测踩过）。
        """
        fen = cur.to_fen()
        if fen != self._last_unexplained:
            self._last_unexplained = fen
            self._unexplained += 1
        self.message = f"看不懂局面变化（第 {self._unexplained} 次）"
        if self._unexplained >= MAX_UNEXPLAINED:
            self._auto_pause(resume_state, message)

    def _record_move(self, before: Board, mv: str) -> None:
        """把我方一步写进记谱。**只在确认走成之后调**（见 `_step_playing` 的注释）。"""
        recorder = self.deps.recorder
        if recorder is None:
            return
        analysis = self.last_analysis
        think_ms = self._think_ms
        if analysis is None or analysis.bestmove != mv:
            # 补收的着法（`_step_my_turn` 那条路）或引擎压根没给分析 —— 不硬凑数据。
            analysis, think_ms = None, None
        recorder.add(before, mv, analysis, think_ms)

    def _adopt(self, mv: str, *, opponent: bool) -> None:
        """采纳一步，把局面推到**修正后**的样子。

        `opponent=True` 是对方走的（采纳完轮到我方，并记进记谱）；
        `opponent=False` 是**我方**自己已经落下的着法（校验超时后补收），采纳完轮到对方 ——
        记谱由调用方在此之前用 `_record_move` 写，因为只有它知道这一手的分析数据。
        """
        before = self.board
        if before is None:  # pragma: no cover - 不该发生
            self.state = State.IDLE
            return
        delta = before.copy()
        make(delta, mv)
        if opponent and self.deps.recorder is not None:
            self.deps.recorder.add(before, mv, None, None)
        self.board = delta
        self._idle_since = self.deps.now()
        self._unexplained = 0
        self._last_unexplained = None
        self.message = f"对方走 {mv}" if opponent else f"走 {mv}"
        self.state = State.MY_TURN if opponent else State.OPPONENT_TURN

    def _note_board(self, cur: Board | None) -> bool:
        """记一笔「这次有没有读到棋盘」。`True` = 失联太久，已经转 `GAME_OVER`。

        对局结束（被将死、认输、结算画面弹出）时棋盘会整个消失，`board_provider()`
        从此一直给 `None`。这时候各状态里的「什么都不改，继续等」会永远等下去，
        最后被别的超时错误地报成「校验失败」。所以「读不到」也要有额度。
        """
        if cur is not None:
            self._board_lost_since = None
            return False
        now = self.deps.now()
        if self._board_lost_since is None:
            self._board_lost_since = now
        if now - self._board_lost_since < BOARD_LOST_TIMEOUT_S:
            return False
        self.state = State.GAME_OVER
        self.message = "已经看不到棋盘了，判定对局结束"
        return True

    def _check_idle(self) -> bool:
        """局面静止过久。返回 `True` 表示已经接管了状态，调用方必须返回。"""
        limit = self.deps.config.gameover_idle_sec
        if self.deps.now() - self._idle_since <= limit:
            return False
        if self.state is State.OPPONENT_TURN:
            # [RF-4] 对方长考、掉线或已经走人 —— 这一局结束了。
            self.state = State.GAME_OVER
            self.message = "局面长时间不变，判定对局结束"
            return True
        # [RF-4] 轮到我方却长时间不动 = 有东西不对劲，停下来等人看，别草率收摊。
        self._auto_pause(State.MY_TURN, "轮到我方但局面异常静止，请检查（按 F8 继续）")
        return True


def _looks_like_opening(board: Board) -> bool:
    """棋子够多、双方将帅都在 —— 这才像一局棋，而不是大厅菜单。"""
    values = board.grid
    if int(np.count_nonzero(values)) < _MIN_OPENING_PIECES:
        return False
    red_king = 1  # PIECE_CHARS 里的 K
    black_king = -1
    return bool((values == red_king).any() and (values == black_king).any())
