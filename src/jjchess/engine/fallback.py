"""浅层 α-β 兜底引擎。

皮卡鱼没就位（没下载、被杀软删了、加载 NNUE 失败）时顶上，保证脚本还能走完
一盘棋。目标只有一个：**别走非法着法**，顺便认得出一步杀。棋力不重要。

`analyse()` 的签名与 `UciEngine.analyse` 逐字一致，可以在 `main.py` 里原地互换。
"""

from __future__ import annotations

import time
from typing import Self

from ..rules.board import RED, A, Board, C, E, H, K, P, R
from ..rules.moves import game_status, legal_moves, make, parse_move, unmake
from .uci import Analysis

#: 子力价值。帅/将给一个压倒性的分值，被吃就等于输棋。
MATERIAL: dict[int, int] = {K: 10000, R: 900, C: 450, H: 400, A: 200, E: 200, P: 100}

#: 兵/卒每前进一排的加分——逼它往前走，别在原地磨。
_PAWN_ADVANCE = 10

_MATE = 30000
_INF = 1 << 20
_MAX_PLY = 100
#: 主变最多记这么多步。
_MAX_PV = 8


class _TimeUp(Exception):
    """内部信号：思考时间到，放弃当前这一层。"""


class FallbackEngine:
    """纯 Python 的 negamax + α-β，迭代加深。"""

    def __init__(self, depth: int = 3) -> None:
        self.depth = max(1, int(depth))
        self._deadline: float | None = None
        self._nodes = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    # ---------------------------------------------------------------- 求招

    def analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis:
        """迭代加深搜索；`depth` 与 `movetime_ms` 谁先到头就停在哪。"""
        board = Board.from_fen(fen)
        started = time.monotonic()
        self._nodes = 0
        self._deadline = started + max(movetime_ms, 1) / 1000.0

        best = ""
        score = 0
        completed = 0
        for d in range(1, self.depth + 1):
            try:
                cand, val = self._root_search(board, d)
            except _TimeUp:
                break
            if not cand:
                break
            best, score, completed = cand, val, d
            # 已经看到杀棋就没必要再往深了搜。
            if abs(val) >= _MATE - _MAX_PLY:
                break

        if completed == 0:
            # 连一层都没搜完：不限时再搜一层，保证一定有棋可走。
            self._deadline = None
            best, score = self._root_search(board, 1)
            completed = 1

        self._deadline = None
        elapsed = max(time.monotonic() - started, 1e-6)
        pv = self._principal_variation(board, best, completed) if best else []

        mate_in: int | None = None
        score_cp: int | None = score
        if abs(score) >= _MATE - _MAX_PLY:
            dist = _MATE - abs(score)
            mate_in = dist if score > 0 else -dist
            score_cp = None

        return Analysis(
            bestmove=best,
            score_cp=score_cp,
            mate_in=mate_in,
            depth=completed,
            sel_depth=completed,
            nps=int(self._nodes / elapsed),
            pv=pv,
        )

    # ---------------------------------------------------------------- 搜索

    def _root_search(self, board: Board, depth: int) -> tuple[str, int]:
        """根节点搜索，返回 `(最佳着法, 分值)`；超时抛 `_TimeUp`。"""
        moves = self._ordered(board, legal_moves(board))
        if not moves:
            return "", 0

        alpha = -_INF
        best_move = ""
        best_val = -_INF
        for mv in moves:
            self._check_time()
            cap = make(board, mv)
            try:
                val = -self._negamax(board, depth - 1, -_INF, -alpha, 1)
            finally:
                unmake(board, mv, cap)
            if val > best_val:
                best_val, best_move = val, mv
            alpha = max(alpha, best_val)
        return best_move, best_val

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        # 终局判定必须排在 depth 前面：深度用完时也可能正好是将死局面。
        status = game_status(board)
        if status == "CHECKMATE":
            return -_MATE + ply
        if status == "STALEMATE":
            # 象棋里「困毙」（无着可走但没被将军）是**判负**，不是和棋 ——
            # 中国象棋规则与围棋/国际象棋不同。早先这里返回 0，兜底引擎在劣势时
            # 会主动把「被对方困毙」当成守和着一个 0 分的着法，能困毙对手时也认不出赢棋。
            return -_MATE + ply
        if depth <= 0:
            self._nodes += 1
            return self._evaluate(board)

        self._check_time()
        best = -_INF
        for mv in self._ordered(board, legal_moves(board)):
            cap = make(board, mv)
            try:
                val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                unmake(board, mv, cap)
            best = max(best, val)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _check_time(self) -> None:
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise _TimeUp

    def _ordered(self, board: Board, moves: list[str]) -> list[str]:
        """吃大子的着法先搜，提高剪枝率。"""

        def gain(mv: str) -> int:
            (_r1, _c1), (r2, c2) = parse_move(mv)
            return MATERIAL.get(abs(int(board.grid[r2][c2])), 0)

        return sorted(moves, key=lambda mv: -gain(mv))

    def _evaluate(self, board: Board) -> int:
        """子力总和 + 兵前进奖励，站在**当前行棋方**的角度。"""
        grid = board.grid
        score = 0
        for rank in range(10):
            row = grid[rank]
            for file in range(9):
                v = int(row[file])
                if v == 0:
                    continue
                if v > 0:
                    score += MATERIAL[v]
                    if v == P:
                        score += rank * _PAWN_ADVANCE
                else:
                    score -= MATERIAL[-v]
                    if v == -P:
                        score -= (9 - rank) * _PAWN_ADVANCE
        return score if board.side == RED else -score

    def _principal_variation(self, board: Board, first: str, depth: int) -> list[str]:
        """顺着每一步的最佳着法走出一条主变（分数不计入 `nps`）。"""
        pv = [first]
        work = board.copy()
        mv = first
        left = depth - 1
        while left > 0 and len(pv) < _MAX_PV:
            make(work, mv)
            if game_status(work) in ("CHECKMATE", "STALEMATE"):
                break
            nxt, _ = self._root_search(work, left)
            if not nxt:
                break
            pv.append(nxt)
            mv = nxt
            left -= 1
        return pv
