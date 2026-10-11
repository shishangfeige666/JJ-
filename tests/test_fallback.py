"""浅层 α-β 兜底引擎测试。

皮卡鱼不在时脚本还能走棋，靠的就是这个。它不追求棋力，只要求：
给出**合法**着法、认得出一步杀、接口与 `UciEngine` 完全一致。
"""

from __future__ import annotations

import inspect

from jjchess.engine.fallback import _INF, _MATE, MATERIAL, FallbackEngine
from jjchess.engine.uci import UciEngine
from jjchess.geometry import BLACK
from jjchess.rules.board import START_FEN, Board
from jjchess.rules.moves import game_status, in_check, legal_moves


def test_找到一步杀():
    e = FallbackEngine(depth=2)
    a = e.analyse("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1")
    assert a.bestmove == "h5h9" and a.mate_in == 1


def test_返回合法着法():
    e = FallbackEngine(depth=2)
    a = e.analyse(START_FEN)
    assert a.bestmove in legal_moves(Board.from_fen(START_FEN))


def test_分析对象字段完整():
    a = FallbackEngine(depth=2).analyse(START_FEN)
    assert isinstance(a.pv, list) and a.depth == 2 and a.nps >= 0


def test_接口与UciEngine一致():
    # 兜底引擎要能原地替换 UciEngine，签名必须一样。
    assert inspect.signature(FallbackEngine.analyse) == inspect.signature(UciEngine.analyse)


def test_可作上下文管理器():
    with FallbackEngine(depth=1) as e:
        a = e.analyse(START_FEN)
    assert a.bestmove in legal_moves(Board.from_fen(START_FEN))


def test_子力价值表():
    # K=1 A=2 E=3 H=4 R=5 C=6 P=7
    assert MATERIAL[1] == 10000 and MATERIAL[7] == 100
    assert (
        MATERIAL[7]
        < MATERIAL[2]
        == MATERIAL[3]
        < MATERIAL[4]
        < MATERIAL[6]
        < MATERIAL[5]
        < MATERIAL[1]
    )


def test_困毙判负而不是判和():
    """象棋里「困毙」（无着可走但没被将军）是**判负**，与围棋/国际象棋不同。

    回归测试：`_negamax` 早先对 `STALEMATE` 返回 0，于是兜底引擎在劣势时会主动把
    「被对方困毙」当成守和着一个 0 分的着法，能困毙对手时也认不出赢棋
    （`mate_in` 永远不填）。
    """
    # 黑将 d9；红车 a8 封住第 8 行、红车 e1 封住 e9，红帅 f1（与黑将不同纵线，不照面）。
    # 黑将的两个落点 e9 / d8 都被封死，而它自己没被将军 —— 标准的困毙。
    board = Board.from_fen("3k5/R8/9/9/9/9/9/9/4RK3/9 b - - 0 1")
    assert game_status(board) == "STALEMATE"
    assert legal_moves(board) == [], "这个局面本来就该是无着可走"
    assert in_check(board, BLACK) is False, "困毙的前提是没被将军"

    e = FallbackEngine(depth=2)
    value = e._negamax(board, depth=3, alpha=-_INF, beta=_INF, ply=2)
    assert value == -_MATE + 2, "困毙要按输棋算，不能给 0 分"
