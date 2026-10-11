"""走法生成、将军判定、终局判定与 perft 的测试。"""

import pytest

from jjchess.rules.board import RED, START_FEN, Board
from jjchess.rules.moves import (
    game_status,
    generate,
    in_check,
    legal_moves,
    make,
    perft,
)


@pytest.mark.parametrize("depth,expected", [(1, 44), (2, 1920), (3, 79666), (4, 3290240)])
def test_perft_startpos(depth, expected):
    assert perft(Board.from_fen(START_FEN), depth) == expected


def test_蹩马腿():
    # b0 有红马、b1 有红兵：马往 a2/c2 的两条路都以 b1 为马腿，被己方兵堵死，
    # 只剩 d1（马腿 c0 为空）可走。
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/1P7/1N2K4 w - - 0 1")
    mvs = {m for m in generate(b) if m.startswith("b0")}
    assert mvs == {"b0d1"}


def test_马腿通畅时八面威风():
    # 同一个马，把 b1 的兵拿掉后，a2/c2 两个落点就通了。
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/1N2K4 w - - 0 1")
    mvs = {m for m in generate(b) if m.startswith("b0")}
    assert mvs == {"b0a2", "b0c2", "b0d1"}


def test_炮必须隔一个炮架才能吃():
    # 红炮 f0、黑车 f3，中间无子 -> 炮不能吃。
    no_screen = Board.from_fen("4k4/9/9/9/9/9/5r3/9/9/3K1C3 w - - 0 1")
    assert "f0f3" not in generate(no_screen)
    assert "f0f1" in generate(no_screen)  # 但可以平移

    # 中间垫一个子当炮架 -> 可以吃。
    one_screen = Board.from_fen("4k4/9/9/9/9/9/5r3/5P3/9/3K1C3 w - - 0 1")
    assert "f0f3" in generate(one_screen)

    # 中间垫两个子 -> 又吃不了。
    two_screens = Board.from_fen("4k4/9/9/9/9/9/5r3/5P3/5P3/3K1C3 w - - 0 1")
    assert "f0f3" not in generate(two_screens)


def test_象不过河():
    """相/象走田字，但不能过河。

    注意用例里的相必须摆在**河界边上**（rank 4）：摆在底线（rank 0）时它的两个
    田字落点本来就在本方半场，这条断言无论规则在不在都通过 —— 是个恒真断言
    （整分支评审实测：把 `_on_own_side` 判断去掉后，c0 相的着法集合一模一样，
    perft(1..4) 也仍然是 44/1920/79666/3290240，因为红象跨河要走 5 个半步，
    depth 4 采不到）。
    """
    edge = Board.from_fen("4k4/9/9/9/9/2E6/9/9/9/4K4 w - - 0 1")
    moves = generate(edge)
    assert "c4a2" in moves and "c4e2" in moves  # 往本方半场走没问题
    assert "c4a6" not in moves and "c4e6" not in moves  # 过河的两步被挡掉

    # 黑象对称：摆到 rank 5（黑方河界边上），不能走到 rank 3 及以下。
    # FEN 字符串第 i 行对应棋盘 rank 9-i，所以 rank 5 写在第 5 行（下标 4）。
    black = Board.from_fen("4k4/9/9/9/2e6/9/9/9/9/4K4 b - - 0 1")
    black_moves = generate(black)
    assert "c5a7" in black_moves and "c5e7" in black_moves
    assert "c5a3" not in black_moves and "c5e3" not in black_moves


def test_塞象眼():
    # c0 相 -> e2 的田心是 d1；在 d1 放一个子就堵死了。
    open_eye = Board.from_fen("4k4/9/9/9/9/9/9/9/9/2E1K4 w - - 0 1")
    assert "c0e2" in generate(open_eye)
    blocked = Board.from_fen("4k4/9/9/9/9/9/9/9/3P5/2E1K4 w - - 0 1")
    assert "c0e2" not in generate(blocked)


def test_兵过河后能横走():
    # b4 兵还没过河（红方河界在 rank 5）。
    b4 = Board.from_fen("4k4/9/9/9/9/1P7/9/9/9/4K4 w - - 0 1")
    assert "b4a4" not in generate(b4) and "b4c4" not in generate(b4)
    assert "b4b5" in generate(b4)  # 但能直进

    # b5 兵已过河，可以横走。
    b5 = Board.from_fen("4k4/9/9/9/1P7/9/9/9/9/4K4 w - - 0 1")
    assert "b5a5" in generate(b5) and "b5c5" in generate(b5)


def test_将帅照面非法():
    # 中路无遮挡 -> 红帅被黑将"照面"攻击。
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    assert in_check(b, RED)

    # 中间垫一个子就不算照面。
    blocked = Board.from_fen("4k4/9/9/9/9/9/9/9/4P4/4K4 w - - 0 1")
    assert not in_check(blocked, RED)


def test_一步杀判定():
    b = Board.from_fen("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1")
    assert "h5h9" in legal_moves(b)
    make(b, "h5h9")
    assert game_status(b) == "CHECKMATE"
