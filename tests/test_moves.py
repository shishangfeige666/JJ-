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
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/2E1K4 w - - 0 1")
    assert all(not m.startswith("c0") or int(m[3]) <= 4 for m in generate(b))


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
