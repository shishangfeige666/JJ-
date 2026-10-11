"""中文纵线记谱的双向转换测试。"""

import pytest

from jjchess.rules.board import START_FEN, Board
from jjchess.rules.moves import make
from jjchess.rules.notation import from_chinese, to_chinese


def test_炮二平五():
    assert to_chinese(Board.from_fen(START_FEN), "h2e2") == "炮二平五"


def test_马8进7():
    b = Board.from_fen(START_FEN)
    make(b, "h2e2")  # 红先走一步，轮到黑
    assert to_chinese(b, "h9g7") == "马8进7"


def test_兵七进一():
    assert to_chinese(Board.from_fen(START_FEN), "c3c4") == "兵七进一"


def test_士象斜走时数值取目标纵线号():
    # 红仕 d0 -> e1：红方纵线 d=六、e=五，斜走用目标纵线号 -> 「仕六进五」。
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/3A1K3 w - - 0 1")
    assert to_chinese(b, "d0e1") == "仕六进五"


def test_黑方用阿拉伯数字且进退方向相反():
    # 黑卒 e6 向前（rank 减小）走一步。
    b = Board.from_fen("4k4/9/9/4p4/9/9/9/9/9/3K5 b - - 0 1")
    assert to_chinese(b, "e6e5") == "卒5进1"

    # 黑卒 e4 已过河，可以横走，横走一律记「平」+ 目标纵线号。
    b2 = Board.from_fen("4k4/9/9/9/9/4p4/9/9/9/3K5 b - - 0 1")
    assert to_chinese(b2, "e4d4") == "卒5平4"


def test_同线双子用前后消歧():
    # a 线两个红车：a0 与 a5 -> 靠前者(a5)称"前车"，靠后者(a0)称"后车"
    b = Board.from_fen("4k4/9/9/9/R8/9/9/9/9/R3K4 w - - 0 1")
    assert to_chinese(b, "a5a6") == "前车进一"
    assert to_chinese(b, "a0a1") == "后车进一"


@pytest.mark.parametrize("mv", ["h2e2", "b2e2", "h2h6", "c3c4", "b0c2", "a0a1"])
def test_往返一致(mv):
    b = Board.from_fen(START_FEN)
    assert from_chinese(b, to_chinese(b, mv)) == mv


def test_胡说八道返回None():
    b = Board.from_fen(START_FEN)
    assert from_chinese(b, "炮九平五") is None
    assert from_chinese(b, "不是着法") is None
