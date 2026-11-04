"""XQ-FEN 解析/序列化与 Board 数据结构的测试。"""

import pytest

from jjchess.rules.board import (
    BLACK,
    RED,
    START_FEN,
    Board,
    K,
    P,
    R,
    parse_square,
    square,
)


def test_start_fen_roundtrip():
    assert Board.from_fen(START_FEN).to_fen() == START_FEN


def test_start_position_has_32_pieces():
    b = Board.from_fen(START_FEN)
    assert len(b.pieces(RED)) == 16 and len(b.pieces(BLACK)) == 16


def test_红方底线在_rank_0():
    b = Board.from_fen(START_FEN)
    assert b.at(0, 0) == R  # a0 = 红车
    assert b.at(0, 4) == K  # e0 = 红帅
    assert b.at(9, 4) == -K  # e9 = 黑将
    assert b.at(3, 0) == P  # a3 = 红兵
    assert b.at(6, 0) == -P  # a6 = 黑卒


def test_square_iccs():
    assert square(0, 0) == "a0"
    assert square(9, 4) == "e9"
    assert parse_square("h2") == (2, 7)


def test_非法_fen_被拒绝():
    for bad in [
        "",
        "9/9/9/9/9/9/9/9/9/9 w - - 0 1",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x",
        "rnbakabnr/8/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
    ]:
        with pytest.raises(ValueError):
            Board.from_fen(bad)
