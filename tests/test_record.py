"""`jjchess.record` 的单元测试：对局记录与中文记谱导出。"""

from __future__ import annotations

from datetime import datetime

from jjchess.engine.uci import Analysis
from jjchess.geometry import BLACK, RED
from jjchess.record import GameRecorder
from jjchess.rules.board import START_FEN, Board
from jjchess.rules.moves import make


def _rec() -> GameRecorder:
    return GameRecorder(RED, START_FEN, datetime(2026, 10, 11, 15, 30, 22))


def _analysis() -> Analysis:
    return Analysis(
        bestmove="h2e2",
        score_cp=27,
        mate_in=None,
        depth=22,
        sel_depth=30,
        nps=14_000_000,
        pv=["h2e2", "h9g7"],
    )


def test_记谱内容正确():
    r = _rec()
    b = Board.from_fen(START_FEN)
    r.add(b, "h2e2", _analysis(), think_ms=1203)
    make(b, "h2e2")
    r.add(b, "h9g7")
    assert r.iccs_line() == "h2e2 h9g7"
    assert [m.chinese for m in r.moves] == ["炮二平五", "马8进7"]
    assert r.moves[0].score_cp == 27 and r.moves[0].depth == 22 and r.moves[0].think_ms == 1203
    assert r.moves[1].score_cp is None


def test_文本格式():
    r = _rec()
    b = Board.from_fen(START_FEN)
    r.add(b, "h2e2")
    t = r.to_text()
    assert "JJ象棋 对局记录" in t
    assert "开始时间: 2026-10-11 15:30:22" in t
    assert "我方执色: 红" in t
    assert f"初始局面: {START_FEN}" in t
    assert " 1. 红 炮二平五   (h2e2)" in t
    assert t.rstrip().endswith("ICCS: h2e2")


def test_保存为utf8(tmp_path):
    r = _rec()
    r.add(Board.from_fen(START_FEN), "h2e2")
    p = tmp_path / "g.txt"
    r.save(p)
    assert "炮二平五" in p.read_text(encoding="utf-8")


def test_黑方着法标注为黑():
    r = _rec()
    b = Board.from_fen(START_FEN)
    make(b, "h2e2")
    rec = r.add(b, "h9g7")
    assert rec.side == BLACK and " 1. 黑 马8进7" in r.to_text()


# --- 以下是计划之外补充的用例 -------------------------------------------------


def test_着法行逐字对齐spec示例():
    """红黑两行的中文记谱要按「显示宽度」补齐到同一列，否则数字与汉字混排会错位。"""
    r = _rec()
    b = Board.from_fen(START_FEN)
    r.add(b, "h2e2", _analysis(), think_ms=1203)
    make(b, "h2e2")
    r.add(b, "h9g7")
    make(b, "h9g7")
    r.add(b, "h0g2", _analysis(), think_ms=1198)
    lines = [ln for ln in r.to_text().splitlines() if ln.startswith((" 1.", " 2."))]
    assert lines == [
        " 1. 红 炮二平五   (h2e2)  +27   d22  1203ms",
        " 1. 黑 马8进7     (h9g7)",
        " 2. 红 马二进三   (h0g2)  +27   d22  1198ms",
    ]


def test_回合号红黑相同():
    r = _rec()
    b = Board.from_fen(START_FEN)
    r.add(b, "h2e2")
    make(b, "h2e2")
    r.add(b, "h9g7")
    make(b, "h9g7")
    r.add(b, "h0g2")
    assert [m.ply for m in r.moves] == [1, 2, 3]
    assert [ln.split(".")[0] for ln in r.to_text().splitlines() if "(" in ln] == [" 1", " 1", " 2"]


def test_结束原因取result字段():
    r = _rec()
    assert "结束原因: 未知" in r.to_text()
    r.add(Board.from_fen(START_FEN), "h2e2")
    r.result = "红方胜"
    assert "结束原因: 红方胜" in r.to_text()


def test_空记录也能导出():
    r = _rec()
    assert r.iccs_line() == ""
    t = r.to_text()
    assert "JJ象棋 对局记录" in t and t.rstrip().endswith("ICCS:")


def test_将杀着法标注mate():
    r = _rec()
    b = Board.from_fen("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1")
    r.add(b, "h5h9", Analysis(bestmove="h5h9", mate_in=1, depth=12), think_ms=41)
    line = r.moves[0]
    assert line.mate_in == 1 and line.score_cp is None
    assert "M1" in r.to_text()


def test_保存自动创建父目录(tmp_path):
    r = _rec()
    r.add(Board.from_fen(START_FEN), "h2e2")
    p = tmp_path / "records" / "sub" / "g.txt"
    r.save(p)
    assert p.is_file()


def test_执黑时我方执色显示为黑():
    r = GameRecorder(BLACK, START_FEN, datetime(2026, 10, 11, 15, 30, 22))
    assert "我方执色: 黑" in r.to_text()
