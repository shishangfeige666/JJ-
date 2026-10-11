"""皮卡鱼 UCI 封装测试。

这些测试会在本机真的把 `vendor/pikafish/pikafish.exe` 拉起来跑搜索，
所以比纯逻辑测试慢（每个几百毫秒到几秒）。引擎不在时整体跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jjchess.engine.uci import EngineError, UciEngine, engine_present
from jjchess.rules.board import START_FEN

ENGINE = Path("vendor/pikafish/pikafish.exe")

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="皮卡鱼未就绪")


def test_握手与求招():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse(START_FEN, movetime_ms=800)
    assert len(a.bestmove) == 4 and a.bestmove[0] in "abcdefghi"
    assert a.depth >= 8 and a.pv and a.pv[0] == a.bestmove


def test_一步杀():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1", movetime_ms=600)
    assert a.bestmove == "h5h9" and a.mate_in == 1


def test_发go后不能提前quit_必须等到bestmove():
    # 回归测试：曾经把 quit 紧跟 go，导致只搜到 depth 1 返回垃圾着法
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse(START_FEN, movetime_ms=1500)
    assert a.depth >= 12, f"搜索被提前终止，depth 只有 {a.depth}"


def test_非法fen不崩溃_自动重启():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1", movetime_ms=300)  # 将帅照面
        assert e.restarts >= 1
        assert a.bestmove == "", "被拒局面不该给出着法"
        b = e.analyse(START_FEN, movetime_ms=400)  # 重启后仍能正常工作
    assert len(b.bestmove) == 4


def test_engine_present():
    assert engine_present(ENGINE) is True
    assert engine_present(Path("vendor/pikafish/不存在.exe")) is False


def test_引擎缺失时构造报错():
    with pytest.raises(EngineError):
        UciEngine(Path("vendor/pikafish/不存在.exe"))
