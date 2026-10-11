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


def test_连续重启计数成功一次就清零():
    """`restarts` 数的是**连续**崩溃，成功一次就该清零。

    回归测试：早先 `self.restarts += 1` 只增不减，而每个被引擎拒绝的局面要走
    「重启一次再试」花掉两次额度 —— 真机上交替跑「非法局面 + 正常局面」两轮，
    第三次会在分析**合法**开局时抛
    `EngineError: 皮卡鱼连续崩溃 3 次，已放弃`，异常穿出 `session.step()` 与
    `_play`，整局记谱丢失。
    """
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        for _ in range(3):
            bad = e.analyse("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1", movetime_ms=200)
            assert bad.bestmove == ""
            good = e.analyse(START_FEN, movetime_ms=300)
            assert len(good.bestmove) == 4, "连续崩溃计数没清零，合法局面也被拖死了"
            assert e.restarts == 0, "成功一次之后计数该归零"


def test_analyse绝不把EngineError抛给调用方():
    """契约：`analyse()` 不抛异常，最坏也给一个空 `Analysis`。

    `session._step_thinking` 没有 try，`main._play` 的异常网也只兜
    `WindowNotFound`/`KeyboardInterrupt` —— 一旦抛出去就是整局记谱丢失。
    这里把引擎搞成「进程没了 + 重启额度也用光」的最坏情况，两种局面都得扛住。
    """
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        e.quit()  # 进程没了，下一次 analyse 必然失败
        e.restarts = 10**6  # 重启额度也用光 -> `restart()` 会抛
        for fen in (START_FEN, "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"):
            a = e.analyse(fen, movetime_ms=300)
            assert a.ok is False, "最坏情况该给空结果，而不是把 EngineError 扔出去"
        assert e.restarts == 10**6, "放弃之后不该再白扣额度"


def test_握手失败不会留下野进程(tmp_path, monkeypatch):
    """握手失败时必须把刚拉起来的进程收掉。

    默认参数下泄漏的是一个 8 线程 / 512MB 置换表的皮卡鱼，而 `_open_engine`
    以前没有兜底，异常会直接从 `UciEngine(...)` 抛到 `main()`。
    """
    from jjchess.engine import uci

    started: list = []
    real_popen = uci.subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(uci.subprocess, "Popen", spy)
    monkeypatch.setattr(
        uci.UciEngine,
        "_handshake",
        lambda self: (_ for _ in ()).throw(EngineError("假装握手超时")),
    )
    with pytest.raises(EngineError):
        UciEngine(ENGINE, threads=2, hash_mb=64)

    assert len(started) == 1
    assert started[0].poll() is not None, "握手失败后进程还活着 —— 泄漏了"


def test_engine_present():
    assert engine_present(ENGINE) is True
    assert engine_present(Path("vendor/pikafish/不存在.exe")) is False


def test_引擎缺失时构造报错():
    with pytest.raises(EngineError):
        UciEngine(Path("vendor/pikafish/不存在.exe"))
