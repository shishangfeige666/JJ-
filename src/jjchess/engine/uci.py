"""皮卡鱼（Pikafish）UCI 封装。

两条硬约束来自实测，改代码时别踩：

1. **发完 `go` 绝不能马上 `quit`**。早前用管道一次性灌完命令，搜索被截断在
   depth 1，返回的是垃圾着法。`analyse()` 必须阻塞读完 `bestmove` 行。
2. **皮卡鱼 2026-09-06 对局面做严格校验**。送将帅照面之类的非法 FEN，它打印
   `info string CRITICAL ERROR` 后**直接 exit code 1 自杀**。所以这里必须能
   发现进程意外退出、自动重启，并且**不把异常抛给调用方**——上层是按帧识别
   的，一帧识别错就整个脚本崩掉是不可接受的。

`bestmove` 行后面可能跟 `ponder <move>`，解析时只取第一个 token。

Windows 下 `Popen.stdout.readline()` 没法设超时，所以每个进程配一个 reader
线程把行推进 `queue.Queue`，主线程用 `queue.get(timeout=...)` 实现超时。
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

#: 仓库里皮卡鱼可执行文件的默认位置。
DEFAULT_ENGINE_PATH = Path("vendor/pikafish/pikafish.exe")

#: 连续重启超过这个次数就认定引擎真坏了，不再吞异常。
MAX_RESTARTS = 3

_HANDSHAKE_TIMEOUT = 20.0
_READY_TIMEOUT = 30.0
#: 求招超时 = 思考时间 + 这么多秒的余量。
_GO_SLACK_SEC = 10.0

#: 皮卡鱼拒绝局面时打印的标记。
_REJECT_MARK = "CRITICAL ERROR"

#: UCI 在无着可走时回的占位符。
_NO_MOVE = "(none)"


class EngineError(RuntimeError):
    """引擎不可用、超时或意外退出。

    注意：`analyse()` 在重试后仍失败时**不会**抛这个异常，而是返回一个
    `bestmove == ""` 的空 `Analysis`，让上层跳过当前帧。只有连续崩溃超过
    `MAX_RESTARTS` 次才会抛出来。
    """


@dataclass
class Analysis:
    """一次求招的结果。没有着法时 `bestmove` 为空串。"""

    bestmove: str = ""
    score_cp: int | None = None
    mate_in: int | None = None
    depth: int = 0
    sel_depth: int = 0
    nps: int = 0
    pv: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否拿到了一个可用的着法。"""
        return len(self.bestmove) == 4


def engine_present(exe: Path | None = None) -> bool:
    """引擎可执行文件是否就位。"""
    path = Path(exe) if exe is not None else DEFAULT_ENGINE_PATH
    return path.is_file()


def _parse_info(line: str, state: dict) -> None:
    """把一行 `info ...` 解析进 `state`（覆盖式，只保留最后一条）。"""
    tok = line.split()
    i = 1
    while i < len(tok):
        t = tok[i]
        if t in ("depth", "seldepth", "nps", "hashfull", "time") and i + 1 < len(tok):
            if t in ("depth", "seldepth", "nps"):
                try:
                    state[t] = int(tok[i + 1])
                except ValueError:
                    pass
            i += 2
        elif t == "score" and i + 2 < len(tok):
            kind, value = tok[i + 1], tok[i + 2]
            try:
                if kind == "cp":
                    state["score_cp"] = int(value)
                    state["mate_in"] = None
                elif kind == "mate":
                    state["mate_in"] = int(value)
                    state["score_cp"] = None
            except ValueError:
                pass
            i += 3
        elif t == "pv":
            state["pv"] = tok[i + 1 :]
            break
        else:
            i += 1


class UciEngine:
    """一个皮卡鱼子进程。

    用作上下文管理器：`with UciEngine(exe) as e: e.analyse(fen)`。
    """

    def __init__(
        self,
        exe: Path,
        threads: int = 8,
        hash_mb: int = 512,
        multipv: int = 1,
        nnue: Path | None = None,
    ) -> None:
        self.exe = Path(exe)
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.nnue = Path(nnue) if nnue is not None else None

        self.restarts = 0

        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._start()

    # ---------------------------------------------------------------- 生命周期

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.quit()

    def _start(self) -> None:
        """拉起进程、起 reader 线程、完成握手。"""
        if not self.exe.is_file():
            raise EngineError(f"找不到皮卡鱼引擎：{self.exe}")

        # 每次重开都换一个新队列，避免旧进程的残留输出污染新一轮读取。
        self._lines = queue.Queue()
        try:
            proc = subprocess.Popen(
                [str(self.exe)],
                cwd=str(self.exe.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise EngineError(f"启动皮卡鱼失败：{exc}") from exc

        self._proc = proc
        self._reader = threading.Thread(target=self._pump, args=(proc, self._lines), daemon=True)
        self._reader.start()
        self._handshake()

    @staticmethod
    def _pump(proc: subprocess.Popen, sink: queue.Queue) -> None:
        """把子进程 stdout 逐行推进队列；EOF 时推一个 `None` 当哨兵。"""
        stream = proc.stdout
        try:
            if stream is not None:
                for line in stream:
                    sink.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            sink.put(None)

    def quit(self) -> None:
        """发 `quit` 并回收进程；不抛异常（清理路径要能随便调）。"""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def restart(self) -> None:
        """重开引擎。连续重启超过 `MAX_RESTARTS` 次则抛 `EngineError`。"""
        self.restarts += 1
        if self.restarts > MAX_RESTARTS:
            raise EngineError(f"皮卡鱼连续崩溃 {MAX_RESTARTS} 次，已放弃")
        self.quit()
        self._start()

    # ---------------------------------------------------------------- 收发

    def _send(self, cmd: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EngineError("皮卡鱼进程未启动")
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError(f"向皮卡鱼发送命令失败：{cmd}") from exc

    def _read_line(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=max(timeout, 0.001))
        except queue.Empty:
            raise EngineError(f"读取皮卡鱼输出超时（{timeout:.1f}s）") from None
        if line is None:
            raise EngineError("皮卡鱼进程已退出")
        return line

    def _read_until(self, want, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError(f"等待皮卡鱼响应超时（{timeout:.0f}s）")
            line = self._read_line(remaining)
            if want(line):
                return line

    def _handshake(self) -> None:
        self._send("uci")
        self._read_until(lambda ln: ln.strip() == "uciok", _HANDSHAKE_TIMEOUT)

        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        if self.multipv != 1:
            self._send(f"setoption name MultiPV value {self.multipv}")
        # 权重是独立文件、必须与 exe 同目录。显式指定一份存在的，缺了就交给
        # 引擎的默认值（它自己会在 exe 同目录找 pikafish.nnue）。
        if self.nnue is not None and self.nnue.is_file():
            self._send(f"setoption name EvalFile value {self.nnue.resolve()}")

        self._send("isready")
        self._read_until(lambda ln: ln.strip() == "readyok", _READY_TIMEOUT)

    # ---------------------------------------------------------------- 求招

    def analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis:
        """求一步棋。

        局面被引擎拒绝或进程崩掉时，重启一次再试；仍然失败就返回
        `bestmove == ""` 的空结果（不抛异常，让上层跳过这一帧）。
        """
        try:
            return self._analyse_once(fen, movetime_ms)
        except EngineError:
            pass

        self.restart()
        try:
            return self._analyse_once(fen, movetime_ms)
        except EngineError:
            return Analysis()

    def _analyse_once(self, fen: str, movetime_ms: int) -> Analysis:
        self._send(f"position fen {fen}")
        self._send(f"go movetime {int(movetime_ms)}")

        deadline = time.monotonic() + movetime_ms / 1000.0 + _GO_SLACK_SEC
        state: dict = {"depth": 0, "seldepth": 0, "nps": 0, "pv": []}

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("等待 bestmove 超时")
            line = self._read_line(remaining)

            if _REJECT_MARK in line:
                raise EngineError(f"皮卡鱼拒绝该局面：{line.strip()}")
            if self._proc is None or self._proc.poll() is not None:
                raise EngineError("皮卡鱼进程意外退出")

            if line.startswith("info "):
                _parse_info(line, state)
                continue
            if line.startswith("bestmove"):
                tok = line.split()
                move = tok[1] if len(tok) > 1 else ""
                if move == _NO_MOVE:
                    move = ""
                return Analysis(
                    bestmove=move,
                    score_cp=state.get("score_cp"),
                    mate_in=state.get("mate_in"),
                    depth=int(state.get("depth", 0)),
                    sel_depth=int(state.get("seldepth", 0)),
                    nps=int(state.get("nps", 0)),
                    pv=list(state.get("pv", [])),
                )
