"""对局记录与中文记谱导出。

一整局走完（或被用户 `ESC` 中断）之后，把着法序列写成一份人可以读的中文记谱 txt，
末尾附一行 ICCS 便于程序复用。格式见 `docs/superpowers/specs/2026-10-11-jjchess-autoplay-design.md` §4.9：

```
JJ象棋 对局记录
开始时间: 2026-10-11 15:30:22
我方执色: 红
初始局面: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
结束原因: 用户退出

 1. 红 炮二平五   (h2e2)  +27   d22  1203ms
 1. 黑 马8进7     (h9g7)

ICCS: h2e2 h9g7
```

记谱本身复用 `jjchess.rules.notation.to_chinese`，本模块只管「记下来」和「排成文本」。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jjchess.geometry import BLACK, RED
from jjchess.rules.notation import to_chinese

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注，避免运行时循环导入
    from jjchess.engine.uci import Analysis
    from jjchess.rules.board import Board

__all__ = ["SIDE_NAMES", "GameRecorder", "MoveRecord"]

SIDE_NAMES: dict[int, str] = {RED: "红", BLACK: "黑"}

#: 中文记谱列在文本里补齐到的显示宽度（汉字算 2 列、数字算 1 列）。
#: 这样「炮二平五」（8 列）与「马8进7」（6 列）后面的 `(iccs)` 才会对齐 —— 见 spec §4.9 的示例。
_NOTATION_COLUMN = 11


def _display_width(text: str) -> int:
    """按终端显示宽度算长度：全角字符算 2 列，其余算 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


@dataclass
class MoveRecord:
    """一个半步（ply）的记录。"""

    ply: int  # 从 1 开始的半步序号
    side: int  # RED(1) / BLACK(-1)：走这一手的一方
    iccs: str  # "h2e2"
    chinese: str  # "炮二平五"
    score_cp: int | None = None  # 我方着法才有；对手着法为 None
    mate_in: int | None = None
    depth: int | None = None
    think_ms: int | None = None

    @property
    def side_name(self) -> str:
        return SIDE_NAMES.get(self.side, "?")

    @property
    def turn(self) -> int:
        """回合号：红黑两行共用同一个号（见 spec §4.9 的示例）。"""
        return (self.ply + 1) // 2

    def line(self, *, with_analysis: bool) -> str:
        """排成一行着法。`with_analysis` 为假时省略评分/深度/耗时后缀。"""
        chinese = self.chinese
        pad = " " * max(1, _NOTATION_COLUMN - _display_width(chinese))
        out = f"{self.turn:2d}. {self.side_name} {chinese}{pad}({self.iccs})"
        if not with_analysis:
            return out

        score = ""
        if self.mate_in is not None:
            score = f"M{abs(self.mate_in)}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        elif self.score_cp is not None:
            score = f"{self.score_cp:+d}"
        if score:
            out += f"  {score}"
        if self.depth:
            out += ("   " if score else "  ") + f"d{self.depth}"
        if self.think_ms is not None:
            out += f"  {self.think_ms}ms"
        return out


class GameRecorder:
    """累积一整局的着法，结束时导出中文记谱文本。"""

    def __init__(self, my_side: int, start_fen: str, started_at: datetime) -> None:
        if my_side not in (RED, BLACK):
            raise ValueError(f"my_side 必须是 RED(1) 或 BLACK(-1)，收到 {my_side!r}")
        self.my_side = my_side
        self.start_fen = start_fen
        self.started_at = started_at
        self.moves: list[MoveRecord] = []
        self.result: str = "未知"

    def add(
        self,
        board_before: Board,
        mv: str,
        analysis: Analysis | None = None,
        think_ms: int | None = None,
    ) -> MoveRecord:
        """记录一步。`board_before` 必须是**走这一步之前**的局面（记谱要靠它定位棋子）。"""
        record = MoveRecord(
            ply=len(self.moves) + 1,
            side=board_before.side,
            iccs=mv,
            chinese=to_chinese(board_before, mv),
            score_cp=analysis.score_cp if analysis is not None else None,
            mate_in=analysis.mate_in if analysis is not None else None,
            depth=analysis.depth if analysis is not None else None,
            think_ms=think_ms,
        )
        self.moves.append(record)
        return record

    def iccs_line(self) -> str:
        """所有着法的 ICCS 单行形式，便于程序复用。"""
        return " ".join(m.iccs for m in self.moves)

    def to_text(self) -> str:
        """中文记谱 txt 全文。"""
        lines = [
            "JJ象棋 对局记录",
            f"开始时间: {self.started_at:%Y-%m-%d %H:%M:%S}",
            f"我方执色: {SIDE_NAMES.get(self.my_side, '?')}",
            f"初始局面: {self.start_fen}",
            f"结束原因: {self.result}",
            "",
        ]
        lines.extend(m.line(with_analysis=(m.side == self.my_side)) for m in self.moves)
        lines.append("")
        lines.append(f"ICCS: {self.iccs_line()}")
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        """写 UTF-8 文件，父目录不存在时自动创建。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text(), encoding="utf-8")
