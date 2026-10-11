"""棋盘数据结构与 XQ-FEN 解析/序列化。

坐标约定（全项目统一，勿改）：

- **板坐标** `rank ∈ [0, 9]`，`rank 0` 是**红方底线**；`file ∈ [0, 8]`，
  `file 0` 是 **a 纵线**（红方视角最左）。
- 内部数组索引一律是 `grid[rank, file]`。
- **ICCS 记法**：`chr(ord('a') + file) + str(rank)`，如 `a0`、`e9`。

棋子编码：红方为正、黑方为负，`K/A/E/H/R/C/P = 1..7`（帅仕相马车炮兵），
`0` 表示空点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- 走子方 ----
RED = 1
BLACK = -1

# ---- 棋子类型（红方取正值，黑方取其相反数）----
K = 1  # 帅 / 将
A = 2  # 仕 / 士
E = 3  # 相 / 象
H = 4  # 马 / 馬
R = 5  # 车 / 車
C = 6  # 炮 / 砲
P = 7  # 兵 / 卒

#: 象棋标准开局 XQ-FEN。
START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

#: 棋子编码 → FEN 字符。红方大写、黑方小写，`0` 记作 `.`。
#:
#: 字母沿用 XQ-FEN（国际象棋式）而非拼音首字母：`B`=相/象、`N`=马/馬。
#: 这是皮卡鱼唯一接受的拼法（开局 FEN 里就是 `rnbakabnr`），
#: 写成拼音的 `E`/`H` 会被引擎判为非法局面并自杀退出。
PIECE_CHARS: dict[int, str] = {
    K: "K",
    A: "A",
    E: "B",
    H: "N",
    R: "R",
    C: "C",
    P: "P",
    -K: "k",
    -A: "a",
    -E: "b",
    -H: "n",
    -R: "r",
    -C: "c",
    -P: "p",
    0: ".",
}

#: FEN 字符 → 棋子编码（`PIECE_CHARS` 的反向表）。
CHAR_VALUES: dict[str, int] = {ch: value for value, ch in PIECE_CHARS.items()}

#: FEN 里表示空点的字符（`0` 与 `.` 都接受，输出统一用 `.`）。
_EMPTY_CHARS = frozenset("0.")

_BOARD_ROWS = 10
_BOARD_COLS = 9
#: XQ-FEN 缺省的后四段（走子方缺省为红方）。
_DEFAULT_TAIL = ("w", "-", "-", "0", "1")


def square(rank: int, file: int) -> str:
    """板坐标 → ICCS 方格名，如 `square(2, 7) == "h2"`。"""
    return f"{chr(ord('a') + file)}{rank}"


def parse_square(sq: str) -> tuple[int, int]:
    """ICCS 方格名 → 板坐标 `(rank, file)`，如 `parse_square("h2") == (2, 7)`。"""
    if len(sq) < 2 or not sq[1:].isdigit():
        raise ValueError(f"非法方格名：{sq!r}")
    file = ord(sq[0]) - ord("a")
    rank = int(sq[1:])
    if not (0 <= file < _BOARD_COLS and 0 <= rank < _BOARD_ROWS):
        raise ValueError(f"方格名越界：{sq!r}")
    return rank, file


def _compress_row(values: list[int]) -> str:
    """把一行棋子编码压成 FEN 行（连续空点写成数字）。"""
    out: list[str] = []
    empty_run = 0
    for value in values:
        if value == 0:
            empty_run += 1
            continue
        if empty_run:
            out.append(str(empty_run))
            empty_run = 0
        out.append(PIECE_CHARS[value])
    if empty_run:
        out.append(str(empty_run))
    return "".join(out) or str(_BOARD_COLS)


@dataclass
class Board:
    """棋盘局面。

    `grid` 形状 `(10, 9)`、dtype `int8`，索引 `[rank, file]`；
    `side` 是**轮到走子的一方**（`RED` 或 `BLACK`）。
    """

    grid: np.ndarray
    side: int = RED

    # ---- 构造 / 转换 ----

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """解析 XQ-FEN；任何非法输入都抛 `ValueError`。"""
        fields = fen.split()
        if not fields:
            raise ValueError(f"非法 FEN：内容为空：{fen!r}")
        while len(fields) < 6:
            fields.append(_DEFAULT_TAIL[len(fields) - 1])

        setup, side_field = fields[0], fields[1]

        rows = setup.split("/")
        if len(rows) != _BOARD_ROWS:
            raise ValueError(f"非法 FEN：棋盘行数/列数不对：{fen!r}")

        grid = np.zeros((_BOARD_ROWS, _BOARD_COLS), dtype=np.int8)
        for i, row in enumerate(rows):
            rank = _BOARD_ROWS - 1 - i
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                    continue
                if ch in _EMPTY_CHARS:
                    file += 1
                    continue
                if ch not in CHAR_VALUES:
                    raise ValueError(f"非法 FEN：未知棋子字符 {ch!r}：{fen!r}")
                if file >= _BOARD_COLS:
                    raise ValueError(f"非法 FEN：棋盘行数/列数不对：{fen!r}")
                grid[rank, file] = CHAR_VALUES[ch]
                file += 1
            if file != _BOARD_COLS:
                raise ValueError(f"非法 FEN：棋盘行数/列数不对：{fen!r}")

        if side_field == "w":
            side = RED
        elif side_field == "b":
            side = BLACK
        else:
            raise ValueError(f"非法 FEN：走子方只能是 'w' 或 'b'，得到 {side_field!r}")

        for value, label in ((K, "红帅"), (-K, "黑将")):
            count = int(np.count_nonzero(grid == value))
            if count != 1:
                raise ValueError(f"非法 FEN：{label}必须恰好一个，实际 {count} 个：{fen!r}")

        return cls(grid=grid, side=side)

    def to_fen(self) -> str:
        """序列化为 XQ-FEN。"""
        rows = [_compress_row(self.grid[rank].tolist()) for rank in range(_BOARD_ROWS - 1, -1, -1)]
        side_field = "w" if self.side == RED else "b"
        return f"{'/'.join(rows)} {side_field} - - 0 1"

    def copy(self) -> Board:
        """深拷贝一份局面（`grid` 是独立数组）。"""
        return Board(grid=self.grid.copy(), side=self.side)

    # ---- 单点访问 ----

    def at(self, rank: int, file: int) -> int:
        """读取 `(rank, file)` 的棋子编码。"""
        return int(self.grid[rank, file])

    def set(self, rank: int, file: int, value: int) -> None:
        """写入 `(rank, file)` 的棋子编码。"""
        self.grid[rank, file] = value

    def pieces(self, side: int) -> list[tuple[int, int]]:
        """列出某一方所有棋子的 `(rank, file)`，按 rank 再 file 升序。"""
        ranks, files = np.nonzero(np.sign(self.grid.astype(np.int16)) == side)
        return sorted(zip(ranks.tolist(), files.tolist(), strict=True))
