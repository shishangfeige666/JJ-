"""中文纵线记谱（「炮二平五」）与 ICCS 着法的双向转换。

记谱规则（spec §4.4）：

- **纵线编号**：红方从自己的右手边数起、用汉字，`num = 9 - file`
  （`file 8` 是「一」，`file 0` 是「九」）；黑方同样从自己的右手边数起、
  用阿拉伯数字，`num = file + 1`。两者都等价于"从本方视角从左往右数"。
- **动作**：同纵线移动记「进 / 退」（红方 `rank` 增大为进，黑方 `rank`
  减小为进）；换纵线记「平」。
- **数值**：车/炮/兵/将走直线时是**移动的格数**；马/象/士走斜线时是
  **目标纵线号**；「平」一律是目标纵线号。
- **消歧**：同一纵线上有 2 个及以上同类同色子时，用「前 / 中 / 后」
  代替纵线号（「前」指更靠近对方底线的那一个）。
"""

from __future__ import annotations

from jjchess.rules.board import RED, A, Board, C, E, H, K, P, R
from jjchess.rules.moves import legal_moves, parse_move

#: 红方棋子名（`帅仕相马车炮兵`）。
PIECE_NAMES_RED: dict[int, str] = {K: "帅", A: "仕", E: "相", H: "马", R: "车", C: "炮", P: "兵"}
#: 黑方棋子名（`将士象马车炮卒`）。
PIECE_NAMES_BLACK: dict[int, str] = {K: "将", A: "士", E: "象", H: "马", R: "车", C: "炮", P: "卒"}
#: 红方纵线号用的汉字（黑方用阿拉伯数字，直接用 `str`）。
CN_DIGITS = "一二三四五六七八九"

#: 走斜线的兵种：它们的「进 / 退」数值取目标纵线号而非格数。
_DIAGONAL_KINDS = (A, E, H)


def _file_number(file: int, side: int) -> int:
    """板坐标 `file` → 该方视角的纵线号（1..9）。"""
    return 9 - file if side == RED else file + 1


def _number_text(number: int, side: int) -> str:
    """纵线号 / 格数 → 记谱用的数字文本（红方汉字、黑方阿拉伯数字）。"""
    return CN_DIGITS[number - 1] if side == RED else str(number)


def _file_marker(board: Board, rank: int, file: int, kind: int, side: int) -> tuple[str, str]:
    """返回 `(名前缀, 纵线号文本)`。

    同一纵线上只有这一个子时，前缀是空的、纵线号照常写在棋子名**后**面
    （「炮二平五」）；同线有 2 个及以上同类同色子时改用「前 / 中 / 后」，
    它写在棋子名**前**面且不再写纵线号（「前车进一」）。
    """
    same_file = [r for r in range(10) if int(board.grid[r, file]) == side * kind]
    if len(same_file) < 2:
        return "", _number_text(_file_number(file, side), side)

    # 「前」= 更靠近对方底线：红方 rank 大，黑方 rank 小。
    ordered = sorted(same_file, reverse=side == RED)
    index = ordered.index(rank)
    count = len(ordered)
    if count == 2:
        marker = "前" if index == 0 else "后"
    elif count == 3:
        marker = ("前", "中", "后")[index]
    # 四个及以上（罕见，只有同一纵线的兵卒会碰到）：两端用前后、中间用数字序号。
    elif index == 0:
        marker = "前"
    elif index == count - 1:
        marker = "后"
    else:
        marker = CN_DIGITS[index]
    return marker, ""


def to_chinese(board_before: Board, mv: str) -> str:
    """把 `board_before` 局面下的着法 `mv` 转成中文纵线记谱。

    `to_chinese(Board.from_fen(START_FEN), "h2e2") == "炮二平五"`
    """
    (rank0, file0), (rank1, file1) = parse_move(mv)
    side = board_before.side
    value = int(board_before.grid[rank0, file0])
    if value == 0:
        raise ValueError(f"起点没有棋子：{mv!r}")
    if (value > 0) != (side > 0):
        raise ValueError(f"走子方与棋子颜色不符：{mv!r}")

    kind = abs(value)
    name = (PIECE_NAMES_RED if side == RED else PIECE_NAMES_BLACK)[kind]
    marker, file_text = _file_marker(board_before, rank0, file0, kind, side)

    if kind in _DIAGONAL_KINDS:
        forward = rank1 > rank0 if side == RED else rank1 < rank0
        action = "进" if forward else "退"
        number = _file_number(file1, side)
    elif file0 == file1:
        step = rank1 - rank0
        forward = step > 0 if side == RED else step < 0
        action = "进" if forward else "退"
        number = abs(step)
    else:
        action = "平"
        number = _file_number(file1, side)

    return f"{marker}{name}{file_text}{action}{_number_text(number, side)}"


def from_chinese(board_before: Board, text: str) -> str | None:
    """中文纵线记谱 → ICCS 着法；歧义或非法时返回 `None`。

    实现方式是「枚举本方**合法**着法 → 各自 `to_chinese` 比对」，
    因此天然覆盖同线多子等所有歧义情形，也不会解析出自将的着法。
    """
    if not text:
        return None
    for mv in legal_moves(board_before):
        if to_chinese(board_before, mv) == text:
            return mv
    return None
