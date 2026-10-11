"""着法生成、攻击判定与终局判定。

坐标与棋子编码见 `jjchess.rules.board`。本模块内部一律以 **嵌套 list**
（`g[rank][file]`）运算：`Board.grid` 是 numpy 数组，逐格下标访问开销大，
而 `perft(4)` 需要做数百万次攻击判定，所以公开函数进入时用一次
`grid.tolist()` 快照，之后全在纯 Python 容器上跑。
"""

from __future__ import annotations

from jjchess.rules.board import (
    BLACK,
    RED,
    A,
    Board,
    C,
    E,
    H,
    K,
    P,
    R,
    parse_square,
    square,
)

_ROWS = 10
_COLS = 9

#: 九宫的三条纵线。
_PALACE_FILES = (3, 4, 5)
#: 九宫的三条横线（红黑各一组）。
_PALACE_RANKS: dict[int, tuple[int, int, int]] = {RED: (0, 1, 2), BLACK: (7, 8, 9)}

#: 正交四方向 `(dr, df)`。
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
#: 马的八个落点 `(dr, df)`。
_HORSE = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
#: 士的四个斜角。
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
#: 象的四个田字落点。
_ELEPHANT = ((2, 2), (2, -2), (-2, 2), (-2, -2))


# ---------------------------------------------------------------- 基础工具


def _on_board(rank: int, file: int) -> bool:
    return 0 <= rank < _ROWS and 0 <= file < _COLS


def _in_palace(rank: int, file: int, side: int) -> bool:
    """`(rank, file)` 是否落在 `side` 的九宫内。"""
    return file in _PALACE_FILES and rank in _PALACE_RANKS[side]


def _on_own_side(rank: int, side: int) -> bool:
    """`rank` 是否还在 `side` 自己这半边（未过河）。"""
    return rank <= 4 if side == RED else rank >= 5


def _is_enemy(value: int, side: int) -> bool:
    return value != 0 and (value > 0) != (side > 0)


def _can_land(g: list[list[int]], rank: int, file: int, side: int) -> bool:
    """目标点在盘内且不是己方棋子。"""
    return _on_board(rank, file) and not (g[rank][file] != 0 and (g[rank][file] > 0) == (side > 0))


def parse_move(mv: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """ICCS 着法 → `(源点, 目标点)` 板坐标，如 `"h2e2" == ((2,7),(2,4))`。"""
    if len(mv) != 4:
        raise ValueError(f"非法着法：{mv!r}")
    return parse_square(mv[:2]), parse_square(mv[2:])


def format_move(src: tuple[int, int], dst: tuple[int, int]) -> str:
    """`(源点, 目标点)` 板坐标 → ICCS 着法字符串。"""
    return square(*src) + square(*dst)


# ---------------------------------------------------------------- 着法生成


def _gen_piece(
    g: list[list[int]], rank: int, file: int, kind: int, side: int, out: list[str]
) -> None:
    """生成单个棋子的伪合法着法，追加到 `out`。"""
    src = square(rank, file)

    if kind == K:
        for dr, df in _ORTHO:
            r, f = rank + dr, file + df
            if _in_palace(r, f, side) and _can_land(g, r, f, side):
                out.append(src + square(r, f))

    elif kind == A:
        for dr, df in _DIAG:
            r, f = rank + dr, file + df
            if _in_palace(r, f, side) and _can_land(g, r, f, side):
                out.append(src + square(r, f))

    elif kind == E:
        for dr, df in _ELEPHANT:
            r, f = rank + dr, file + df
            if not _on_board(r, f) or not _on_own_side(r, side):
                continue  # 象不过河
            if g[rank + dr // 2][file + df // 2] != 0:
                continue  # 塞象眼
            if _can_land(g, r, f, side):
                out.append(src + square(r, f))

    elif kind == H:
        for dr, df in _HORSE:
            r, f = rank + dr, file + df
            if not _on_board(r, f):
                continue
            # 蹩马腿：先走的那一步正交邻格必须为空。
            lr, lf = (rank + dr // 2, file) if abs(dr) == 2 else (rank, file + df // 2)
            if g[lr][lf] != 0:
                continue
            if _can_land(g, r, f, side):
                out.append(src + square(r, f))

    elif kind == R:
        for dr, df in _ORTHO:
            r, f = rank + dr, file + df
            while _on_board(r, f):
                if g[r][f] == 0:
                    out.append(src + square(r, f))
                else:
                    if _is_enemy(g[r][f], side):
                        out.append(src + square(r, f))
                    break
                r += dr
                f += df

    elif kind == C:
        for dr, df in _ORTHO:
            r, f = rank + dr, file + df
            # 第一段：未遇子之前只能平移，不能吃。
            while _on_board(r, f) and g[r][f] == 0:
                out.append(src + square(r, f))
                r += dr
                f += df
            if not _on_board(r, f):
                continue
            # 越过炮架（有且仅有一个）。
            r += dr
            f += df
            while _on_board(r, f) and g[r][f] == 0:
                r += dr
                f += df
            if _on_board(r, f) and _is_enemy(g[r][f], side):
                out.append(src + square(r, f))

    elif kind == P:
        dr = 1 if side == RED else -1
        r = rank + dr
        if _can_land(g, r, file, side):
            out.append(src + square(r, file))
        if not _on_own_side(rank, side):  # 已过河才能横走
            for df in (-1, 1):
                f = file + df
                if _can_land(g, rank, f, side):
                    out.append(src + square(rank, f))


def _generate(g: list[list[int]], side: int) -> list[str]:
    """按 `side` 生成全部**伪合法**着法（不检查是否自将）。"""
    out: list[str] = []
    for rank in range(_ROWS):
        row = g[rank]
        for file in range(_COLS):
            value = row[file]
            if value == 0 or (value > 0) != (side > 0):
                continue
            _gen_piece(g, rank, file, abs(value), side, out)
    return out


def generate(board: Board) -> list[str]:
    """按 `board.side` 生成全部**伪合法**着法，返回 ICCS 字符串列表。"""
    return _generate(board.grid.tolist(), board.side)


# ---------------------------------------------------------------- 攻击判定


def _find_king(g: list[list[int]], side: int) -> tuple[int, int] | None:
    """将/帅永远在九宫内，所以只扫九宫那 9 格。"""
    target = K if side == RED else -K
    for rank in _PALACE_RANKS[side]:
        row = g[rank]
        for file in _PALACE_FILES:
            if row[file] == target:
                return rank, file
    return None


def _attacked(g: list[list[int]], rank: int, file: int, by_side: int) -> bool:
    """`(rank, file)` 是否被 `by_side` 一方攻击。"""
    # 车 / 炮 / 将帅照面：四个正交方向各扫一遍。
    for dr, df in _ORTHO:
        r, f = rank + dr, file + df
        while _on_board(r, f) and g[r][f] == 0:
            r += dr
            f += df
        if not _on_board(r, f):
            continue
        first = g[r][f]
        if first == by_side * R:
            return True
        if first == by_side * K and df == 0:
            return True  # 同纵线、中间无子 -> 将帅照面
        # 越过炮架，再找第二个子：是敌方炮就是炮将军。
        r += dr
        f += df
        while _on_board(r, f) and g[r][f] == 0:
            r += dr
            f += df
        if _on_board(r, f) and g[r][f] == by_side * C:
            return True

    # 马：反过来从目标点找能跳到这里的敌马，再验它的马腿。
    for dr, df in _HORSE:
        hr, hf = rank + dr, file + df
        if not _on_board(hr, hf) or g[hr][hf] != by_side * H:
            continue
        lr, lf = (hr - dr // 2, hf) if abs(dr) == 2 else (hr, hf - df // 2)
        if g[lr][lf] == 0:
            return True

    # 兵 / 卒：正面一步，以及过河后的左右横走。
    fwd = 1 if by_side == RED else -1
    pr = rank - fwd
    if _on_board(pr, file) and g[pr][file] == by_side * P:
        return True
    if not _on_own_side(rank, by_side):
        for df in (-1, 1):
            f = file + df
            if _on_board(rank, f) and g[rank][f] == by_side * P:
                return True

    return False


def is_attacked(board: Board, rank: int, file: int, by_side: int) -> bool:
    """`(rank, file)` 是否被 `by_side` 一方攻击（含将帅照面）。"""
    return _attacked(board.grid.tolist(), rank, file, by_side)


def _in_check(g: list[list[int]], side: int) -> bool:
    pos = _find_king(g, side)
    return pos is not None and _attacked(g, pos[0], pos[1], -side)


def in_check(board: Board, side: int) -> bool:
    """`side` 一方当前是否被将军。"""
    return _in_check(board.grid.tolist(), side)


# ---------------------------------------------------------------- 合法着法


def _apply(g: list[list[int]], src: tuple[int, int], dst: tuple[int, int]) -> int:
    captured = g[dst[0]][dst[1]]
    g[dst[0]][dst[1]] = g[src[0]][src[1]]
    g[src[0]][src[1]] = 0
    return captured


def _undo(g: list[list[int]], src: tuple[int, int], dst: tuple[int, int], captured: int) -> None:
    g[src[0]][src[1]] = g[dst[0]][dst[1]]
    g[dst[0]][dst[1]] = captured


def make(board: Board, mv: str) -> int:
    """就地走一步并翻转走子方；返回被吃棋子编码（`0` 表示没吃到）。"""
    src, dst = parse_move(mv)
    grid = board.grid
    captured = int(grid[dst[0], dst[1]])
    grid[dst[0], dst[1]] = grid[src[0], src[1]]
    grid[src[0], src[1]] = 0
    board.side = -board.side
    return captured


def unmake(board: Board, mv: str, captured: int) -> None:
    """撤销 `make`：把 `captured` 放回目标点并翻转走子方。"""
    src, dst = parse_move(mv)
    grid = board.grid
    grid[src[0], src[1]] = grid[dst[0], dst[1]]
    grid[dst[0], dst[1]] = captured
    board.side = -board.side


def _legal_moves(g: list[list[int]], side: int) -> list[str]:
    out: list[str] = []
    for mv in _generate(g, side):
        src, dst = parse_move(mv)
        captured = _apply(g, src, dst)
        ok = not _in_check(g, side)
        _undo(g, src, dst, captured)
        if ok:
            out.append(mv)
    return out


def legal_moves(board: Board) -> list[str]:
    """`board.side` 的全部**合法**着法（已滤掉走完自己被将军的）。"""
    g = board.grid.tolist()
    return _legal_moves(g, board.side)


# ---------------------------------------------------------------- 终局判定


def game_status(board: Board) -> str:
    """判定当前局面：`ONGOING` / `CHECK` / `CHECKMATE` / `STALEMATE`。"""
    g = board.grid.tolist()
    side = board.side
    if _legal_moves(g, side):
        return "CHECK" if _in_check(g, side) else "ONGOING"
    return "CHECKMATE" if _in_check(g, side) else "STALEMATE"


def perft(board: Board, depth: int) -> int:
    """统计 `depth` 层内的合法着法序列数（走法生成正确性的金标准）。"""
    return _perft(board.grid.tolist(), board.side, depth)


def _perft(g: list[list[int]], side: int, depth: int) -> int:
    if depth <= 0:
        return 1
    moves = _legal_moves(g, side)
    if depth == 1:
        return len(moves)
    total = 0
    for mv in moves:
        src, dst = parse_move(mv)
        captured = _apply(g, src, dst)
        total += _perft(g, -side, depth - 1)
        _undo(g, src, dst, captured)
    return total
