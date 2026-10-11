"""棋盘几何：客户区像素 ↔ 棋盘格互转，以及"屏幕格 ↔ 板坐标"的执色映射。

两个坐标系要分清（spec §5.3）：

- **屏幕格 `(row, col)`**：棋盘格按**屏幕方向**编号，`row 0` 是最上面那条横线、
  `col 0` 是最左边那条纵线。它随执红/执黑翻转。
- **板坐标 `(rank, file)`**：`rank 0` 永远是红方底线、`file 0` 永远是 a 纵线。
  与屏幕方向无关，`rules/` 只认这一套。

`BoardGeometry` 工作在**客户区坐标**（窗口左上角为原点）；要拿屏幕绝对坐标
（给鼠标用）先 `with_offset(*client_origin_on_screen)`。
"""

from __future__ import annotations

from dataclasses import dataclass

from jjchess.rules.board import BLACK, RED

_ROWS = 10
_COLS = 9

#: `is_plausible` 允许的 dx/dy 相对偏差。
_MAX_ASPECT_SKEW = 0.2
#: `is_plausible` 允许的"棋子直径 / 格距"区间。
_CELL_PX_RATIO_RANGE = (0.5, 1.0)


@dataclass(frozen=True)
class BoardGeometry:
    """棋盘在客户区里的位置。

    `origin` 是**客户区坐标**下 `(row=0, col=0)` 那个交叉点的中心，轴序为
    `(x, y)`；`dx` / `dy` 是相邻纵线 / 横线的像素间距；`cell_px` 是棋子直径
    （识别用的 ROI 边长由它推出来）。
    """

    origin: tuple[float, float]
    dx: float
    dy: float
    cell_px: float

    # ---- 坐标换算 ----

    def cell_to_px(self, row: int, col: int) -> tuple[float, float]:
        """屏幕格 `(row, col)` → 该交叉点中心的客户区像素坐标 `(x, y)`。"""
        if not (0 <= row < _ROWS and 0 <= col < _COLS):
            raise ValueError(f"屏幕格越界：(row={row}, col={col})")
        return (self.origin[0] + col * self.dx, self.origin[1] + row * self.dy)

    def px_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """客户区像素坐标 `(x, y)` → 最近的屏幕格 `(row, col)`。"""
        row = round((y - self.origin[1]) / self.dy)
        col = round((x - self.origin[0]) / self.dx)
        if not (0 <= row < _ROWS and 0 <= col < _COLS):
            raise ValueError(f"像素坐标落在棋盘外：(x={x}, y={y})")
        return (row, col)

    def with_offset(self, ox: float, oy: float) -> BoardGeometry:
        """客户区坐标 → 屏幕绝对坐标（`ox` / `oy` 是客户区原点在屏幕上的位置）。"""
        return BoardGeometry(
            origin=(self.origin[0] + ox, self.origin[1] + oy),
            dx=self.dx,
            dy=self.dy,
            cell_px=self.cell_px,
        )

    # ---- 自检 ----

    def is_plausible(self) -> bool:
        """几何参数是否像一块真的棋盘（用来判定自动定位是否成功）。"""
        if self.dx <= 5 or self.dy <= 5 or self.cell_px <= 0:
            return False
        if abs(self.dx - self.dy) / max(self.dx, self.dy) >= _MAX_ASPECT_SKEW:
            return False
        low, high = _CELL_PX_RATIO_RANGE
        return low <= self.cell_px / min(self.dx, self.dy) <= high

    # ---- 序列化 ----

    def to_json(self) -> dict:
        """转成可写进 `config/calibration.json` 的普通 dict。"""
        return {
            "origin": [self.origin[0], self.origin[1]],
            "dx": self.dx,
            "dy": self.dy,
            "cell_px": self.cell_px,
        }

    @classmethod
    def from_json(cls, d: dict) -> BoardGeometry:
        """从 `to_json()` 的结果还原（`origin` 还原成 tuple 才能相等比较）。"""
        ox, oy = d["origin"]
        return cls(
            origin=(float(ox), float(oy)),
            dx=float(d["dx"]),
            dy=float(d["dy"]),
            cell_px=float(d["cell_px"]),
        )


class ScreenMapper:
    """屏幕格 ↔ 板坐标的双向映射，由"我执哪一方"决定朝向。

    执红时棋盘正着放：`rank = 9 - row, file = col`。
    执黑时小程序把棋盘 180° 翻转：`rank = row, file = 8 - col`。
    """

    def __init__(self, my_side: int) -> None:
        if my_side not in (RED, BLACK):
            raise ValueError(f"my_side 只能是 RED({RED}) 或 BLACK({BLACK})，得到 {my_side!r}")
        self.my_side = my_side

    def to_board(self, row: int, col: int) -> tuple[int, int]:
        """屏幕格 → 板坐标 `(rank, file)`。"""
        if not (0 <= row < _ROWS and 0 <= col < _COLS):
            raise ValueError(f"屏幕格越界：(row={row}, col={col})")
        if self.my_side == RED:
            return (9 - row, col)
        return (row, 8 - col)

    def to_screen(self, rank: int, file: int) -> tuple[int, int]:
        """板坐标 `(rank, file)` → 屏幕格。"""
        if not (0 <= rank < _ROWS and 0 <= file < _COLS):
            raise ValueError(f"板坐标越界：(rank={rank}, file={file})")
        if self.my_side == RED:
            return (9 - rank, file)
        return (rank, 8 - file)
