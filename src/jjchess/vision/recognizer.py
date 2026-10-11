"""逐格模板匹配识别：把一帧画面还原成棋盘局面。

这是「画面 → 棋盘」的唯一入口。它给出的局面一旦是错的，后面引擎算得再对也没用，
所以这里只做一件简单可预测的事：对 90 个交叉点各切一块窗口交给 `TemplateSet` 打分，
拿最高分的类名当结果，分数不够低就**承认自己不认识**而不是硬猜。

## 两套坐标系，别混

- `Recognition.board` 用**板坐标** `[rank, file]`（rank 0 = 红方底线），要靠 `mapper` 换算；
- `Recognition.confidence` 与 `Recognition.unknown` 用**屏幕格坐标** `(row, col)`
  （左上角 = `(0, 0)`），因为面板要拿它们在画面上画叠加层，直接用 `geom.cell_to_px(row, col)`
  就行，不必再翻一次。

执红时两者正好差一个上下翻转，肉眼很难发现写反了 —— `tests/test_vision.py` 里
`test_置信度是屏幕格坐标` 就是为这个坑钉的。

## 算错的三种表现

1. **把棋子认成另一个棋子** —— 最危险。开局帧实测 top1 与 top2 有 0.17 以上的间隔，
   只要间隔够大就不会发生；发生了一定是模板脏了（重跑 `tools/calibrate.py --build`）。
2. **把棋子认成空**（或反过来）—— 用拉普拉斯方差判空之后，两者相差 7 倍以上，不会串。
3. **画面不对却照常返回一个局面** —— 空白帧、大厅画面。这个由上游挡：
   `window.is_blank_frame` 与 `vision.locator.locate`（找不到棋盘就抛 `BoardNotFound`）。
   `recognize` 本身**不做**画面合法性判断，它假定传进来的几何是对的。

## 一个逃过 `locate` 的漏网之鱼

`main.py` 只会在 `geom is None` 时重新 `locate()`，棋盘一旦定位成功就一直复用那套几何。
可是**结算画面**（对局结束时弹出的「再来一局」）根本没法定位，`locate()` 会抛
`BoardNotFound` —— 而它压根不会被再调一次。于是拿旧几何去认结算画面：

    真机实测 `_probe/e2e/now.png`（结算画面）→ `9/9/9/9/9/9/9/9/9/9 w`，
    `unknown` 15 格、最低置信度 0.129

也就是一个「什么都不认识」的**空盘子**。这个空盘子会被当成合法局面喂给会话，
把「棋盘消失 → `GAME_OVER`」那条路堵死（`_note_board` 只在 `None` 时才记账）。
所以 `recognize` 之外还得有一个便宜的事后检查：`both_kings_present`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jjchess.config import DEFAULTS
from jjchess.geometry import BoardGeometry, ScreenMapper
from jjchess.rules.board import CHAR_VALUES, RED, Board
from jjchess.vision.templates import TemplateSet, make_patch

__all__ = ["Recognition", "both_kings_present", "recognize"]

BOARD_ROWS = 10
BOARD_COLS = 9

#: `CHAR_VALUES` 里红帅 / 黑将的取值。写成常量而不是散落的字面量，
#: 免得「1 是帅还是兵」这种问题每次都要回去翻 board.py。
KING_RED = CHAR_VALUES["K"]
KING_BLACK = CHAR_VALUES["k"]


@dataclass
class Recognition:
    """一帧画面的识别结果。

    - `board`：**板坐标**局面；认不出的格子已经置空。
    - `confidence`：`(10, 9) float32` 置信度矩阵，索引是**屏幕** `[row, col]`。
    - `unknown`：认不出的**屏幕** `(row, col)` 列表（按行优先顺序）。
      面板据此把那些格子标红，session 据此刻意放慢重抓的节奏。
    """

    board: Board
    confidence: np.ndarray
    unknown: list[tuple[int, int]] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        """整盘有没有认不出的格子。"""
        return not self.unknown


def recognize(
    img: np.ndarray,
    geom: BoardGeometry,
    templates: TemplateSet,
    mapper: ScreenMapper,
    side_to_move: int = RED,
    conf_min: float | None = None,
) -> Recognition:
    """把一帧画面识别成局面。

    `side_to_move` 填进返回局面的 `side` 字段 —— 识别本身看不出轮到谁走，
    这个信息得由 session 层的着法解释器来维护。

    置信度低于 `conf_min`（默认取 `config.DEFAULTS.conf_min`）的格子：
    棋盘上置 `0`（空），同时记进 `unknown`。**宁可当作空也不要猜一个棋子** ——
    猜错会让引擎基于一个虚构的局面思考，而当作空最多让后续的合法着法解释器
    发现少了一个子，从而抓下一帧重来。
    """
    if conf_min is None:
        conf_min = DEFAULTS.conf_min

    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    confidence = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    unknown: list[tuple[int, int]] = []

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            name, conf = templates.match(make_patch(img, geom, row, col))
            confidence[row, col] = conf
            if conf < conf_min:
                unknown.append((row, col))
                value = 0
            else:
                value = CHAR_VALUES[name]
            rank, file = mapper.to_board(row, col)
            grid[rank, file] = value

    return Recognition(
        board=Board(grid=grid, side=side_to_move), confidence=confidence, unknown=unknown
    )


def both_kings_present(board: Board) -> bool:
    """双方将帅都在棋盘上 —— 这才是「一局棋」，而不是大厅、结算画面或认崩的一帧。

    象棋里将帅**永远不会被吃掉**（被将死时对局当场结束），所以「两个将帅都在」
    是一局进行中的棋必须成立的不变量。反过来说，一盘识别结果里少任何一个将帅，
    它就不可能是真实局面 —— 与其拿它去喂引擎，不如当作「这一帧没读到棋盘」。
    """
    values = board.grid
    return bool((values == KING_RED).any() and (values == KING_BLACK).any())
