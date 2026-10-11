"""逐格模板匹配识别测试。

识别层是「画面 → 棋盘」的唯一入口，一旦它给出的局面是错的，后面引擎算得再对也没用。
所以这里的用例全部钉在**真机帧**上，而不是合成图。

依赖 `config/templates.npz`（由 `uv run python tools/calibrate.py --build` 生成，不入库）。
它不存在时整个模块 skip —— 与 `tests/test_uci.py` 对引擎的处理方式一致。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess import geometry as geo
from jjchess.geometry import BoardGeometry, ScreenMapper
from jjchess.rules.board import Board
from jjchess.vision.locator import locate
from jjchess.vision.recognizer import Recognition, recognize
from jjchess.vision.templates import DEFAULT_TEMPLATES_PATH, TemplateSet

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OPENING_PNG = FIXTURES / "opening.png"
OPENING_FEN = FIXTURES / "opening.fen"

pytestmark = pytest.mark.skipif(
    not (OPENING_PNG.is_file() and OPENING_FEN.is_file() and DEFAULT_TEMPLATES_PATH.is_file()),
    reason="缺少真机 fixture 或 config/templates.npz（先跑 tools/calibrate.py --build）",
)

# 中局帧由 Task 10 的后续补交（要靠真人对局产生，合成图对识别没有价值）
MID_CASES = ["mid_01", "mid_02", "mid_03"]


def _load_rgb(path: Path) -> np.ndarray:
    """`cv2.imread` 在非 ASCII 路径上会静默返回 None，一律走 fromfile。"""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"读不出图片：{path}"
    return img


@pytest.fixture(scope="module")
def opening() -> tuple[np.ndarray, BoardGeometry, TemplateSet, Board]:
    img = _load_rgb(OPENING_PNG)
    geom = locate(img)
    templates = TemplateSet.load(DEFAULT_TEMPLATES_PATH)
    expect = Board.from_fen(OPENING_FEN.read_text(encoding="utf-8").strip())
    return img, geom, templates, expect


def test_开局32子全对(opening):
    img, geom, templates, expect = opening
    rec = recognize(img, geom, templates, ScreenMapper(geo.RED))

    assert rec.board.grid.tolist() == expect.grid.tolist()
    assert rec.unknown == []
    assert rec.confidence.shape == (10, 9)
    assert rec.confidence.dtype == np.float32
    assert rec.confidence.min() > 0.75  # [RF-2] 没有一格含糊


def test_置信度是屏幕格坐标(opening):
    """`confidence` 与 `unknown` 用屏幕 (row, col)，`board` 用板坐标 —— 这两套
    坐标系混起来是本节最容易踩的坑，所以显式钉一条。

    开局时黑方在屏幕上方：屏幕 row=0 是黑方底线 → 板坐标 rank=9。
    """
    img, geom, templates, _ = opening
    rec = recognize(img, geom, templates, ScreenMapper(geo.RED))

    assert rec.board.at(9, 0) == -5  # 屏幕左上角那个黑车
    assert rec.confidence[0, 0] > 0.75  # 同一个格子在 confidence 里是 [0, 0]


def _mirror_board_cells(img: np.ndarray, geom: BoardGeometry) -> np.ndarray:
    """模拟执黑时看到的画面：棋盘 180° 翻转，棋子字形**仍然正着**。

    游戏里棋子上的字永远是正的 —— 红方视角下黑方的「車馬象士將」是正的，
    所以执黑翻转棋盘时红方的「俥傌相仕帥」同样保持正立，只是位置换到了上方。
    翻转只挪位置、不转字形，因此这里**逐格搬运**而不是 `cv2.rotate`：
    直接旋转整张位图会得到一堆倒过来的字，那是现实里不存在的画面。

    棋盘线不随翻转变化，所以几何可以沿用原图的 `geom`。
    """
    out = img.copy()
    half = round(geom.cell_px * 0.55)
    for row in range(10):
        for col in range(9):
            sx, sy = geom.cell_to_px(row, col)
            mx, my = geom.cell_to_px(9 - row, 8 - col)
            sx, sy, mx, my = round(sx), round(sy), round(mx), round(my)
            out[my - half : my + half, mx - half : mx + half] = img[
                sy - half : sy + half, sx - half : sx + half
            ]
    return out


@pytest.mark.parametrize("name", MID_CASES)
def test_中局识别正确(name):
    png, fen = FIXTURES / f"{name}.png", FIXTURES / f"{name}.fen"
    if not (png.is_file() and fen.is_file()):
        pytest.skip(f"缺少中局 fixture {name}.png/.fen（Task 10 待补）")

    img = _load_rgb(png)
    rec = recognize(
        img, locate(img), TemplateSet.load(DEFAULT_TEMPLATES_PATH), ScreenMapper(geo.RED)
    )
    assert (
        rec.board.grid.tolist()
        == Board.from_fen(fen.read_text(encoding="utf-8").strip()).grid.tolist()
    )


def test_选中高亮帧仍能识别(opening):
    """走子前点一下棋子会画一圈高亮，高亮会盖住棋子边缘 —— 识别不能因此崩掉。"""
    img, geom, templates, expect = opening
    hi = img.copy()
    x, y = geom.cell_to_px(0, 1)
    cv2.circle(hi, (round(x), round(y)), round(geom.cell_px * 0.6), (0, 215, 255), 8)

    rec = recognize(hi, geom, templates, ScreenMapper(geo.RED))
    assert rec.board.grid.tolist() == expect.grid.tolist()


def test_执黑视角识别(opening):
    """执黑时棋盘 180° 翻转，`ScreenMapper(BLACK)` 负责把屏幕格映回板坐标。

    这条用例是为了钉死映射方向：`to_board` 写成 `(row, 8-col)` 还是 `(9-row, col)`
    在棋盘对称的局面下看不出来，而开局局面是对称的 —— 所以必须靠**整盘**比对，
    只有映射方向正确才能重建出和红方视角完全一样的 `START_FEN`。

    ⚠️ 这是按「翻转只挪位置、不转字形」建模的（依据：红方视角下黑方棋子就是正立的）。
    真机执黑帧到手后应换成真帧再跑一遍。
    """
    img, geom, templates, expect = opening
    black_view = _mirror_board_cells(img, geom)

    rec = recognize(black_view, geom, templates, ScreenMapper(geo.BLACK))
    assert rec.board.grid.tolist() == expect.grid.tolist()
    assert rec.unknown == []


def test_低置信度进unknown(opening):
    """认不出来的格子必须报 unknown 并置空，绝不能瞎猜一个棋子填上去。

    这里用高频噪声盖住一个空点：它不是空（拉普拉斯方差远高于阈值），
    又跟 15 类模板都不像 → 落到 unknown 分支。

    注：计划原本写的是「盖一块纯色矩形」。纯色块的拉普拉斯方差接近 0，会被
    空点分支以接近 1.0 的置信度判成空 —— 那是 Task 12 改用拉普拉斯判空之后的
    正确行为，触发不了 unknown。所以要改成噪声。
    """
    img, geom, templates, _ = opening
    noise = img.copy()
    x, y = geom.cell_to_px(5, 4)
    x, y = round(x), round(y)
    rng = np.random.default_rng(20261011)
    half = round(geom.cell_px * 0.5)
    noise[y - half : y + half, x - half : x + half] = rng.integers(
        0, 256, (2 * half, 2 * half, 3), dtype=np.uint8
    )

    rec = recognize(noise, geom, templates, ScreenMapper(geo.RED))
    assert (5, 4) in rec.unknown
    assert rec.confidence[5, 4] < 0.6


def test_返回类型与不变式():
    """`Recognition` 的字段契约 —— 面板与 session 都直接读这些字段。"""
    rec = Recognition(
        board=Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"),
        confidence=np.zeros((10, 9), np.float32),
        unknown=[],
    )
    assert rec.board.at(9, 4) == -1  # FEN 第一行是最上面一行 = rank 9
    assert rec.board.at(0, 4) == 1
    assert rec.board.side == geo.RED
    assert rec.confident is True
