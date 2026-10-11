"""模板集（15 类棋子的标准外观）的构建、存取与匹配。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from jjchess.geometry import ScreenMapper
from jjchess.rules.board import RED, Board
from jjchess.vision.locator import locate
from jjchess.vision.templates import (
    CLASS_NAMES,
    SEARCH_PX,
    TEMPLATE_PX,
    TemplateSet,
    build_templates,
    make_patch,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OPENING_PNG = FIXTURES / "opening.png"
OPENING_FEN = FIXTURES / "opening.fen"

pytestmark = pytest.mark.skipif(
    not OPENING_PNG.is_file(), reason="缺少真机 fixture tests/fixtures/opening.png"
)


def _load_opening() -> tuple[np.ndarray, Board]:
    raw = np.fromfile(str(OPENING_PNG), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    board = Board.from_fen(OPENING_FEN.read_text(encoding="utf-8").strip())
    return img, board


@pytest.fixture(scope="module")
def opening() -> tuple[np.ndarray, Board, object]:
    img, board = _load_opening()
    return img, board, locate(img)


def test_类名有15个且含empty() -> None:
    assert len(CLASS_NAMES) == 15
    assert CLASS_NAMES[0] == "."
    assert len(set(CLASS_NAMES)) == 15


def test_类名与棋盘字符表一致() -> None:
    """类名必须就是 `PIECE_CHARS` 的取值 —— 否则识别结果没法直接还原成棋盘。"""
    from jjchess.rules.board import PIECE_CHARS

    assert set(CLASS_NAMES) == set(PIECE_CHARS.values())
    # 红方在前、黑方在后，便于人工核对模板图。
    assert CLASS_NAMES.index("K") < CLASS_NAMES.index("k")


def test_patch尺寸与类型(opening) -> None:
    """`make_patch` 返回的是**匹配窗口**（比模板大一圈），匹配要在它里面滑动搜索。"""
    img, _, geom = opening
    patch = make_patch(img, geom, 9, 4)
    assert patch.shape == (SEARCH_PX, SEARCH_PX)
    assert SEARCH_PX > TEMPLATE_PX
    assert patch.dtype == np.uint8
    assert patch.ndim == 2


def test_同一块patch平移几个像素仍能匹配自己(opening) -> None:
    """回归：定位误差是常态，匹配必须对它免疫。

    实测两个同类棋子之间相关长度只有约 1 px（平移 2 px 相关掉到 0.18），
    所以"同尺寸直接比"会把同类棋子算成 0.06 —— 必须在窗口里滑动取最大。
    """
    img, board, geom = opening
    suite = build_templates(img, geom, board, ScreenMapper(RED))
    window = make_patch(img, geom, 9, 4)  # 红帅，模板由单样本构成，正中就在窗口中心
    exact, _ = suite.match(window)
    assert exact == "K"
    for shift in (3, -3, 6, -6):
        moved = cv2.warpAffine(
            window,
            np.float32([[1, 0, shift], [0, 1, shift]]),
            (SEARCH_PX, SEARCH_PX),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        name, conf = suite.match(moved)
        assert name == "K", f"平移 {shift} px 后判成了 {name}"
        assert conf > 0.75, f"平移 {shift} px 后置信度只剩 {conf:.3f}"


def test_从开局帧构建模板并全部匹配正确(opening) -> None:
    img, board, geom = opening
    mapper = ScreenMapper(RED)
    suite = build_templates(img, geom, board, mapper)
    assert suite.data.shape == (15, TEMPLATE_PX, TEMPLATE_PX)
    assert suite.data.dtype == np.uint8

    wrong = []
    for row in range(10):
        for col in range(9):
            rank, file = mapper.to_board(row, col)
            expected = _char_of(board, rank, file)
            got, conf = suite.match(make_patch(img, geom, row, col))
            # 实测开局 32 个棋子置信度最低 0.798、58 个空点最低 0.872。
            if got != expected or conf < 0.75:
                wrong.append((row, col, expected, got, round(conf, 3)))
    assert wrong == []


def test_模板集npz往返(opening, tmp_path) -> None:
    img, board, geom = opening
    suite = build_templates(img, geom, board, ScreenMapper(RED))
    path = tmp_path / "t.npz"
    suite.save(path)
    again = TemplateSet.load(path)
    assert again.names == suite.names
    assert np.array_equal(again.data, suite.data)
    # 往返之后匹配行为也要一致（npz 里的 dtype 别被改掉）。
    patch = make_patch(img, geom, 9, 4)
    assert again.match(patch) == suite.match(patch)


def test_空点被剪枝为empty(opening) -> None:
    """空点必须走「先比空模板」的快速通道，且置信度很高。"""
    img, board, geom = opening
    mapper = ScreenMapper(RED)
    suite = build_templates(img, geom, board, mapper)
    row, col = 5, 3
    assert _char_of(board, *mapper.to_board(row, col)) == "."
    name, conf = suite.match(make_patch(img, geom, row, col))
    assert name == "."
    assert conf > 0.9


def test_模板保存后目录自动创建(opening, tmp_path) -> None:
    img, board, geom = opening
    suite = build_templates(img, geom, board, ScreenMapper(RED))
    nested = tmp_path / "a" / "b" / "t.npz"
    suite.save(nested)
    assert nested.is_file()


def _char_of(board: Board, rank: int, file: int) -> str:
    from jjchess.rules.board import PIECE_CHARS

    return PIECE_CHARS[board.at(rank, file)]
