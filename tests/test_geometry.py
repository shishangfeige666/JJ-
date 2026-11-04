"""棋盘几何（像素 ↔ 格互转、执色映射）测试。"""

import pytest

from jjchess.geometry import BoardGeometry, ScreenMapper
from jjchess.rules.board import BLACK, RED

#: 测试统一用一个"好看"的几何：左上角交叉点在客户区 (100, 60)，格距 80px，
#: 棋子直径 68px。数值本身不重要，够大够整即可。
GEOM = BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)


def test_互逆():
    g = GEOM
    for row in range(10):
        for col in range(9):
            assert g.px_to_cell(*g.cell_to_px(row, col)) == (row, col)


def test_越界拒绝():
    g = GEOM
    with pytest.raises(ValueError):
        g.cell_to_px(10, 0)
    with pytest.raises(ValueError):
        g.cell_to_px(0, 9)
    with pytest.raises(ValueError):
        g.px_to_cell(-1.0, 0.0)
    with pytest.raises(ValueError):
        g.px_to_cell(0.0, 60.0 + 9 * 80.0 + 40.0)


def test_with_offset():
    g = GEOM
    assert g.with_offset(-1440.0, -279.0).cell_to_px(0, 0) == (-1340.0, -219.0)


def test_json_往返():
    g = GEOM
    assert BoardGeometry.from_json(g.to_json()) == g


def test_is_plausible():
    assert GEOM.is_plausible()
    # dx 与 dy 差太多 -> 说明定位错了
    assert not BoardGeometry(origin=(0.0, 0.0), dx=80.0, dy=40.0, cell_px=68.0).is_plausible()
    # 格距太小 -> 不可能是棋盘
    assert not BoardGeometry(origin=(0.0, 0.0), dx=4.0, dy=4.0, cell_px=68.0).is_plausible()
    # 棋子比格距还大 -> 不合理
    assert not BoardGeometry(origin=(0.0, 0.0), dx=80.0, dy=80.0, cell_px=200.0).is_plausible()


def test_执红映射():
    m = ScreenMapper(RED)
    assert m.to_board(9, 0) == (0, 0)  # 屏幕左下角 = a0 = 红车
    assert m.to_board(0, 0) == (9, 0)  # 屏幕左上角 = a9
    assert m.to_screen(0, 0) == (9, 0)


def test_执黑映射_棋盘180翻转():
    m = ScreenMapper(BLACK)
    assert m.to_board(9, 0) == (9, 8)  # 黑方视角左下角 = a9
    assert m.to_board(0, 0) == (0, 8)  # 左上角 = i0
    for row in range(10):
        for col in range(9):
            assert m.to_screen(*m.to_board(row, col)) == (row, col)


def test_映射越界拒绝():
    with pytest.raises(ValueError):
        ScreenMapper(RED).to_board(10, 0)
    with pytest.raises(ValueError):
        ScreenMapper(BLACK).to_screen(0, 9)
    with pytest.raises(ValueError):
        ScreenMapper(0)
