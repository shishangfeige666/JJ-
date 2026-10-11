"""棋盘自动定位：在一帧画面里找出 9 条纵线与 10 条横线的位置。

思路（spec §4.3）是**投影峰值法**：棋盘线是深色细线，先用自适应阈值把它们抠成
白掩膜，再分别做纵向 / 横向投影 —— 每一条纵线会在自己的那一列上攒出一个尖峰，
横线同理。于是"找棋盘"退化成"在一维信号里找等距尖峰"。

真正麻烦的不是找峰，而是**判断峰是不是真的**：

- 大厅菜单、聊天列表这类画面里也有大量竖向纹理，随便凑 9 个峰出来会得到一组
  看起来很合理、其实毫无意义的"棋盘"（[RF-3]）。所以峰数不足、或峰的位置拟不上
  等距栅格时，一律 `BoardNotFound`，绝不硬凑。
- 棋盘上还压着 32 个棋子，它们的轮廓也会在投影里冒峰，甚至可能比真线还高。
  因此**不能简单地"取最高的 N 个"**，要用等距栅格去筛（见 `find_lines`）。

定位失败时的兜底是 `tools/calibrate.py` 手工点两个交叉点。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import cv2
import numpy as np

from jjchess.geometry import BoardGeometry

BOARD_ROWS = 10
BOARD_COLS = 9

#: 只看中间这一条带 —— 顶部是双方头像与计时，底部是功能按钮。
ROI_TOP, ROI_BOTTOM = 0.15, 0.80
ROI_LEFT, ROI_RIGHT = 0.01, 0.99

#: 峰值门槛：低于最高峰这个比例的局部极大一律不算候选。
_PEAK_RATIO = 0.3
#: 候选峰与栅格点的最大容许偏差（按格距的比例算）。
#:
#: 这个值必须**小**。放到 0.35 这种量级时，一个栅格点能"够到"半个格子外的假峰，
#: 于是几乎任何格距都能凑出满覆盖 —— 实测开局帧就被拟合成了 spacing=160.7
#: （真值 154.5），看起来同样合理，但 `dx` 直接偏掉 4%。真棋盘线本身很均匀
#: （实测纵线 8 个间距全是 154.5），给 10% 的余量已经很宽松。
_TOL_RATIO = 0.10
#: 平滑投影的 sigma。
_SMOOTH_SIGMA = 1.5
#: 等距重拟合的最多轮数。
_MAX_REFIT = 3
#: 棋盘至少要占投影长度这个比例 —— 用来把"格距估得太小"的假解挡掉。
_MIN_SPAN_RATIO = 0.35
#: 候选格距去重时的取整精度（px）。
_SPACING_ROUND = 0.5
#: 匹配到的峰强度中位数至少要达到峰值的这个比例，否则认为这些线不是贯穿棋盘的网格线。
_MIN_FILL_RATIO = 0.45
#: 棋子直径 / 格距。
_CELL_PX_RATIO = 0.85

_HINT = "请把棋盘完整显示出来再试；若仍失败，运行 `uv run python tools/calibrate.py` 手工标定"


class BoardNotFound(RuntimeError):
    """画面里找不到棋盘。"""


def _peaks(profile: np.ndarray, ratio: float) -> list[tuple[float, float]]:
    """一维投影找局部极大，返回按位置升序的 `(位置, 强度)`。

    位置用抛物线插值细化到亚像素 —— 整数下标会让 `dx` / `dy` 带上系统性偏差。
    """
    smooth = cv2.GaussianBlur(
        profile.astype(np.float32).reshape(1, -1), (0, 0), sigmaX=_SMOOTH_SIGMA
    ).ravel()
    top = float(smooth.max())
    if top <= 0:
        return []
    floor = top * ratio

    out: list[tuple[float, float]] = []
    for i in range(1, len(smooth) - 1):
        v = float(smooth[i])
        if v < floor or v < smooth[i - 1] or v < smooth[i + 1]:
            continue
        left, right = float(smooth[i - 1]), float(smooth[i + 1])
        denom = left - 2.0 * v + right
        offset = 0.0 if denom == 0 else max(-0.5, min(0.5, 0.5 * (left - right) / denom))
        out.append((i + offset, v))
    return out


@dataclass(frozen=True)
class _GridFit:
    """一组等距栅格的拟合结果，带上"这组线可信吗"的证据。"""

    origin: float
    spacing: float
    lines: list[float]
    #: 每个栅格点匹配到的候选峰强度；`0` 表示该栅格点没有任何峰支撑。
    matched: list[float]

    @property
    def covered(self) -> int:
        return sum(1 for s in self.matched if s > 0)


def _spacing_candidates(pos: np.ndarray, count: int, length: float) -> list[float]:
    """列出所有"有可能"的格距。

    棋盘上任意两条线的距离必然是格距的整数倍，所以候选格距就是**任意两个峰的距离
    除以 1..count-1**。这里刻意不取"相邻峰间距的中位数"——棋子的轮廓会在真线旁边
    挤出成串的假峰，把相邻间距彻底带偏（实测开局帧的横线间距中位数被带成 35px，
    真值是 148px）。
    """
    lo = _MIN_SPAN_RATIO * length / (count - 1)
    hi = length / (count - 1)
    out: set[float] = set()
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            gap = pos[j] - pos[i]
            for k in range(1, count):
                spacing = gap / k
                if lo <= spacing <= hi:
                    out.add(round(spacing / _SPACING_ROUND) * _SPACING_ROUND)
    return sorted(out)


def _fit_grid(cands: list[tuple[float, float]], count: int, length: float) -> _GridFit | None:
    """在候选峰里找一条等距栅格 `origin + i * spacing`（`i = 0..count-1`）。

    **不能简单地"取最强的 count 个峰"** —— 棋盘上压着 32 个棋子，它们的轮廓经常
    比真线还高。所以这里是"猜格距 + 拿每个峰轮流当原点试"，用*覆盖了几个栅格点*
    来打分；再由最小二乘把 origin / spacing 精修一遍。

    评分优先看覆盖数（9 条纵线就得 9 个点都有峰撑着），再取"匹配到的峰强度中位数"
    —— 真棋盘线的投影强度远高于噪声纹理，这一项能把"凑巧等距的假解"筛掉。
    """
    if len(cands) < count:
        return None
    pos = np.array([p for p, _ in cands], np.float64)
    strength = np.array([s for _, s in cands], np.float64)
    order = np.argsort(pos)
    pos, strength = pos[order], strength[order]
    n = len(pos)

    best: tuple[tuple[float, ...], float, float, np.ndarray] | None = None
    for spacing in _spacing_candidates(pos, count, length):
        tol = _TOL_RATIO * spacing
        # d[j, i] = 第 i 个峰相对"以第 j 个峰为原点"的偏移
        d = pos[None, :] - pos[:, None]
        step = np.rint(d / spacing).astype(np.int64)
        valid = (np.abs(d - step * spacing) <= tol) & (step >= 0) & (step < count)
        oj, ci = np.nonzero(valid)
        if oj.size < count:
            continue
        # ci 是"峰的编号"，peak 的列却是"栅格点编号"，别混。
        peak = np.zeros((n, count), np.float64)
        np.maximum.at(peak, (oj, step[oj, ci]), strength[ci])
        covered = (peak > 0).sum(axis=1)
        if covered.max() < count:
            continue
        median = np.median(np.where(peak > 0, peak, 0.0), axis=1)
        total = peak.sum(axis=1)
        for j in np.nonzero(covered == count)[0]:
            score = (float(covered[j]), float(median[j]), float(total[j]), spacing)
            if best is None or score > best[0]:
                best = (score, float(pos[j]), spacing, peak[j].copy())

    if best is None:
        return None
    _, origin, spacing, matched = best

    # 最小二乘精修：把候选峰分派到最近的栅格点，线性回归出更准的 origin / spacing。
    #
    # 注意一个坑：**同一个栅格点常常有好几个峰凑上来**（真线 + 紧挨着它的棋子轮廓）。
    # 若把这些峰一股脑丢进 polyfit，弱峰会把该栅格点的等效位置拽偏 —— 实测横线
    # origin 因此从 230 被拽到 222（8px，半个格子的 5%）。所以每个栅格点只认
    # **最强的那个峰**，其余交给相邻栅格点。
    grid = origin + np.arange(count) * spacing
    tol = _TOL_RATIO * spacing
    for _ in range(_MAX_REFIT):
        dist = np.abs(pos[:, None] - grid[None, :])
        nearest = dist.argmin(axis=1)
        keep = dist[np.arange(n), nearest] <= tol
        idx: list[float] = []
        val: list[float] = []
        for i in np.unique(nearest[keep]):
            members = np.nonzero(keep & (nearest == i))[0]
            winner = members[int(np.argmax(strength[members]))]
            idx.append(float(i))
            val.append(float(pos[winner]))
        if len(idx) < 2:
            break
        slope, intercept = np.polyfit(np.array(idx), np.array(val), 1)
        if slope <= 1.0:
            break
        refined = intercept + np.arange(count) * slope
        if np.allclose(refined, grid, atol=0.05):
            break
        grid, spacing, origin = refined, float(slope), float(intercept)

    return _GridFit(
        origin=origin,
        spacing=spacing,
        lines=grid.tolist(),
        matched=matched.tolist(),
    )


def find_lines(profile: np.ndarray, count: int) -> list[float]:
    """从一维投影里找出 `count` 条等距线的位置。

    峰数不足、或峰的位置拟不上等距栅格时抛 `BoardNotFound` —— 宁可判"没有棋盘"，
    也不返回一组凑出来的假线。
    """
    if count < 2:
        raise ValueError(f"count 至少为 2，得到 {count}")
    if profile is None or profile.size < count:
        raise BoardNotFound(f"投影长度不足以容纳 {count} 条线；{_HINT}")

    cands = _peaks(np.asarray(profile).ravel(), _PEAK_RATIO)
    if len(cands) < count:
        raise BoardNotFound(f"只找到 {len(cands)} 条候选线，不足以凑出 {count} 条棋盘线；{_HINT}")

    fitted = _fit_grid(cands, count, float(profile.size))
    if fitted is None:
        raise BoardNotFound(f"候选线拟不上 {count} 条等距栅格，画面里可能根本没有棋盘；{_HINT}")

    # 覆盖数够了不代表就是棋盘：大厅菜单的卡片、聊天列表也有一堆近似等距的边。
    # 真棋盘线是**贯穿整个 ROI 的长线**，投影强度远高于这些噪声纹理，用中位数卡一道。
    peak = max(strength for _, strength in cands)
    median = statistics.median([s for s in fitted.matched if s > 0])
    if median < _MIN_FILL_RATIO * peak:
        raise BoardNotFound(
            f"候选线的投影强度中位数只有 {median:.0f}（最强候选线 {peak:.0f}），"
            f"这些线强弱悬殊，不像是贯穿棋盘的网格线；{_HINT}"
        )

    return fitted.lines


def locate(img: np.ndarray) -> BoardGeometry:
    """在一帧客户区画面里定位棋盘，失败抛 `BoardNotFound`。"""
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise BoardNotFound(f"画面格式不对（期望 BGR 三通道），无法定位棋盘；{_HINT}")
    height, width = img.shape[:2]
    if height < 100 or width < 100:
        raise BoardNotFound(f"画面太小（{width}x{height}），无法定位棋盘；{_HINT}")

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    y0, y1 = int(ROI_TOP * height), int(ROI_BOTTOM * height)
    x0, x1 = int(ROI_LEFT * width), int(ROI_RIGHT * width)
    roi = gray[y0:y1, x0:x1]

    # 深色网格线在 THRESH_BINARY_INV 之后变成白（255）。
    mask = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    profile_x = mask.sum(axis=0) / 255.0
    profile_y = mask.sum(axis=1) / 255.0

    vlines = [x + x0 for x in find_lines(profile_x, BOARD_COLS)]
    hlines = [y + y0 for y in find_lines(profile_y, BOARD_ROWS)]

    dx = statistics.fmean(np.diff(vlines).tolist())
    dy = statistics.fmean(np.diff(hlines).tolist())
    geometry = BoardGeometry(
        origin=(vlines[0], hlines[0]),
        dx=dx,
        dy=dy,
        cell_px=_CELL_PX_RATIO * min(dx, dy),
    )
    if not geometry.is_plausible():
        raise BoardNotFound(f"定位出的棋盘几何不可信（{geometry}）；{_HINT}")
    return geometry
