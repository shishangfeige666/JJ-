"""15 类棋子外观模板（14 种棋子 + 空点）的构建、存取与匹配。

识别走的是**模板匹配**而不是 CNN：象棋棋子的外观只有 14 种，同一类棋子在
不同格子上就是同一张素材图平移到交叉点，所以只要拿到一份"标准外观"就够用了。
这份标准外观就是本模块的模板集。

模板从**真机开局帧**构建（`tools/calibrate.py --build`）：开局局面 32 个子齐全，
14 个类全都至少出现一次，而且每个交叉点的正确答案都由 `opening.fen` 给定 ——
等于白送一份标注数据。同一类可能有多份样本（开局有 5 个兵），取**逐像素中位数**
当模板，单格噪声（高光、抗锯齿差异）会被压掉。

**匹配必须在搜索窗口里滑动，不能拿同尺寸的 patch 直接比。** 这是实测踩出来的：
棋子贴图是高频纹理，两个同类棋子的相关长度只有约 1 像素 —— 把同一块 patch 平移
0.5 px 相关就掉到 0.87，平移 2 px 掉到 0.18。而棋盘定位给出的格子中心至少有
±3 px 的误差（棋盘是带透视的梯形，横线间距从 144.5 px 递增到 151.5 px，用
"起点 + 均匀间距"表达必然在两端欠准），于是"同尺寸直接比"会把同一类棋子算出
0.06~0.14 的相关，彻底不可用。改成在 `SEARCH_SCALE` 的窗口里 `matchTemplate`
取最大值之后，开局 90 格全部判对。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from jjchess.geometry import BoardGeometry, ScreenMapper
from jjchess.rules.board import PIECE_CHARS, A, Board, C, E, H, K, P, R

#: 定义**模板外观**的裁剪比例：ROI 边长 = 该比例 × 格距。棋子直径约为 0.85 格，
#: 取 1.30 能让棋子占满模板、又不至于把相邻棋子切进来。
ROI_SCALE = 1.30
#: 匹配时实际裁剪的比例。多出来的 `(SEARCH_SCALE - ROI_SCALE) / 2 = 0.3` 格
#: 就是滑动搜索的余量（约 ±38 px），足以吸收定位误差。
SEARCH_SCALE = 1.90
#: 模板（以及模板集中每个类的外观）统一缩放到这个边长。
TEMPLATE_PX = 72
#: 匹配窗口的边长 —— 与模板保持同一物理比例，否则 `matchTemplate` 会当成不同尺度。
SEARCH_PX = round(TEMPLATE_PX * SEARCH_SCALE / ROI_SCALE)

#: 空点判据：单格中心区的**拉普拉斯方差**低于它就算空点。
#: 实测（开局帧 58 个空点 / 32 个棋子、CLAHE 之后）空点 250~1149、棋子 8809~18627，
#: 取几何中点 3000，两侧各留 2.6 倍余量。**不能用"跟空模板比相关"** —— 空-空相关
#: 本身只有 0.65，那个机制在真机画面上不成立。
EMPTY_LAPLACIAN = 3000.0
#: 算拉普拉斯方差时取窗口正中心多大比例（固定物理大小，与裁剪比例无关）。
_CORE_SCALE = 0.72
#: 中心区在匹配窗口里的边长。
_CORE_PX = max(9, round(TEMPLATE_PX * (ROI_SCALE / SEARCH_SCALE) * _CORE_SCALE))
#: 建模板时把同类样本互相平移对齐的搜索半径（模板像素）。
_ALIGN_RADIUS = 8
#: CLAHE 的对比度限制与分块 —— 抹掉棋盘上从左到右的渐变阴影，让同类棋子可比。
_CLAHE_CLIP = 2.0
_CLAHE_TILES = (8, 8)

#: 模板集默认落盘位置（`config/` 已在 .gitignore 里，这是**生成物**不是源码）。
DEFAULT_TEMPLATES_PATH = Path("config/templates.npz")

#: 15 个类名，**取值就是 `PIECE_CHARS` 的取值**（XQ-FEN 规范字母，相= B、马= N），
#: 这样 `TemplateSet.match()` 的结果可以直接经 `CHAR_VALUES` 还原成棋盘上的数。
#: 顺序固定为「空 → 红 7 子 → 黑 7 子」，便于人工看模板图时对照。
CLASS_NAMES: list[str] = [PIECE_CHARS[0]] + [
    PIECE_CHARS[value] for value in (K, A, E, H, R, C, P, -K, -A, -E, -H, -R, -C, -P)
]

_EMPTY = CLASS_NAMES[0]


def _crop_replicate(img: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """裁一块 `side x side` 的正方形；越界部分用边缘像素铺满。

    棋子贴近画面边缘时（棋盘最外圈）ROI 会探出去，直接切片会拿到一小块不规则的
    区域，缩放到模板尺寸之后和模板完全对不上。复制边缘像素的代价最小。
    """
    height, width = img.shape[:2]
    x1, y1 = x0 + side, y0 + side
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    patch = img[sy0:sy1, sx0:sx1]
    top, bottom = sy0 - y0, y1 - sy1
    left, right = sx0 - x0, x1 - sx1
    if top or bottom or left or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return patch


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILES).apply(gray)


def _cell_patch(
    img: np.ndarray, geom: BoardGeometry, row: int, col: int, scale: float, px: int
) -> np.ndarray:
    """按 `scale × 格距` 裁出某个交叉点周围的方块，灰度化 + CLAHE，缩放到 `px`。"""
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("画面必须是 BGR 三通道")
    center_x, center_y = geom.cell_to_px(row, col)
    side = max(8, round(scale * geom.cell_px))
    patch = _crop_replicate(img, round(center_x - side / 2), round(center_y - side / 2), side)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return _clahe(cv2.resize(gray, (px, px), interpolation=cv2.INTER_AREA))


def make_patch(img: np.ndarray, geom: BoardGeometry, row: int, col: int) -> np.ndarray:
    """取某个交叉点的**匹配窗口**：`(SEARCH_PX, SEARCH_PX) uint8` 灰度图（已 CLAHE）。

    它比模板大一圈，`TemplateSet.match()` 会在里面滑动搜索，从而对定位误差免疫。
    """
    return _cell_patch(img, geom, row, col, SEARCH_SCALE, SEARCH_PX)


def reference_patch(img: np.ndarray, geom: BoardGeometry, row: int, col: int) -> np.ndarray:
    """取某个交叉点的**模板外观**：`(TEMPLATE_PX, TEMPLATE_PX) uint8`，用于建模板集。"""
    return _cell_patch(img, geom, row, col, ROI_SCALE, TEMPLATE_PX)


def laplacian_variance(patch: np.ndarray) -> float:
    """匹配窗口正中心区的拉普拉斯方差 —— 空点与棋子的判据（空点≈10，棋子≈2000）。"""
    side = min(_CORE_PX, patch.shape[0], patch.shape[1])
    y0 = (patch.shape[0] - side) // 2
    x0 = (patch.shape[1] - side) // 2
    core = patch[y0 : y0 + side, x0 : x0 + side]
    return float(cv2.Laplacian(core, cv2.CV_64F).var())


def _peak_score(window: np.ndarray, template: np.ndarray) -> float:
    """模板在窗口里滑动，取最高的归一化相关。`TM_CCOEFF_NORMED` 先减均值，
    所以对整体亮度差异不敏感（实测亮度整体缩放 1.3 倍后相关仍有 0.998）。"""
    if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
        raise ValueError(
            f"匹配窗口 {window.shape[:2]} 比模板 {template.shape[:2]} 还小；"
            f"请用 make_patch() 取窗口，而不是 reference_patch()"
        )
    score = float(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED).max())
    return score if np.isfinite(score) else -1.0


def _fallback_template(done: dict[str, np.ndarray], value: int) -> np.ndarray | None:
    """某一类在开局帧里没出现时的兜底：拿同色系已有模板的平均值凑一个占位。"""
    same_side = [
        v for k, v in done.items() if (ord(k[0]) < ord("a")) == (value > 0) and k != _EMPTY
    ]
    return np.mean(same_side, axis=0).astype(np.uint8) if same_side else None


@dataclass(eq=False)
class TemplateSet:
    """15 类标准外观。`data[i]` 是 `names[i]` 的 `(TEMPLATE_PX, TEMPLATE_PX)` 灰度模板。

    `eq=False`：`data` 是 ndarray，自动生成的 `__eq__` 会返回数组而不是布尔值，
    在 `assert` 或 `in` 里会直接抛异常。
    """

    names: list[str]
    data: np.ndarray

    def __post_init__(self) -> None:
        if len(self.names) != len(self.data):
            raise ValueError(f"names 有 {len(self.names)} 项而 data 有 {len(self.data)} 项")
        if self.data.dtype != np.uint8:
            raise ValueError(f"data 必须是 uint8，实际是 {self.data.dtype}")

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"模板集里没有类 {name!r}") from exc

    def template(self, name: str) -> np.ndarray:
        return self.data[self.index(name)]

    def match(self, window: np.ndarray) -> tuple[str, float]:
        """在匹配窗口里找最像的一类，返回 `(类名, 置信度)`。

        置信度口径（两个分支数值含义不同，但都"越高越可信"）：
        - 空点：`1 / (1 + (拉普拉斯方差/门槛)²)`，实测最差的空点也有 0.87；
        - 棋子：`matchTemplate` 的最高归一化相关，实测开局 32 个棋子最低 0.798。

        两者都落在 `conf_good` 之上、离 `conf_min` 有 0.2 的缓冲 —— 也就是说
        **判错时分数会掉到 `conf_min` 以下被标成 unknown**，而不是悄悄给个错答案。
        """
        lap = laplacian_variance(window)
        if lap < EMPTY_LAPLACIAN:
            ratio = lap / EMPTY_LAPLACIAN
            return _EMPTY, 1.0 / (1.0 + ratio * ratio)
        best_name, best_score = _EMPTY, -1.0
        for index, name in enumerate(self.names):
            if name == _EMPTY:
                continue
            score = _peak_score(window, self.data[index])
            if score > best_score:
                best_name, best_score = name, score
        return best_name, best_score

    def match_all(self, windows: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.match(window) for window in windows]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, names=np.array(self.names), data=self.data)

    @classmethod
    def load(cls, path: Path) -> TemplateSet:
        with np.load(Path(path), allow_pickle=False) as handle:
            names = [str(name) for name in handle["names"]]
            data = np.asarray(handle["data"], dtype=np.uint8)
        return cls(names, data)

    def save_png(self, path: Path, cols: int = 8, pad: int = 2) -> None:
        """把 15 个模板拼成一张图，便于人眼核对（`tools/calibrate.py` 会用到）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        px = self.data.shape[1]
        rows = (len(self.names) + cols - 1) // cols
        sheet = np.full((rows * (px + pad) + pad, cols * (px + pad) + pad), 255, np.uint8)
        for index, name in enumerate(self.names):
            r, c = divmod(index, cols)
            y0, x0 = pad + r * (px + pad), pad + c * (px + pad)
            sheet[y0 : y0 + px, x0 : x0 + px] = self.data[index]
        ok, buffer = cv2.imencode(".png", sheet)
        if not ok:
            raise RuntimeError(f"模板图编码失败：{path}")
        path.write_bytes(buffer.tobytes())


def _align_to(sample: np.ndarray, reference: np.ndarray, radius: int = _ALIGN_RADIUS) -> np.ndarray:
    """把 `sample` 平移整数像素，使其与 `reference` 对齐，返回同尺寸的切片。

    这是建模板的关键一步：开局帧里 5 个兵虽然长得一样，但落点相对"格子中心"的
    误差各不相同（实测有 ±6 px 的线性偏差）。**不对齐就直接取中位数，模板会被
    糊掉** —— 实测两个黑车（棋盘两端）的中位数模板只有 0.66 的自相关。
    """
    if radius <= 0:
        return sample
    size = sample.shape[0]
    padded = cv2.copyMakeBorder(sample, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
    response = cv2.matchTemplate(padded, reference, cv2.TM_CCOEFF_NORMED)
    _, _, _, best = cv2.minMaxLoc(response)
    top, left = best[1], best[0]
    return padded[top : top + size, left : left + size]


def _class_template(samples: list[np.ndarray]) -> np.ndarray:
    """同类样本 → 一张模板：先全部对齐到第一个样本，再取逐像素中位数。"""
    reference = samples[0]
    aligned = [reference] + [_align_to(sample, reference) for sample in samples[1:]]
    return np.median(np.stack(aligned), axis=0).astype(np.uint8)


def build_templates(
    img: np.ndarray, geom: BoardGeometry, board: Board, mapper: ScreenMapper
) -> TemplateSet:
    """按开局局面（或任何已知局面）从一帧画面里构建模板集。

    `board` 是这一帧的正确答案、`mapper` 说明棋盘方向 —— 两者一起给出每个交叉点
    的类别，等于把画面变成一份带标注的训练集。
    """
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    for row in range(10):
        for col in range(9):
            rank, file = mapper.to_board(row, col)
            name = PIECE_CHARS[board.at(rank, file)]
            buckets[name].append(reference_patch(img, geom, row, col))

    done: dict[str, np.ndarray] = {}
    for name in CLASS_NAMES:
        samples = buckets[name]
        if samples:
            done[name] = _class_template(samples)
        elif name != _EMPTY:
            fallback = _fallback_template(done, 1 if name.isupper() else -1)
            if fallback is None:
                raise ValueError(f"开局帧里既没有 {name!r} 也没有任何同色棋子可作兜底")
            done[name] = fallback
        else:
            done[name] = np.zeros((TEMPLATE_PX, TEMPLATE_PX), np.uint8)
    return TemplateSet(list(CLASS_NAMES), np.stack([done[name] for name in CLASS_NAMES]))


def load_or_build(
    img: np.ndarray,
    geom: BoardGeometry,
    board: Board,
    mapper: ScreenMapper,
    path: Path = DEFAULT_TEMPLATES_PATH,
    rebuild: bool = False,
) -> TemplateSet:
    """有存好的模板集就直接读，否则从当前帧构建并落盘。"""
    path = Path(path)
    if path.is_file() and not rebuild:
        suite = TemplateSet.load(path)
        missing = [name for name in CLASS_NAMES if name not in suite.names]
        if not missing:
            return suite
    suite = build_templates(img, geom, board, mapper)
    suite.save(path)
    return suite
