"""程序入口：解析命令行、组装各层组件、驱动主循环。

主循环的形状是「抓图 → 定位 → 识别 → 推进一步状态机 → 画面板」，每一步都
**不许抛异常打断整局**：窗口没了、棋盘找不到了、抓到坏帧了，都只是这一帧没有
新信息，局面认知原样保留（见 `session.py` 的 [RF-2]）。

三个容易漏掉的点：

1. **坏帧绝不能变成空盘。** `PrintWindow` 会零星返回全黑帧与残缺帧，它们里面
   的格子看起来就是空的。抓图之后一律先过 `frame_usable`，坏帧把
   `board_provider` 置空，而不是喂给识别层。
2. **窗口可能被拖动/改尺寸/换显示器**（[RF-5]）。每帧拿实际抓到的尺寸和缓存的
   `client_size` 比一次，不一致就 `refresh()` 并把 `BoardGeometry` 丢掉重定位。
3. **命令行默认值不能盖掉配置文件。** `--movetime` 这类开关的 argparse 默认值和
   `DEFAULTS` 相同，光看 `args` 分不出「用户敲了 `--movetime 1200`」和「用户没敲」。
   所以覆盖项只在 `argv` 里**真的出现过**那个开关时才生效。
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from jjchess.config import DEFAULT_CONFIG_PATH, Config, load_config, parse_human_delay
from jjchess.control import Mouse
from jjchess.engine.fallback import FallbackEngine
from jjchess.engine.uci import EngineError, UciEngine, engine_present
from jjchess.geometry import BLACK, RED, BoardGeometry, ScreenMapper
from jjchess.record import GameRecorder
from jjchess.rules.board import START_FEN, Board
from jjchess.session import Session, SessionDeps, State
from jjchess.ui.hotkeys import Hotkeys
from jjchess.ui.panel import Panel
from jjchess.vision.locator import BoardNotFound, locate
from jjchess.vision.recognizer import BOARD_COLS, BOARD_ROWS, both_kings_present, recognize
from jjchess.vision.templates import (
    DEFAULT_TEMPLATES_PATH,
    EMPTY_LAPLACIAN,
    TemplateSet,
    build_templates,
    laplacian_variance,
    make_patch,
)
from jjchess.window import WindowNotFound, find_window, frame_usable

__all__ = [
    "FrameSettle",
    "build_overrides",
    "build_parser",
    "count_pieces",
    "detect_side",
    "dump_frame",
    "grab_usable",
    "log_state",
    "main",
    "record_path",
    "resolve_side",
    "screen_geometry",
    "trace_frame",
]

#: 游戏窗口的标题。微信小程序宿主的标题就是小游戏名字。
WINDOW_TITLE = "JJ象棋"
#: 开局棋子数。数到这个数就认为「对局已经开始，且正处在标准开局」。
OPENING_PIECES = 32
#: 抓图重试：连抓 3 次都不可用就当这一帧没有信息（不改变局面认知）。
GRAB_ATTEMPTS = 3
GRAB_DELAY = 0.15
#: 连续多少帧完全抓不到画面就退出（窗口关了/最小化了）。
MAX_BAD_STREAK = 10
#: 主循环每轮的额外歇息，避免空转把 CPU 吃满。
LOOP_SLEEP = 0.02
#: 画面要连续稳定这么久，才把识别结果当成「新的局面」交给会话。
#:
#: 棋子是**滑过去**的，主循环 20ms 一轮，动画中途抓到的帧里棋子正卡在两格之间。
#: 真机实测：黑方「卒 6 平 5 吃兵」的过程被逐帧喂进 `interpret`，模型先认成
#: 「炮 7 进 8」、再认成别的，一步错步步错，最后只能暂停。等画面停下来再上报，
#: 中间那些过渡帧就自然被丢掉了。
#:
#: 0.30 → 0.50：吃子那一下棋子会闪一下（被吃的子短暂消失又出现），0.30 秒只有
#: 约 2.5 帧，闪出来的假局面照样判「稳」。实测一轮约 0.12 秒，0.50 秒约 4 帧。
SETTLE_SEC = 0.50
#: 每隔这么久回头复查一次「棋盘还在不在」（秒）。
#:
#: 棋盘定位成功之后 `geom` 就一直复用，`locate()` 再也不会被调用 —— 那条
#: 「找不到棋盘 → `BoardNotFound` → `GAME_OVER`」的路在实战里从来没被走过。
#: 对局结束切到结算弹窗时，旧几何会把结算画面认成一盘空棋，会话读到的就是
#: 「局面不对」而不是「棋盘没了」。复查本身只要约 12ms，代价可以忽略。
RELOCATE_EVERY_S = 2.0
#: 判定「这一排是红字」时，笔画像素的 R-B 至少要这么大。
_RED_SIDE_MARGIN = 60.0
#: 一格中心区域里至少要有这么多笔画像素才拿来判色。
_MIN_INK_PIXELS = 30
#: 判色时取格子中心多大一块（占格距的比例，半径）。
_INK_RADIUS_RATIO = 0.35

EXIT_OK = 0
#: 对局中途出了预料之外的错（不是窗口问题也不是抓图问题）。
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_NO_WINDOW = 2
EXIT_NO_FRAME = 3

# 「再来一局」按钮的位置与颜色特征（都按客户区尺寸的比例算，换分辨率也认得出）。
#
# 一局下完游戏会弹结算画面，绿色的「再来一局」按钮固定在底部中间。要连着下，
# 就得先**认出**这个画面再点 —— 不能盲点：同一个位置在对局中是空棋盘，在大厅里
# 可能是别的按钮，点错了比不点更糟。
RESULT_BAND_Y = 0.910
RESULT_BAND_X = (0.30, 0.70)
PLAY_AGAIN_POINT = (0.502, 0.910)
#: 按钮的绿色通道下限，以及「绿比蓝至少要高出多少」。
#: 真机实测：结算画面 G=172.7 / B=90.5（差 82.2），对局中同一处 G=82.4 / B=70.6（差 11.8）。
RESULT_GREEN_MIN = 140.0
RESULT_GREEN_GAP = 60.0
#: 等结算画面出现、等新一局棋盘出现，各给多久。
RESULT_WAIT_S = 15.0
NEW_BOARD_WAIT_S = 20.0
#: 认出来的棋盘上至少要有这么多子才算「真棋盘」。
#: `locate()` 会被空的装饰棋盘骗到（见主循环里那段注释），所以定位成功之后还得数一数。
MIN_PIECES_ON_BOARD = 2

#: 复查时「新拟合的几何还算同一个棋盘」的容差（原点像素 / 格距像素）。
#
# 真机踩过一次：换局那一下画面正在过渡，`locate()` 拟合出的网格**纵向偏了 6 像素**
# （origin y = 602.69，正常 609.1）。偏一点点平时看不出来，但有子走到红方底线那排时，
# 那一格的匹配置信度掉到 0.37 被判 `unknown`，而 `recognize` 把认不出的格子当**空**处理
# —— 「画面上多了一个子」在会话眼里变成「什么都没变」，直接早退。脚本就这么停在那里
# 一动不动十几分钟。差一点点就换过去，能自愈；差得多（比如「正在匹配」页面上那块装饰
# 棋盘，格距 121 对真棋盘 154）说明这一帧根本不是真棋盘，宁可不换。
GEOM_CLOSE_PX = 25.0
GEOM_CLOSE_CELL_PX = 6.0

#: 「有几格认不出来」这条提醒最多多久打一次（秒）。
#: 几何一直偏着的话每帧都会撞上，不节流就是 20 行/秒地刷屏。
UNKNOWN_WARN_EVERY_S = 5.0
_unknown_warn_at = 0.0


# ================================================================ 命令行


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jjchess",
        description="JJ象棋自动下棋（Windows + 微信小游戏 + 皮卡鱼引擎）。",
    )
    parser.add_argument(
        "--side",
        choices=["auto", "red", "black"],
        default="auto",
        help="我方执色。auto 从画面上「哪一方的子在自己这边」自动判定（默认 auto）",
    )
    parser.add_argument(
        "--movetime", type=int, default=1200, help="每步思考时间，毫秒（默认 1200）"
    )
    parser.add_argument("--threads", type=int, default=8, help="引擎线程数（默认 8）")
    parser.add_argument("--hash", type=int, default=512, help="引擎置换表大小，MB（默认 512）")
    parser.add_argument(
        "--dry-run", action="store_true", help="彩排：照常识别与思考，但不真的点鼠标"
    )
    parser.add_argument("--no-panel", action="store_true", help="不显示实时可视化面板")
    parser.add_argument("--no-record", action="store_true", help="不导出中文记谱文件")
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        metavar="N",
        help="连下多少局后退出；0（默认）表示一直下，直到 ESC 或出错。每局单独出一份记谱",
    )
    parser.add_argument("--calibrate", action="store_true", help="运行标定工具（抓帧建模板）后退出")
    parser.add_argument("--debug-dir", default=None, help="调试产物（记谱、坏帧）写到这个目录")
    parser.add_argument(
        "--human-delay",
        nargs="?",
        const="",
        default=None,
        metavar="MIN-MAX",
        help="拟人化思考延迟，秒。裸写 --human-delay 等价于 --human-delay 0.5-2.0",
    )
    parser.add_argument(
        "--config", default=None, help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}）"
    )
    parser.add_argument("--engine", default=None, help="皮卡鱼可执行文件路径")
    parser.add_argument("--nnue", default=None, help="NNUE 权重文件路径")
    return parser


def _given(argv: list[str], flag: str) -> bool:
    """`argv` 里是否真的出现过 `flag`（含 `--flag=value` 写法）。"""
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def build_overrides(args: argparse.Namespace, argv: list[str]) -> dict:
    """把 CLI 参数翻成 `load_config` 的 overrides。

    只在开关**真的出现在命令行里**时才覆盖 —— 否则 argparse 的默认值会悄悄
    盖掉配置文件（两者数值恰好相同，用户改了配置文件却不见效，极难排查）。
    """
    human_delay = parse_human_delay(args.human_delay)  # 非法区间抛 ValueError，由调用方转中文提示
    return {
        "engine_path": Path(args.engine) if args.engine else None,
        "nnue_path": Path(args.nnue) if args.nnue else None,
        "movetime_ms": args.movetime if _given(argv, "--movetime") else None,
        "threads": args.threads if _given(argv, "--threads") else None,
        "hash_mb": args.hash if _given(argv, "--hash") else None,
        "debug_dir": Path(args.debug_dir) if args.debug_dir else None,
        "human_delay": human_delay,
        "record": False if args.no_record else None,
    }


# ================================================================ 小工具


def _cell_center(img: np.ndarray, geom: BoardGeometry, row: int, col: int) -> np.ndarray | None:
    """取某个交叉点中心的一小块，越界返回 None。"""
    x, y = geom.cell_to_px(row, col)
    half = max(4, int(geom.cell_px * _INK_RADIUS_RATIO))
    x0, y0 = round(x) - half, round(y) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None
    return img[y0:y1, x0:x1]


def detect_side(img: np.ndarray, geom: BoardGeometry) -> int:
    """判断我方执红还是执黑 —— 看屏幕**最下面那一排**棋子的字是红还是黑。

    依据：这类对局界面总是把「我方」画在自己这一侧（下方），红方棋子用红字、
    黑方用黑字。取的是格子中心的笔画像素，用 `R - B` 区分 —— 木色棋盘本身
    R 也大于 B，所以必须先靠「比周围暗」把笔画挑出来再判色。
    """
    reddish = 0
    inked = 0
    for col in range(BOARD_COLS):
        patch = _cell_center(img, geom, BOARD_ROWS - 1, col)
        if patch is None or patch.size == 0:
            continue
        gray = patch.astype(np.float32).mean(axis=2)
        ink = gray < float(gray.mean() - gray.std())
        if int(ink.sum()) < _MIN_INK_PIXELS:
            continue
        mean_bgr = patch[ink].mean(axis=0)
        inked += 1
        if float(mean_bgr[2]) - float(mean_bgr[0]) > _RED_SIDE_MARGIN:
            reddish += 1
    if inked == 0:
        return RED  # 一片空白（比如大厅画面）：给个安全默认值，别抛异常
    return RED if reddish * 2 >= inked else BLACK


def resolve_side(name: str, img: np.ndarray | None, geom: BoardGeometry | None) -> int:
    """把 `--side` 的取值落成 `RED` / `BLACK`。"""
    if name == "red":
        return RED
    if name == "black":
        return BLACK
    if img is None or geom is None:
        return RED
    return detect_side(img, geom)


def screen_geometry(
    geom: BoardGeometry | None, client_origin: tuple[int, int]
) -> BoardGeometry | None:
    """把**客户区**几何平移到**屏幕绝对**坐标。

    `locate()` 跑在 `grab()` 抓到的画面上，那个画面是客户区，左上角是 `(0, 0)`。
    而 `mouse.play_move` 要的是屏幕绝对像素。本机 JJ象棋 的客户区原点在
    `(-1440, -279)`（竖屏副显示器，窗口顶部还探出屏幕 279px），不换算的话点击会
    落到**另一块显示器**上去 —— 而且干跑看不出来，因为 `[dry-run] 点击` 打印的就是
    几何里那两个数。

    面板那边要的仍然是客户区几何（它叠加的是客户区画面），所以两种几何并存，
    这个函数专门产屏幕那一份。
    """
    if geom is None:
        return None
    return geom.with_offset(*client_origin)


def count_pieces(img: np.ndarray, geom: BoardGeometry) -> int:
    """数棋盘上有多少个格子「有子」。

    判据是拉普拉斯方差（棋子上的笔画让它比空格高两个数量级，实测空点最高
    1148、棋子最低 8808，中间取 3000）。用它的场合是「模板还没建、但需要知道
    现在是不是标准开局」—— 开局正好 32 个。
    """
    total = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if laplacian_variance(make_patch(img, geom, row, col)) >= EMPTY_LAPLACIAN:
                total += 1
    return total


def grab_usable(
    window, attempts: int = GRAB_ATTEMPTS, delay: float = GRAB_DELAY
) -> np.ndarray | None:
    """连抓几次，返回第一张 `frame_usable` 的帧；全是坏帧则返回 None。

    坏帧是 `PrintWindow` 的固有毛病（全黑帧、右下角残缺帧），重抓基本能好。
    返回 None 时调用方**必须**当「这一帧没有信息」，绝不能当「棋盘空了」。
    """
    for index in range(max(1, int(attempts))):
        img = window.grab()
        if frame_usable(img):
            return img
        if index + 1 < attempts and delay > 0:
            time.sleep(delay)
    return None


def record_path(debug_dir: Path | None, started_at: dt.datetime) -> Path:
    """记谱文件的落盘位置：`<debug-dir>/game-YYYYmmdd-HHMMSS.txt`。"""
    base = Path(debug_dir) if debug_dir is not None else Path("records")
    return base / f"game-{started_at.strftime('%Y%m%d-%H%M%S')}.txt"


def _run_calibrate(argv: list[str]) -> int:
    """把控制权交给 `tools/calibrate.py`（它不在包里，只能按路径加载）。"""
    path = Path(__file__).resolve().parents[2] / "tools" / "calibrate.py"
    if not path.is_file():
        print(f"错误：找不到标定工具 {path}", file=sys.stderr)
        return EXIT_USAGE
    spec = importlib.util.spec_from_file_location("jjchess_calibrate", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 环境异常
        print(f"错误：加载标定工具失败 {path}", file=sys.stderr)
        return EXIT_USAGE
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(list(argv)))


def _open_engine(cfg: Config):
    """有皮卡鱼就用皮卡鱼，没有就退回内置简易引擎（棋力很低，只保证能跑）。"""
    if engine_present(cfg.engine_path):
        print(f"引擎：{cfg.engine_path}（线程 {cfg.threads}，置换表 {cfg.hash_mb}MB）")
        try:
            return UciEngine(
                cfg.engine_path, threads=cfg.threads, hash_mb=cfg.hash_mb, nnue=cfg.nnue_path
            )
        except EngineError as exc:
            # 握手失败（引擎被杀软删了、exe 架构不对、启动超时）。早先这里没有任何兜底：
            # 异常穿出 `main()`，用户拿到的是英文 traceback，而我们明明有一个能用的
            # 内置引擎可以退。
            print(f"警告：皮卡鱼启动失败（{exc}），退回内置简易引擎（棋力很低）。", file=sys.stderr)
            return FallbackEngine()
    print(
        f"警告：没找到皮卡鱼引擎 {cfg.engine_path}，退回内置简易引擎（棋力很低）。"
        "运行 `uv run python tools/fetch_pikafish.py` 可以自动下载。",
        file=sys.stderr,
    )
    return FallbackEngine()


# ================================================================ 主流程


def _wait_for_board(args: argparse.Namespace, cfg: Config, window, hotkeys: Hotkeys):
    """开局前的等待：抓图 → 定位 → 判定执色 → 备好模板 → 首次识别。

    返回 `(img, geom, rec, mapper, templates, my_side)`；用户按 ESC 或
    连续抓不到画面时返回 `None`。
    """
    my_side = None if args.side == "auto" else resolve_side(args.side, None, None)
    templates = (
        TemplateSet.load(DEFAULT_TEMPLATES_PATH) if DEFAULT_TEMPLATES_PATH.is_file() else None
    )
    if templates is not None:
        print(f"已加载识别模板：{DEFAULT_TEMPLATES_PATH}（{len(templates.names)} 类）")
    else:
        print("还没建过识别模板，会在看到标准开局局面时自动采集")

    geom: BoardGeometry | None = None
    bad_streak = 0
    announced = False

    while True:
        if "quit" in hotkeys.poll():
            print("收到退出指令，结束。")
            return None
        img = grab_usable(window)
        if img is None:
            bad_streak += 1
            if bad_streak >= MAX_BAD_STREAK:
                print(
                    "错误：连续抓不到可用画面。JJ象棋 窗口可能被最小化或关闭了。",
                    file=sys.stderr,
                )
                return None
            print(f"抓到的画面不可用（第 {bad_streak} 次），重试…", file=sys.stderr)
            time.sleep(0.4)
            continue
        bad_streak = 0

        if geom is None:
            try:
                geom = locate(img)
            except BoardNotFound:
                if not announced:
                    print("还没看到棋盘（可能停在大厅/菜单），等待对局开始…")
                    announced = True
                time.sleep(1.0)
                continue
            announced = False
            print(f"棋盘定位：{geom}")

        if my_side is None:
            my_side = detect_side(img, geom)
            print(f"自动判定我方执{'红' if my_side == RED else '黑'}")

        mapper = ScreenMapper(my_side)
        if templates is None:
            pieces = count_pieces(img, geom)
            if pieces != OPENING_PIECES:
                print(f"当前画面数出 {pieces} 个棋子，不是标准开局（应为 {OPENING_PIECES}），等待…")
                time.sleep(1.0)
                continue
            templates = build_templates(img, geom, Board.from_fen(START_FEN), mapper)
            templates.save(DEFAULT_TEMPLATES_PATH)
            print(f"已从开局局面采集识别模板：{DEFAULT_TEMPLATES_PATH}")

        try:
            rec = recognize(img, geom, templates, mapper, side_to_move=RED, conf_min=cfg.conf_min)
        except Exception as exc:  # noqa: BLE001 - 首帧识别失败不该直接崩
            print(f"识别失败（{exc}），重试…", file=sys.stderr)
            time.sleep(0.5)
            continue

        if rec.unknown:
            print(f"注意：有 {len(rec.unknown)} 个格子置信度不足，已按「空」处理")
        print(f"识别完成，最低置信度 {float(rec.confidence.min()):.3f}")
        return img, geom, rec, mapper, templates, my_side


def log_state(session: Session, previous: State | None) -> State:
    """状态一变就打印一行，返回新状态。

    `--no-panel` 时这是**唯一**能看出会话在干什么的窗口。实测踩到过：不带面板跑
    真机，脚本卡在 `PAUSED` 里一动不动，终端上却一句话都没有 —— 状态和 `message`
    只画在面板上，人只能靠猜。有面板时这一行也不碍事，反而留下了可回看的轨迹。
    """
    if session.state is not previous:
        suffix = f"  {session.message}" if session.message else ""
        print(f"[{session.state.value}]{suffix}")
    return session.state


def dump_frame(img, cfg: Config, tag: str) -> None:
    """把当前画面存进 `--debug-dir`，用于事后复盘。

    真机跑出问题时不存图就只能靠猜：`VERIFY` 失败的那一刻画面长什么样、识别成什么，
    事后完全看不到。文件名带 `tag` 与毫秒时间戳，同一次故障可以存多张。

    **不用 `cv2.imwrite`** —— 它在非 ASCII 路径上会静默失败，只返回 False。
    """
    if cfg.debug_dir is None or img is None:
        return
    try:
        cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%H%M%S-%f")[:-3]
        path = cfg.debug_dir / f"frame-{stamp}-{tag}.png"
        ok, buf = cv2.imencode(".png", img)
        if ok:
            path.write_bytes(buf.tobytes())
    except OSError as exc:  # 存图失败绝不能拖垮整局
        print(f"存图失败（{exc}）", file=sys.stderr)


def trace_frame(rec, previous: str | None) -> str | None:
    """画面里的局面一变就打一行 `[画面] <FEN>`，返回这次的 FEN。

    这一行和 `log_state` 的 `[状态]` 行合起来才是完整的现场：只看状态会以为
    「对方走 h9i7」是真的，其实那只是 `interpret` 猜出来的。把画面 FEN 单独打出来，
    模型和 Reality 的分歧一眼就能对上。
    """
    if rec is None:
        return previous
    fen = rec.board.to_fen()
    if fen != previous:
        print(f"[画面] {fen}")
    return fen


class FrameSettle:
    """把「画面还没停」的过渡帧挡在会话之外。

    主循环 20ms 一轮，棋子却是几十毫秒滑过去的，所以一次走子会接连抓到好几帧
    「棋子卡在两格之间」的过渡局面。这些帧直接喂给 `Session` 是灾难：`interpret`
    会照着它们编出一个能自圆其说的假着法，模型就此一步错、步步错。

    用法：每帧 `feed(fen, now)` 一次，只有返回 True 时才把这一帧的棋盘交给会话。
    """

    def __init__(self, seconds: float = SETTLE_SEC) -> None:
        self.seconds = seconds
        self._fen: str | None = None
        self._since = 0.0

    def feed(self, fen: str, now: float) -> bool:
        """喂一帧的 FEN，返回「画面已经停稳，可以当成新局面了吗」。"""
        if fen != self._fen:
            self._fen = fen
            self._since = now
            return False
        return now - self._since >= self.seconds

    def reset(self) -> None:
        self._fen = None
        self._since = 0.0


def _handle_hotkeys(keys: set[str], session: Session, mouse: Mouse) -> None:
    for key in keys:
        if key == "pause":
            if session.state is State.PAUSED:
                session.resume()
                print("已继续（F8）")
            else:
                session.pause()
                print("已暂停（F8 继续）")
        elif key == "step":
            session.step_once()
            print(f"单步（F9）→ {session.state.value}")
        elif key == "dryrun":
            mouse.dry_run = not mouse.dry_run
            print(f"彩排模式：{'开' if mouse.dry_run else '关'}（F10）")
        elif key == "quit":
            session.request_stop()
            print("收到退出指令（ESC）")


def _handle_delay_hotkeys(keys: set[str], session: Session, mouse: Mouse) -> None:
    """拟人延迟期间的热键处理：**只理「暂停」和「退出」**。

    这个函数是挂在 `interruptible_sleep` 的 `tick` 上的 —— 主循环那时正阻塞在
    `session.step()` 里面，只有它能去读键盘（不挂的话按 F8 要等延迟走完才生效，
    真机实测等了 3.4 秒）。

    为什么不像 `_handle_hotkeys` 那样四个键都理：

    - `F9`（单步）调的是 `session.step_once()`，而这段延迟本来就在 `session.step()`
      里面 —— 放它进来就是**递归跑状态机**；
    - `F10` 改的是 `mouse.dry_run`，延迟期间没有点击，改了也不生效，等延迟结束
      之后下一轮再处理更清楚。

    `pause` 加 `state is not PAUSED` 的守卫：延迟期间状态是 `THINKING`，重复 `pause()`
    会把 `_state_before_pause` 覆写成 `THINKING`，恢复时回到一个已经过期的状态。
    """
    for key in keys:
        if key == "pause" and session.state is not State.PAUSED:
            session.pause()
            print("已暂停（F8 继续）")
        elif key == "quit":
            session.request_stop()
            print("收到退出指令（ESC）")


def geom_close(
    new: BoardGeometry,
    old: BoardGeometry,
    tol_px: float = GEOM_CLOSE_PX,
    tol_cell: float = GEOM_CLOSE_CELL_PX,
) -> bool:
    """新拟合的几何是不是「同一个棋盘，只是更准一点」。

    为什么要比这一下：几何只在开局/换局时定位一次，之后一直复用，而换局那一下画面
    正在过渡，`locate()` 可能拟合出一个偏了几像素的网格。偏一点点就换过去能自愈；
    差得多说明这一帧根本不是真棋盘（「正在匹配」页面上那块装饰棋盘格距只有 121，
    真棋盘 154），宁可不换。常量的来历见 `GEOM_CLOSE_PX` 上面的注释。
    """
    if abs(new.origin[0] - old.origin[0]) > tol_px:
        return False
    if abs(new.origin[1] - old.origin[1]) > tol_px:
        return False
    return abs(new.dx - old.dx) <= tol_cell and abs(new.dy - old.dy) <= tol_cell


def _retry_unknown(img, geom, templates, mapper, cfg, rec):
    """认出棋盘了、但有几格没认出来时，换个刚拟合的几何再认一遍。

    `recognize` 把认不出的格子当**空**处理，这是最危险的一种错：画面上明明多了一个子，
    会话看到的却是「什么都没变」，`_grid_diff == 0` 直接早退，脚本停在那儿一动不动
    （真机踩过，停了十几分钟）。换个几何认干净了就换过去；认不干净就原样返回，
    交给上层照旧处理，并打一条节流提醒。

    换过去之前还要过一道 `geom_close`：新拟合的几何得是「同一个棋盘、只是更准一点」。
    差得太多说明这一帧根本不是真棋盘（「正在匹配」页面上那块装饰棋盘格距只有 121），
    那就宁可不换。
    """
    global _unknown_warn_at
    try:
        fresh = locate(img)
    except BoardNotFound:
        fresh = None
    if (
        fresh is not None
        and geom_close(fresh, geom)
        and count_pieces(img, fresh) >= MIN_PIECES_ON_BOARD
    ):
        retry = recognize(img, fresh, templates, mapper, side_to_move=RED, conf_min=cfg.conf_min)
        if retry is not None and not retry.unknown:
            return retry
    now = time.monotonic()
    if not both_kings_present(rec.board):
        # 双方将帅都不在，那根本不是一个棋局画面（「正在匹配」、大厅、结算弹窗都会
        # 认出一盘几乎全空的棋）。上层会把它当「棋盘已消失」处理，这里就别再刷
        # 「有 N 格认不出来」了 —— 那块空棋盘当然格格里都认不出东西。
        return rec
    if now - _unknown_warn_at >= UNKNOWN_WARN_EVERY_S:
        _unknown_warn_at = now
        cells = "、".join(f"第 {r} 行第 {c} 列" for r, c in rec.unknown[:3])
        print(
            f"[画面] 有 {len(rec.unknown)} 格认不出来（{cells}），这一帧先照旧认",
            file=sys.stderr,
        )
    return rec


def is_result_screen(img) -> bool:
    """画面底部那个绿色的「再来一局」按钮在不在。

    只看颜色不看文字：按钮是一块很大、很饱和的亮绿，对局中同一位置是空棋盘
    （暗绿木头色），两者差得远 —— 真机实测绿通道 172.7 vs 82.4，绿蓝差 82.2 vs 11.8。
    """
    if img is None:
        return False
    height, width = img.shape[:2]
    y = int(RESULT_BAND_Y * height)
    x0, x1 = int(RESULT_BAND_X[0] * width), int(RESULT_BAND_X[1] * width)
    band = img[max(0, y - 12) : y + 12, x0:x1]
    if band.size == 0:
        return False
    blue, green, _red = (float(v) for v in band.reshape(-1, 3).mean(axis=0))
    return green >= RESULT_GREEN_MIN and green - blue >= RESULT_GREEN_GAP


def _new_board_geom(frame: np.ndarray, prev_geom: BoardGeometry | None) -> BoardGeometry | None:
    """这一帧是不是「真棋盘」？是就返回它的几何，不是就返回 `None`。

    三道闸门，每一道都是真机咬过之后补的：

    1. 结算画面本身不能算棋盘 —— 练习局的结算画面把**终局局面**留在屏幕上，
       定位得到、双方将帅也都认得出，看起来就是一盘下到一半的棋。上一局刚赢
       就被它骗了一次「点击后局面校验失败」。
    2. 「正在匹配」那类页面上有一块**空的**装饰棋盘，`locate()` 照样拟得出网格
       （真机实测格距 102.7，真棋盘 125.8）。→ 数一数盘上有没有子。
    3. 结算画面上那块棋盘是**缩小过的**，而且摆着终局局面 —— 子数一点不少，
       刚点完「再来一局」的一瞬间画面还没切走，光数子拦不住（真机 run17 实测：
       点完 137ms 后抓到 `origin=(242.2, 380.5) dx=120.9`，真棋盘是
       `(101.5, 609.3) dx=154.6`）。→ 新棋盘得和上一局「差不多在同一个地方」。
    """
    if is_result_screen(frame):
        return None
    try:
        geom = locate(frame)
    except BoardNotFound:
        return None
    if count_pieces(frame, geom) < MIN_PIECES_ON_BOARD:
        return None
    if prev_geom is not None and not geom_close(geom, prev_geom):
        return None
    return geom


def _next_round(window, hotkeys: Hotkeys, mouse: Mouse, prev_geom: BoardGeometry | None = None):
    """一局结束后开下一局：等结算画面 → 点「再来一局」→ 等新棋盘出现。

    成功返回新一局的 `BoardGeometry`；用户按了 ESC、或者等不到该等的东西，返回 `None`
    （调用方据此收工）。**每一步都有超时** —— 上一局结束时棋盘就是消失的，看不出
    「点击没生效」和「还在加载」的区别，不设上限就会一直空转。
    """
    if getattr(mouse, "dry_run", False):
        print("彩排模式不会真的点鼠标，不再自动开新局。")
        return None

    print("准备下一局：等结算画面…")
    deadline = time.monotonic() + RESULT_WAIT_S
    img = None
    while time.monotonic() < deadline:
        if "quit" in hotkeys.poll():
            print("收到退出指令（ESC），不再开新局。")
            return None
        frame = grab_usable(window)
        if frame is not None and is_result_screen(frame):
            img = frame
            break
        time.sleep(0.3)
    if img is None:
        print("等不到「再来一局」按钮，不再自动开新局。", file=sys.stderr)
        return None

    height, width = img.shape[:2]
    mouse.click_client(
        int(PLAY_AGAIN_POINT[0] * width),
        int(PLAY_AGAIN_POINT[1] * height),
        window.client_origin,
    )
    print("已点「再来一局」，等新一局开始…")

    deadline = time.monotonic() + NEW_BOARD_WAIT_S
    while time.monotonic() < deadline:
        if "quit" in hotkeys.poll():
            print("收到退出指令（ESC），不再开新局。")
            return None
        frame = grab_usable(window)
        if frame is not None:
            geom = _new_board_geom(frame, prev_geom)
            if geom is not None:
                print(f"新一局棋盘定位：{geom}")
                return geom
        time.sleep(0.3)
    print("点了「再来一局」却等不到棋盘，不再自动开新局。", file=sys.stderr)
    return None


def _play(args: argparse.Namespace, cfg: Config, window, engine) -> int:
    hotkeys = Hotkeys()
    # 启动时正停在结算画面（上一局刚下完）→ 先把「再来一局」点了再等棋盘。
    #
    # 不点的话 `_wait_for_board` 会拿到终局局面：练习局的结算画面**棋盘还留在
    # 屏幕上**（显示终局局面），定位得到、双方将帅也都认得出，看起来就是一盘棋，
    # 于是脚本会在一盘已经下完的棋上面走子。真机 run16 之后就停在这儿。
    first = grab_usable(window)
    if first is not None and is_result_screen(first) and not args.dry_run:
        print("启动时正停在结算画面，先点「再来一局」。")
        _next_round(window, hotkeys, Mouse(dry_run=False, my_side=RED))
    ready = _wait_for_board(args, cfg, window, hotkeys)
    if ready is None:
        return EXIT_OK
    _first_img, geom, rec, mapper, templates, my_side = ready
    # 最近一次**确认过是对的**棋盘几何。`geom` 会在棋盘消失时被清掉，换局那一下
    # 正需要它 —— 见 `_new_board_geom` 的第 3 道闸门。
    last_geom = geom

    mouse = Mouse(dry_run=args.dry_run, my_side=my_side)
    if args.dry_run:
        print("彩排模式：不会真的点鼠标（F10 可随时切换）")

    latest: dict[str, Board | None] = {"board": rec.board}
    panel = Panel(enabled=not args.no_panel)

    def new_round() -> tuple[Session, GameRecorder | None, dt.datetime]:
        """开一局新的：新的会话 + 新的记谱。

        每局都得换一份 `GameRecorder`，否则第二局的着法会追加到第一局的文件里；
        会话也必须从 `IDLE` 重新起，不然它带着上一局的模型去认新一局的开局。
        """
        started_at = dt.datetime.now()
        rec_obj = GameRecorder(my_side, START_FEN, started_at) if cfg.record else None
        sess = Session(
            SessionDeps(
                board_provider=lambda: latest["board"],
                engine=engine,
                mouse=mouse,
                mapper=mapper,
                recorder=rec_obj,
                config=cfg,
                geometry_provider=lambda: screen_geometry(geom, window.client_origin),
            )
        )
        # 拟人延迟期间也要能按 F8 / ESC：主循环那时正阻塞在 `session.step()` 里面，
        # 没有别的地方会去读键盘。绑在**这一局自己的** session 上，不跟着外面的变量跑。
        sess.deps.delay_tick = lambda: _handle_delay_hotkeys(hotkeys.poll(), sess, mouse)
        return sess, rec_obj, started_at

    session, recorder, started = new_round()

    print("开始对局。热键：F8 暂停/继续，F9 单步，F10 彩排，ESC 退出。")
    if not args.rounds:
        print("下完一局会自动点「再来一局」接着下；想只下一局用 --rounds 1。")
    bad_streak = 0
    last_state: State | None = None
    last_fen: str | None = None
    dumped = False
    board_lost = False
    settle = FrameSettle()
    last_locate = time.monotonic()
    exit_code = EXIT_OK
    finished = 0
    saved = False
    try:
        while not session.should_stop:
            _handle_hotkeys(hotkeys.poll(), session, mouse)
            if session.should_stop:
                break
            if session.state is State.GAME_OVER:
                # spec 4.x 的终局检测写的是「判定对局结束，停止并打印对局记录」。
                # 少了这个出口，进程会一直空转打日志、记谱文件永远不落盘。
                #
                # 判断放在**循环开头**而不是末尾：循环体里有 `continue`（棋盘定位不到
                # 时的早退分支），末尾的判断会被它绕过去 —— 真机上就卡在这里空转。
                # 放在开头，无论从哪条路走到终局，下一轮都会收尾。
                print(f"对局结束（{session.message}），正在收尾…")
                _finish(session, recorder, cfg, started)
                saved = True
                finished += 1
                if args.rounds and finished >= args.rounds:
                    print(f"已下满 {finished} 局（--rounds {args.rounds}），收工。")
                    break
                geom = _next_round(window, hotkeys, mouse, last_geom)
                if geom is None:
                    break
                last_geom = geom
                # 开下一局：会话、记谱、逐帧状态全部重置。
                latest["board"] = None
                settle.reset()
                last_state = None
                last_fen = None
                dumped = False
                board_lost = False
                bad_streak = 0
                last_locate = time.monotonic()
                session, recorder, started = new_round()
                saved = False
                print(f"第 {finished + 1} 局开始。")
                continue

            img = grab_usable(window)
            if img is None:
                bad_streak += 1
                if bad_streak >= MAX_BAD_STREAK:
                    print("错误：连续抓不到可用画面，退出。", file=sys.stderr)
                    # 早先这里是 `return EXIT_NO_FRAME` —— 从 `try` 里直接返回会**跳过**
                    # 循环后面的 `panel.close()` 与 `_finish()`（没有 finally），
                    # 这一局已经下过的着法就此丢失。把游戏窗口最小化几秒就能命中。
                    exit_code = EXIT_NO_FRAME
                    break
                # [RF-2] 坏帧绝不能变成「棋盘空了」：只把这一帧的信息丢掉。
                latest["board"] = None
                rec = None
                session.step()
                last_state = log_state(session, last_state)
                panel.render(None, geom, None, session)
                time.sleep(0.3)
                continue
            bad_streak = 0

            if window.size_changed(img.shape[1], img.shape[0]):
                # [RF-5] 窗口被拖动/改过尺寸，标定失效，必须重新定位。
                print("窗口尺寸变了，重新定位棋盘…", file=sys.stderr)
                window.refresh()
                geom = None
            now = time.monotonic()
            if geom is not None and now - last_locate >= RELOCATE_EVERY_S:
                # 定期回头确认「棋盘还在不在」，顺便**把漂掉的几何纠回来**。
                #
                # 「还在不在」这一半的来历：棋盘一旦定位成功，`geom` 就一直被复用，
                # 于是 `locate()` 再也不会被调用 —— 那条「找不到棋盘 → `BoardNotFound`
                # → 棋盘消失 → `GAME_OVER`」的路**在实战里从来没被走过**。对局结束时
                # 画面切到结算弹窗，拿旧几何去认会得到一盘空棋（或者一帧面目全非的
                # 乱子），会话读到的是「局面不对」而不是「棋盘没了」，最后被误报成
                # 「点击后局面校验失败」暂停。所以每 `RELOCATE_EVERY_S` 秒复查一次：
                # 查不到就把 `geom` 清掉，让下一帧走回 `BoardNotFound` 那条老路。
                #
                # 「纠回来」这一半的来历（真机第二次踩）：几何只在开局/换局时定位一次，
                # 而换局那一下画面正在过渡，拟合可能**偏几像素**（实测 origin y 602.69，
                # 正常 609.1）。偏一点点平时没事，可一旦有子走到红方底线那排，那格的
                # 匹配置信度会掉到 0.37 被判 `unknown`，而 `recognize` 把认不出的格子当
                # **空**处理 —— 「画面上多了一个子」在会话眼里变成「什么都没变」，
                # `_grid_diff == 0` 直接早退；`opponent_idle_sec` 默认是 0（用户要求：
                # 对方想多久都等，不许因为局面静止就收摊），于是脚本停在那里一动不动
                # 十几分钟。
                #
                # 纠偏**不在这里做**：`locate()` 自己每次都会抖 ±1.5 像素，每 2 秒换一次
                # 几何等于让采样窗口一直晃，反而把临界格晃出更多 `unknown`（真机实测：
                # 每 2 秒换一次的版本跑一分钟就撞上「点击后局面校验失败」）。
                # 换几何只在一个明确信号下做 —— 见 `_retry_unknown`。
                #
                # 复查本身只花约 12ms。
                last_locate = now
                try:
                    locate(img)
                except BoardNotFound:
                    print("这一帧找不到棋盘，先当作棋盘已消失…", file=sys.stderr)
                    geom = None
            if geom is None:
                try:
                    found = locate(img)
                except BoardNotFound:
                    latest["board"] = None
                    rec = None
                    session.step()
                    last_state = log_state(session, last_state)
                    panel.render(img, None, None, session)
                    time.sleep(0.5)
                    continue
                if count_pieces(img, found) < MIN_PIECES_ON_BOARD:
                    # 定位成功 ≠ 找到了棋盘。「正在匹配」那类页面上画着一块**空的**
                    # 装饰棋盘，`locate()` 照样能拟出网格来（真机实测格距 102.7，
                    # 真棋盘是 125.8）。拿它去认只会得到一盘空棋，然后会话就卡在
                    # IDLE 里等一个永远等不到的开局。棋盘上总得有子，没子就不是。
                    latest["board"] = None
                    rec = None
                    session.step()
                    last_state = log_state(session, last_state)
                    panel.render(img, None, None, session)
                    time.sleep(0.5)
                    continue
                geom = found
                last_geom = found
                last_locate = time.monotonic()
                print(f"棋盘重新定位：{geom}")

            try:
                rec = recognize(
                    img, geom, templates, mapper, side_to_move=RED, conf_min=cfg.conf_min
                )
                if rec is not None and rec.unknown:
                    # 有几格认不出来 —— 换个「这一帧刚拟合出来」的几何再认一遍。
                    #
                    # `recognize` 把认不出的格子当**空**处理，这是最危险的一种错：
                    # 画面上明明多了一个子，会话看到的却是「什么都没变」。真机踩过一次
                    # （几何纵向偏 6 像素，红方底线那格置信度掉到 0.37），脚本因此停了
                    # 十几分钟不动。认干净了就换过去。
                    rec = _retry_unknown(img, geom, templates, mapper, cfg, rec)
                if rec is not None and is_result_screen(img):
                    # 练习局（「JJ象棋陪练」）的结算画面**棋盘还留在屏幕上**，
                    # 显示的是终局局面 —— 定位得到，双方将帅也都认得出，
                    # 看起来和一盘正在下的棋一模一样，只有底下那颗绿色的
                    # 「再来一局」能把它认出来。
                    #
                    # 不拦的话，会话会拿终局局面去和「我方这步之后的预期」比，
                    # 5 秒后报「点击后局面校验失败，已暂停」—— 而棋其实早就下完了。
                    # 真机 run16 撞上：我方绝杀赢了，脚本反而停住了。
                    reason = "这一帧是结算画面（有「再来一局」），当作棋盘已消失"
                elif rec is not None and not both_kings_present(rec.board):
                    # 结算画面（「再来一局」弹窗）定位不了棋盘，可 `geom` 一直在复用，
                    # 于是拿旧几何去认会得到一盘空棋。那不是局面，是「画面没了」。
                    # 转成 `None` 交给会话，让它走 `BOARD_LOST` → `GAME_OVER` 那条路。
                    #
                    # 只在这一段「看不到棋盘」的**头一帧**打一次：结算画面停在那里之后
                    # 每帧都打，几十秒能刷几百行，把真正有用的日志淹掉。
                    reason = "认不到双方将帅，当作棋盘已消失"
                elif rec is not None and not rec.board.is_legal_position():
                    # 将帅照面 / 将帅摆出九宫。这种局面喂给皮卡鱼，它会判
                    # `CRITICAL ERROR` 并**自杀退出** —— 引擎层虽然有重启兜底，
                    # 但白烧一次额度不说，模型一旦采纳了这种局面，后面全盘错位。
                    reason = "认出的局面不合法（将帅照面或不在九宫），跳过这一帧"
                else:
                    reason = ""
                if reason:
                    if not board_lost:
                        print(f"[画面] {reason}", file=sys.stderr)
                        board_lost = True
                    rec = None
                else:
                    board_lost = False
            except Exception as exc:  # noqa: BLE001 - 单帧识别失败不该打断整局
                print(f"识别失败（{exc}），跳过这一帧", file=sys.stderr)
                latest["board"] = None
                rec = None

            if rec is None:
                latest["board"] = None

            if rec is not None and settle.feed(rec.board.to_fen(), time.monotonic()):
                latest["board"] = rec.board

            last_fen = trace_frame(rec, last_fen)
            session.step()
            if last_state is not session.state and session.state is State.PAUSED:
                # 暂停是要人来看的那一刻：把画面存下来，顺手把「模型以为的」和
                # 「画面上的」两个 FEN 并排打出来，分歧一眼可见。
                dump_frame(img, cfg, "paused")
                if session.board is not None:
                    print(f"[模型] {session.board.to_fen()}")
                dumped = True
            if session.state is not State.PAUSED and dumped:
                dumped = False
            last_state = log_state(session, last_state)
            panel.render(img, geom, rec, session)
            time.sleep(LOOP_SLEEP)
    except WindowNotFound as exc:
        print(f"窗口没了：{exc}", file=sys.stderr)
        exit_code = EXIT_NO_WINDOW
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在退出…")
    except Exception as exc:  # noqa: BLE001 - 兜住任何意外，保证收尾一定会跑
        # `mouse.play_move` 会抛 `OSError`（`SetCursorPos`/`SendInput` 失败），
        # `panel.render` 会抛 `cv2.error`，`engine.analyse` 的异常也在这里落地。
        # 早先它们会直接穿出 `main()`：既跳过收尾（记谱丢失），又给用户一段
        # 英文 traceback —— 而本项目的硬约束是「一切面向人的输出用中文」。
        print(f"出错了：{exc}", file=sys.stderr)
        exit_code = EXIT_FAILURE
    finally:
        # 收尾放在 `finally` 里，任何一条退出路径都跑得到。
        panel.close()
        # 终局那条路已经自己收过尾了（还要留着进程去开下一局），别重复写一遍文件。
        if not saved:
            _finish(session, recorder, cfg, started)
    return exit_code


def _finish(
    session: Session, recorder: GameRecorder | None, cfg: Config, started: dt.datetime
) -> None:
    print(f"结束状态：{session.state.value}")
    if session.message:
        print(f"最后消息：{session.message}")
    if recorder is None:
        return
    recorder.result = "对局结束" if session.state is State.GAME_OVER else "用户退出"
    if not recorder.moves:
        print("这局一步都没走过，不生成记谱文件。")
        return
    path = record_path(cfg.debug_dir, started)
    try:
        recorder.save(path)
    except OSError as exc:
        # 目录不可写、磁盘满、文件被占用。这里的异常要是穿出去，用户看到的是英文
        # traceback，而这一局明明已经走完了 —— 宁可丢记谱也要把中文原因说清楚。
        print(f"警告：记谱写入失败（{exc}），这一局的记录没能保存。", file=sys.stderr)
        return
    print(f"记谱已保存：{path}（共 {len(recorder.moves)} 步）")


def main(argv: list[str] | None = None) -> int:
    """程序主入口，返回进程退出码：0 正常，2 参数/窗口问题，3 抓不到画面。"""
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    if args.rounds < 0:
        print("错误：--rounds 不能是负数（0 表示一直下，直到 ESC 或出错）。", file=sys.stderr)
        return EXIT_USAGE

    if args.calibrate:
        return _run_calibrate(["--build"])

    try:
        overrides = build_overrides(args, raw)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        cfg = load_config(Path(args.config) if args.config else DEFAULT_CONFIG_PATH, overrides)
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_USAGE

    if cfg.human_delay is not None:
        print(f"拟人化思考延迟：{cfg.human_delay[0]:.1f}~{cfg.human_delay[1]:.1f} 秒")

    try:
        window = find_window(WINDOW_TITLE, timeout_s=10.0)
    except WindowNotFound as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_NO_WINDOW
    width, height = window.client_size
    print(f"已找到窗口：{window.title}（hwnd={window.hwnd}，客户区 {width}x{height}）")

    with _open_engine(cfg) as engine:
        return _play(args, cfg, window, engine)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
