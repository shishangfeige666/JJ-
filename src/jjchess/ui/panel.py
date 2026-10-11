"""OpenCV 实时调试面板。

左半是缩放后的游戏画面 + 识别叠加（每格一个彩色圆点、棋子字符、低置信度红叉），
右半是状态、FEN、引擎输出与热键说明，底部一条状态灯。

**为什么全是 ASCII**：`cv2.putText` 用的是 Hershey 矢量字体，只有 ASCII 字形，
中文字符会画成一串问号。为了守住「不引入新依赖」（PIL / freetype 都不在依赖表里），
面板文字一律走 `to_ascii()` 转换，中文只出现在终端日志和导出的记谱文件里。

`compose()` 是**纯函数**：给同样的入参必得同样的图，也不碰任何窗口。
`Panel` 才负责 `imshow`。这样无头环境（CI、没有桌面会话）也能把面板逻辑测干净。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any

import cv2
import numpy as np

from jjchess.config import DEFAULTS
from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.rules.board import PIECE_CHARS
from jjchess.session import State
from jjchess.vision.recognizer import BOARD_COLS, BOARD_ROWS, Recognition

__all__ = ["Panel", "compose", "conf_color", "status_lines", "to_ascii"]

_COLOR_BG = (28, 28, 32)
_COLOR_STATUS_BG = (48, 48, 56)
_COLOR_TEXT = (228, 228, 228)
_COLOR_DIM = (150, 150, 160)
_COLOR_GREEN = (80, 220, 80)
_COLOR_YELLOW = (60, 200, 230)
_COLOR_RED = (60, 60, 230)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE_H = 22
_LEFT_RATIO = 0.60
_STATUS_H = 32
_TITLE = "JJ象棋 自动下棋"
_FEN_WRAP = 46

#: 交给 OpenCV 的窗口标识，**必须是 ASCII**：OpenCV 的 Win32 后端用 `CreateWindowA`
#: 建窗口，中文会按本地代码页解码成乱码标题（实测 `JJ象棋 自动下棋` 会变成
#: `JJ璞℃…`）。窗口标题另外用 `SetWindowTextW` 补，见 `_find_own_window`。
_WINDOW_KEY = "jjchess-panel"

#: 会话消息里的中文片段 → ASCII 等价物。按长度从长到短替换，避免短片段先吃掉长片段。
_ASCII_TABLE: tuple[tuple[str, str], ...] = (
    (
        "轮到我方但局面异常静止，请检查（按 F8 继续）",
        "my turn but board is frozen, check vision (F8 to resume)",
    ),
    (
        "无法解释局面变化，已暂停（请检查识别）",
        "cannot explain board change, paused (check vision)",
    ),
    ("引擎连续多次没有返回着法，已暂停", "engine returned no move repeatedly, paused"),
    ("点击后局面校验失败，已暂停", "board did not match after click, paused"),
    (
        "彩排：没真的点鼠标，局面不会变，已暂停（按 F10 关闭彩排）",
        "rehearsal: no real click, board unchanged, paused (F10 to disable)",
    ),
    ("拿不到棋盘几何，无法点击，已暂停", "no board geometry, cannot click, paused"),
    ("局面长时间不变，判定对局结束", "board unchanged for long, game over"),
    ("已经看不到棋盘了，判定对局结束", "board is gone, game over"),
    ("已暂停（按 F8 继续）", "PAUSED (F8 to resume)"),
    ("看不懂局面变化（第 ", "cannot explain board change (attempt "),
    ("引擎没有返回着法（第 ", "engine returned no move (attempt "),
    ("引擎给出的着法 ", "engine move "),
    ("在当前局面下不合法，已暂停", " is illegal here, paused"),
    ("，等待画面确认", ", waiting for confirmation"),
    ("等待对局开始…", "waiting for game to start..."),
    ("检测到开局局面", "opening position detected"),
    ("走子已确认", "move confirmed"),
    ("对方走 ", "opponent played "),
    ("已走 ", "played "),
    ("走 ", "played "),
    ("继续", "RESUMED"),
    (" 次）", ")"),
)


def to_ascii(text: str) -> str:
    """把会话消息转成 `cv2.putText` 画得出来的 ASCII。

    已知消息走查表得到可读英文；查表没覆盖到的汉字统一折成 `?` 并去掉重复，
    保证「画出来是问号」这件事不会发生 —— 宁可少看到信息，也不要看到乱码。
    """
    out = text
    for chinese, english in _ASCII_TABLE:
        out = out.replace(chinese, english)
    out = "".join(ch if ch.isascii() else "?" for ch in out)
    while "??" in out:
        out = out.replace("??", "?")
    return out.strip()


def conf_color(
    conf: float, conf_min: float | None = None, conf_good: float | None = None
) -> tuple[int, int, int]:
    """置信度 → BGR 颜色：够好绿、及格黄、不及格红。"""
    low = DEFAULTS.conf_min if conf_min is None else conf_min
    high = DEFAULTS.conf_good if conf_good is None else conf_good
    if conf >= high:
        return _COLOR_GREEN
    if conf >= low:
        return _COLOR_YELLOW
    return _COLOR_RED


def status_lines(session: Any, rec: Recognition | None = None) -> list[str]:
    """右半面板的文字行。纯函数，方便直接断言。"""
    state = getattr(session, "state", State.IDLE)
    name = state.value if isinstance(state, State) else str(state)
    my_side = getattr(session, "my_side", RED)
    lines = [
        f"STATE: {name}",
        f"SIDE : {'RED' if my_side == RED else 'BLACK'}",
    ]
    if rec is not None:
        lines.append(f"LOW  : {len(rec.unknown)} cells")
    lines.append(f"MSG  : {to_ascii(getattr(session, 'message', '') or '')}")

    analysis = getattr(session, "last_analysis", None)
    if analysis is not None:
        if getattr(analysis, "mate_in", None) is not None:
            score = f"mate {analysis.mate_in:+d}"
        elif getattr(analysis, "score_cp", None) is not None:
            score = f"{analysis.score_cp:+d} cp"
        else:
            score = "?"
        lines.append(
            f"BEST : {getattr(analysis, 'bestmove', '')}  {score}  d{getattr(analysis, 'depth', 0)}"
        )
        pv = list(getattr(analysis, "pv", []) or [])[:8]
        if pv:
            lines.append(f"PV   : {' '.join(pv)}")

    board = getattr(session, "board", None)
    fen = board.to_fen() if board is not None else "(none)"
    for i in range(0, len(fen), _FEN_WRAP):
        prefix = "FEN  : " if i == 0 else "       "
        lines.append(prefix + fen[i : i + _FEN_WRAP])

    lines.append("")
    lines.append("F8 pause  F9 step  F10 dry-run  ESC quit")
    return lines


def compose(
    frame: np.ndarray | None,
    geom: BoardGeometry | None,
    rec: Recognition | None,
    session: Any,
    width: int = 1500,
    height: int = 900,
) -> np.ndarray:
    """把一帧画面、识别结果与会话状态合成成一张调试图。"""
    if width < 64 or height < 64:
        raise ValueError(f"面板尺寸太小：{width}x{height}")
    canvas = np.full((height, width, 3), _COLOR_BG, np.uint8)
    split = int(width * _LEFT_RATIO)
    body_h = max(1, height - _STATUS_H)
    _draw_board(canvas[:body_h, :split], frame, geom, rec, session)
    _draw_info(canvas[:body_h, split:], session, rec)
    _draw_status(canvas[body_h:], session)
    return canvas


def _draw_board(
    canvas: np.ndarray,
    frame: np.ndarray | None,
    geom: BoardGeometry | None,
    rec: Recognition | None,
    session: Any,
) -> None:
    h, w = canvas.shape[:2]
    if frame is None:
        cv2.putText(canvas, "(no frame)", (12, 26), _FONT, 0.6, _COLOR_DIM, 1, cv2.LINE_AA)
        return
    if frame.ndim != 3 or frame.shape[2] != 3:
        cv2.putText(canvas, "(bad frame)", (12, 26), _FONT, 0.6, _COLOR_RED, 1, cv2.LINE_AA)
        return

    # 有几何时只显示棋盘那一块：1440x2560 的整帧缩到 900 高之后棋盘只有 ~340px 宽，
    # 棋子字符会小到看不清。留 1.2 格边距，保证「棋盘外的异常」仍能瞥见一点。
    view, cut_x, cut_y = frame, 0, 0
    if geom is not None and geom.is_plausible():
        margin = geom.cell_px * 1.2
        cut_x = max(0, int(geom.origin[0] - margin))
        cut_y = max(0, int(geom.origin[1] - margin))
        stop_x = min(frame.shape[1], int(geom.origin[0] + (BOARD_COLS - 1) * geom.dx + margin))
        stop_y = min(frame.shape[0], int(geom.origin[1] + (BOARD_ROWS - 1) * geom.dy + margin))
        if stop_x > cut_x and stop_y > cut_y:
            view = frame[cut_y:stop_y, cut_x:stop_x]

    scale = min(w / view.shape[1], h / view.shape[0])
    tw = max(1, int(view.shape[1] * scale))
    th = max(1, int(view.shape[0] * scale))
    small = cv2.resize(
        view, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    ox, oy = (w - tw) // 2, (h - th) // 2
    canvas[oy : oy + th, ox : ox + tw] = small

    if geom is None or rec is None:
        return
    mapper = ScreenMapper(getattr(session, "my_side", RED))
    unknown = set(rec.unknown)
    for row in range(rec.confidence.shape[0]):
        for col in range(rec.confidence.shape[1]):
            x, y = geom.cell_to_px(row, col)
            px = round(ox + (x - cut_x) * scale)
            py = round(oy + (y - cut_y) * scale)
            if not (0 <= px < w and 0 <= py < h):
                continue
            color = conf_color(float(rec.confidence[row, col]))
            cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)
            if (row, col) in unknown:
                cv2.drawMarker(canvas, (px, py), _COLOR_RED, cv2.MARKER_TILTED_CROSS, 16, 2)
                continue
            rank, file = mapper.to_board(row, col)
            char = PIECE_CHARS.get(int(rec.board.at(rank, file)), "?")
            if char != ".":
                # 棋子字符用「黑描边 + 白字」，因为背景是木色棋盘，直接用彩色画会看不清。
                # 置信度由圆点的颜色表达，字符只负责说「认成了什么子」。
                at = (px + 6, py - 6)
                cv2.putText(canvas, char, at, _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, char, at, _FONT, 0.55, _COLOR_TEXT, 1, cv2.LINE_AA)


def _draw_info(canvas: np.ndarray, session: Any, rec: Recognition | None) -> None:
    y = _LINE_H
    for line in status_lines(session, rec):
        if y > canvas.shape[0] - 4:
            break
        if line:
            cv2.putText(canvas, line, (10, y), _FONT, 0.52, _COLOR_TEXT, 1, cv2.LINE_AA)
        y += _LINE_H


def _draw_status(canvas: np.ndarray, session: Any) -> None:
    canvas[:] = _COLOR_STATUS_BG
    state = getattr(session, "state", State.IDLE)
    if state is State.GAME_OVER:
        label, color = "GAME OVER", _COLOR_YELLOW
    elif state is State.PAUSED:
        label, color = "PAUSED", _COLOR_RED
    else:
        label, color = "RUNNING", _COLOR_GREEN

    mouse = getattr(getattr(session, "deps", None), "mouse", None)
    if getattr(mouse, "dry_run", False):
        label += "  |  DRY-RUN"

    cv2.putText(canvas, label, (10, canvas.shape[0] - 10), _FONT, 0.55, color, 1, cv2.LINE_AA)
    msg = to_ascii(getattr(session, "message", "") or "")
    if msg:
        cv2.putText(
            canvas, msg, (240, canvas.shape[0] - 10), _FONT, 0.5, _COLOR_DIM, 1, cv2.LINE_AA
        )


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _find_own_window() -> int | None:
    """找本进程创建的可见顶层窗口（也就是 OpenCV 的高亮窗口）。

    按 PID 找而不是按标题找：标题此刻还是乱码，按标题匹配太脆。
    """
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
    except OSError:  # pragma: no cover - 非 Windows 才会到这
        return None
    pid = os.getpid()
    found: list[int] = []

    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(_WNDENUMPROC(cb), 0)
    return found[0] if found else None


def _set_window_caption(title: str) -> bool:
    """把高亮窗口的标题改成 `title`（UTF-16，绕开 OpenCV 的 ANSI 路径）。"""
    hwnd = _find_own_window()
    if hwnd is None:
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetWindowTextW(wintypes.HWND(hwnd), title)
    except OSError:  # pragma: no cover - 非 Windows 才会到这
        return False
    return True


class Panel:
    """把 `compose()` 的结果显示在窗口里。`enabled=False` 时只算不显示。"""

    def __init__(
        self, enabled: bool = True, width: int = 1500, height: int = 900, title: str = _TITLE
    ) -> None:
        self.enabled = enabled
        self.width = width
        self.height = height
        self.title = title
        self.frames = 0
        self._caption_done = False

    def render(
        self,
        frame: np.ndarray | None,
        geom: BoardGeometry | None,
        rec: Recognition | None,
        session: Any,
    ) -> np.ndarray:
        out = compose(frame, geom, rec, session, self.width, self.height)
        self.frames += 1
        if self.enabled:
            cv2.imshow(_WINDOW_KEY, out)
            if not self._caption_done:
                # 第一次 imshow 才真正建出窗口，标题只能在这之后补。
                self._caption_done = _set_window_caption(self.title)
            cv2.waitKey(1)
        return out

    def close(self) -> None:
        """关窗口。可以重复调用 —— 窗口没开过、或者用户已经自己关掉了，都不该报错。"""
        if not self.enabled:
            return
        try:
            cv2.destroyWindow(_WINDOW_KEY)
        except cv2.error:
            pass
        self._caption_done = False
        cv2.waitKey(1)
