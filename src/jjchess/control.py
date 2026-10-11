"""鼠标执行：把 ICCS 着法变成两次真实的点击。

## 为什么需要「拟人化」

游戏会检测异常操作。一个每步都在固定毫秒内点到像素级同一点的账号，看起来就不像人。
所以这里做三件小事：落点加 ±`jitter_px` 的随机偏移、按下与抬起之间停 60~140ms、
起止两点之间停 180~350ms。**这些数字不是为了骗过谁，是为了让操作节奏落在人类范围内**
（spec §4.7）。思考层面的停顿 `--human-delay` 在 `session.py`，不在这里。

## 坐标怎么来的

    板坐标 (rank, file) --ScreenMapper(my_side).to_screen--> 屏幕格 (row, col) --cell_to_px--> 屏幕像素

`play_move` 收到的是 ICCS 字符串（如 `"h2e2"`），不是屏幕坐标。
`geom` 必须是**已经加过窗口屏幕偏移**的几何 —— 也就是 `BoardGeometry.origin` 是
屏幕绝对像素。`window.py` 给出的客户区原点在本机是 `(-1440, -279)`，
所以这一步不能想当然。

## 运行前提：进程完整性级别必须是 Medium

Low 完整性级别的进程受 UIPI 限制，`SendInput` 注入的鼠标事件会被**静默丢弃**（或直接
返回失败），表现是"脚本跑得好好的，就是棋盘不动"。这不是代码问题 —— 检查方法：

    uv run python tools/check_capture.py

它会打印当前进程的完整性级别。修复见 README「运行前提」一节。
`_send_mouse_input` 的返回值因此**必须检查**：宁可抛异常停下，也不要静默地空点一整局。

## 假设：进程 DPI 感知与窗口一致

`SetCursorPos` 收的是**屏幕物理像素**。本机显示器是 96 DPI（缩放 100%），
虚拟化不会发生，所以客户区坐标与画面像素一一对应。若将来换到缩放非 100% 的显示器，
需要先给进程设 DPI 感知上下文，否则坐标会被 Windows 悄悄缩放。
"""

from __future__ import annotations

import ctypes
import math
import random
import sys
import time
from ctypes import wintypes

from jjchess.geometry import RED, BoardGeometry, ScreenMapper
from jjchess.rules.moves import parse_move

__all__ = ["MOVE_GAP_S", "PRESS_HOLD_S", "Mouse"]

# 拟人化节奏（秒）
PRESS_HOLD_S = (0.06, 0.14)  # 按下 → 抬起
MOVE_GAP_S = (0.18, 0.35)  # 起点点击 → 终点点击

_INPUT_MOUSE = 0
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_DEFAULT_JITTER_PX = 6

_user32 = ctypes.WinDLL("user32", use_last_error=True)


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        # ULONG_PTR：x64 下必须是 8 字节，写成 ULONG(4) 会让整个 INPUT 少 8 字节，
        # SendInput 会以 ERROR_INVALID_PARAMETER 失败。
        ("dwExtraInfo", ctypes.c_void_p),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", _MouseInput),)


class _Input(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _InputUnion))


def _set_cursor_pos(x: int, y: int) -> bool:
    """把光标挪到屏幕物理像素 `(x, y)`。抽成模块级函数是为了测试能替换掉它。"""
    return bool(_user32.SetCursorPos(int(x), int(y)))


def _send_mouse_input(press: bool) -> int:
    """发一次左键按下 / 抬起。返回 `SendInput` 实际插入的事件数（成功时为 1）。

    用 `SetCursorPos` 定位、不带 `MOUSEEVENTF_ABSOLUTE` 让事件落在当前位置 ——
    绝对坐标那条路要先按虚拟桌面归一化到 `0..65535`，而本机虚拟桌面原点是
    `(-1440, -279)` 不是 `(0, 0)`，多一次换算就多一个错的机会。
    """
    event = _Input()
    event.type = _INPUT_MOUSE
    event.mi.dx = 0
    event.mi.dy = 0
    event.mi.mouseData = 0
    event.mi.dwFlags = _MOUSEEVENTF_LEFTDOWN if press else _MOUSEEVENTF_LEFTUP
    event.mi.time = 0
    event.mi.dwExtraInfo = None
    return int(_user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)))


def virtual_desktop_rect() -> tuple[int, int, int, int]:
    """虚拟桌面 `(left, top, width, height)`，用于自检与日志。"""
    metrics = (
        wintypes.UINT(0x4C),  # SM_XVIRTUALSCREEN
        wintypes.UINT(0x4D),  # SM_YVIRTUALSCREEN
        wintypes.UINT(0x4E),  # SM_CXVIRTUALSCREEN
        wintypes.UINT(0x4F),  # SM_CYVIRTUALSCREEN
    )
    return tuple(int(_user32.GetSystemMetrics(m.value)) for m in metrics)  # type: ignore[return-value]


def _check_coords(x: float, y: float) -> tuple[float, float]:
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"非法点击坐标：({x}, {y})")
    return x, y


def _check_on_screen(x: float, y: float) -> None:
    """坐标必须落在虚拟桌面里。

    `SetCursorPos` 越界**不会失败** —— 它把光标钳到虚拟桌面边缘然后返回 TRUE
    （本机实测 `SetCursorPos(99999, 99999)` 落点 `(-1, 2280)`，`GetLastError()` 是 0）。
    于是几何一旦算错（手工标定缩放、过期的几何、窗口被拖走），脚本照样"点成功"、
    照样记进 `mouse.clicks`，下一帧局面校验才失败 —— 现场表现是"点了没反应"。
    在**发出去之前**拦下来，报错信息才指得准。
    """
    left, top, width, height = virtual_desktop_rect()
    if not (left <= x < left + width and top <= y < top + height):
        raise ValueError(
            f"点击坐标 ({x:.0f}, {y:.0f}) 落在虚拟桌面 "
            f"[{left}, {top}, {left + width}, {top + height}] 之外 —— "
            "多半是棋盘几何算错了（窗口挪过位置，或手工标定的几何过期）。"
        )


class Mouse:
    """拟人化鼠标。

    `my_side` 决定 `ScreenMapper` 的翻转方向 —— 执黑时棋盘是 180° 翻转的，
    同一个 ICCS 着法要落到屏幕另一头的格子上。整局棋里它不会变，所以放在构造参数里。

    `clicks` 记录**每一次**实际点击的屏幕坐标（`dry_run` 时也记），
    面板与调试日志都读它；`--dry-run` 就是靠它做到"只说不做"。
    """

    def __init__(
        self, dry_run: bool = False, jitter_px: int = _DEFAULT_JITTER_PX, my_side: int = RED
    ) -> None:
        if my_side not in (RED, -RED):
            raise ValueError(f"非法执色：{my_side!r}（只能是 RED=1 或 BLACK=-1）")
        if jitter_px < 0:
            raise ValueError(f"jitter_px 不能是负数：{jitter_px}")
        self.dry_run = dry_run
        self.jitter_px = jitter_px
        self.my_side = my_side
        self.clicks: list[tuple[float, float]] = []

    # ---- 底层动作 ----

    def move_to(self, x: float, y: float) -> None:
        """只把光标挪过去，不点击。`dry_run` 下什么也不做（也不打印，免得刷屏）。"""
        x, y = _check_coords(x, y)
        if self.dry_run:
            return
        _check_on_screen(x, y)
        ix, iy = round(x), round(y)
        if not _set_cursor_pos(ix, iy):
            raise OSError(f"SetCursorPos({ix}, {iy}) 失败：{ctypes.get_last_error()}")

    def click_screen(self, x: float, y: float) -> None:
        """在**屏幕绝对像素** `(x, y)` 点一下左键。"""
        self._click(x, y, label=f"({x:.0f}, {y:.0f})")

    def click_client(self, x: float, y: float, origin: tuple[int, int]) -> None:
        """在**窗口客户区**坐标 `(x, y)` 点一下；`origin` 是客户区原点在屏幕上的位置。"""
        self._click(origin[0] + float(x), origin[1] + float(y), label=f"客户区 ({x:.0f}, {y:.0f})")

    def _click(self, x: float, y: float, label: str) -> None:
        x, y = _check_coords(x, y)
        self.clicks.append((x, y))
        if self.dry_run:
            print(f"[dry-run] 点击 {label}")
            return

        self.move_to(x, y)
        # 按下之后**无论怎么退出**都要补一次抬起，否则系统里左键一直停在按下状态，
        # 用户之后随便动一下鼠标就变成拖拽（而且多半拖在微信窗口上）。
        # 中间那个 60~140ms 的停顿就是 `Ctrl+C` 最可能落进来的窗口 —— 本脚本是长跑的
        # 循环程序，Ctrl+C 是常规退出方式。
        pressed = False
        try:
            if _send_mouse_input(True) != 1:
                raise OSError(
                    "SendInput 发送左键按下失败 —— 通常是本进程完整性级别低于游戏窗口"
                    "（UIPI 拦截）。跑 `uv run python tools/check_capture.py` 看完整性级别，"
                    "修复见 README「运行前提」。"
                )
            pressed = True
            time.sleep(random.uniform(*PRESS_HOLD_S))
        finally:
            if pressed and _send_mouse_input(False) != 1:
                # 这里故意不抛：抛出会盖掉真正的原因（多半是上面的 OSError 或
                # KeyboardInterrupt）。左键卡住是"已经发生的事"，报出来更有用。
                print(
                    "警告：SendInput 发送左键抬起失败，鼠标左键可能停在按下状态 —— "
                    "按一下左键即可复位。",
                    file=sys.stderr,
                )

    # ---- 走子 ----

    def play_move(
        self, mv: str, geom: BoardGeometry
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """把 ICCS 着法走成两次点击。

        `geom` 必须是**已加过屏幕偏移**的几何。返回实际点击的
        `(起点屏幕坐标, 终点屏幕坐标)`，供面板叠加与日志使用。
        """
        src, dst = parse_move(mv)  # 非法着法在这里就 ValueError
        mapper = ScreenMapper(self.my_side)

        start = self._jitter(*geom.cell_to_px(*mapper.to_screen(*src)))
        end = self._jitter(*geom.cell_to_px(*mapper.to_screen(*dst)))

        self.click_screen(*start)
        # dry_run 下不等 —— 彩排要的是快点把整局走完看坐标对不对，不是还原真实节奏
        if not self.dry_run:
            time.sleep(random.uniform(*MOVE_GAP_S))
        self.click_screen(*end)
        return start, end

    def _jitter(self, x: float, y: float) -> tuple[float, float]:
        if self.jitter_px == 0:
            return (x, y)
        return (
            x + random.uniform(-self.jitter_px, self.jitter_px),
            y + random.uniform(-self.jitter_px, self.jitter_px),
        )
