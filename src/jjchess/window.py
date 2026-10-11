r"""JJ象棋窗口的发现与抓图（纯 ctypes，不引入第三方依赖）。

抓图的唯一可行方案是 **`PrintWindow(hwnd, hdc, 2)`（`PW_RENDERFULLCONTENT`）**，
实测结论：

- `PrintWindow(..., 0)` → **全白**（Chromium / DirectComposition 窗口的典型表现）；
- `GetWindowDC` + `BitBlt` → 有内容，但**只有屏幕内可见的那部分**，窗口顶部
  279px 在屏幕外时抓不到；
- `PrintWindow(..., 2)` → 完整 1440x2560 彩色画面，**含屏幕外部分**。

窗口是微信电脑版的小程序宿主（进程 `WeChatAppEx`），不是模拟器，所以不需要 adb。

`PrintWindow` 也会**零星**返回两种不可用的帧，抓完图必须过 `frame_usable`：
整帧全黑/全白的**空白帧**（`is_blank_frame`），以及右下角整块未初始化的**残缺帧**
（`is_torn_frame`）。两者都会被识别层误读成「棋盘上的子没了」。

Win64 上句柄是 64 位，**所有 API 都必须显式声明 `argtypes` / `restype`**，
不声明会被 ctypes 按 `c_int` 截断。

⚠️ **进程完整性级别是硬前提。** 若本进程跑在 Low 完整性级别，`PrintWindow`
对任何 Medium 级别的窗口都返回 `ERROR_ACCESS_DENIED`（错误码 5），抓到全零。
实测原因：项目目录被 DSH 沙箱标了 `Mandatory Label\Low Mandatory Level`，
目录里所有可执行文件因此都以 Low 级别启动（`.venv\Scripts\python.exe` 亦然）。
修复命令（已执行，见 README「运行前提」）：

    icacls "<项目目录>" /setintegritylevel Medium

判断方法：`src/jjchess/control.py` 里的鼠标注入同理受 UIPI 限制，也依赖这一条。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

_HGDIOBJ = getattr(wintypes, "HGDIOBJ", wintypes.HANDLE)
_HBITMAP = getattr(wintypes, "HBITMAP", wintypes.HANDLE)

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

_HAS_DPI = hasattr(user32, "GetDpiForWindow")
if _HAS_DPI:
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.GetDpiForWindow.restype = wintypes.UINT

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = _HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, _HGDIOBJ]
gdi32.SelectObject.restype = _HGDIOBJ
gdi32.DeleteObject.argtypes = [_HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL
gdi32.GdiFlush.argtypes = []
gdi32.GdiFlush.restype = wintypes.BOOL

#: `PrintWindow` 的取值：只有 2 能抓到 DirectComposition 窗口的完整内容。
PW_RENDERFULLCONTENT = 2
DIB_RGB_COLORS = 0
BI_RGB = 0


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(_BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = _HBITMAP


class WindowNotFound(RuntimeError):
    """找不到（或已经关掉）JJ象棋窗口。"""


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _visible_windows() -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            title = _window_title(hwnd)
            if title:
                found.append((int(hwnd), title))
        return True

    user32.EnumWindows(_WNDENUMPROC(_cb), 0)
    return found


def _dpi_of(hwnd: int) -> int:
    if not _HAS_DPI:
        return 96
    try:
        dpi = int(user32.GetDpiForWindow(hwnd))
    except OSError:
        return 96
    return dpi if dpi > 0 else 96


def find_window(title: str = "JJ象棋", timeout_s: float = 0.0) -> GameWindow:
    """按标题找窗口，返回一个 `GameWindow`；超时抛 `WindowNotFound`。

    先精确匹配标题；没有精确命中的再退化成子串匹配。同类候选多于一个时
    挑客户区面积最大的那个——微信的小程序宿主会同时存在若干辅助窗口。
    """
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    while True:
        candidates = _visible_windows()
        exact = [hwnd for hwnd, text in candidates if text == title]
        if not exact and title:
            exact = [hwnd for hwnd, text in candidates if title in text]
        if exact:
            best = max(exact, key=lambda h: _client_area(h))
            return GameWindow.of(best, _window_title(best))
        if time.monotonic() >= deadline:
            break
        time.sleep(0.15)

    raise WindowNotFound(
        f"未找到标题为 {title!r} 的窗口："
        "请确认 JJ象棋 已在微信里打开、窗口没有最小化，且标题没被改过"
    )


def _client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise WindowNotFound("读取窗口客户区尺寸失败，窗口可能已经关闭")
    return (rect.left, rect.top, rect.right, rect.bottom)


def _client_area(hwnd: int) -> int:
    try:
        left, top, right, bottom = _client_rect(hwnd)
    except WindowNotFound:
        return -1
    return max(right - left, 0) * max(bottom - top, 0)


def _to_gray(img: np.ndarray | None) -> np.ndarray | None:
    """转成 float32 灰度图；空输入返回 `None`。"""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        return arr[..., :3].astype(np.float32).mean(axis=2)
    return arr.astype(np.float32)


def is_blank_frame(img: np.ndarray | None) -> bool:
    """判断是不是"什么都没抓到"的空白帧。

    最小化、被完全遮挡、或 `PrintWindow` 失败时会拿到纯色图。**这种帧绝不能
    当成「32 子全空」**，否则会走出一堆非法着法。
    """
    gray = _to_gray(img)
    if gray is None:
        return True
    # 纯色（含纯白、纯黑）→ 空白。
    if float(gray.std()) < 1.0:
        return True
    # 近全白（压缩噪点会留几个杂点）也算空白。
    return float((gray >= 254.0).mean()) >= 0.99


#: 「残缺帧」判据：近黑像素占比超过这个值就认为 PrintWindow 只画了一部分。
#: 实测（灰度 < 10 的像素占比）：好帧 ~1.1%（`lobby.png` 0.0111 / `opening.png` 0.0113），
#: 残缺帧 ~30%（`torn_frame.png` 0.2993）。27 倍的间隔，取 0.10 很稳。
TORN_DARK_RATIO = 0.10
TORN_DARK_LEVEL = 10.0


def is_torn_frame(img: np.ndarray | None) -> bool:
    """判断是不是 PrintWindow 只画了一部分的"残缺帧"。

    真机实测（2026-10-11）：`PrintWindow(..., 2)` 偶尔会返回一张**右下角整块全黑**
    的画面（未初始化的位图区），其余部分完全正常。它不是空白帧（平均亮度 88.66，
    远不到纯色判据），所以 `is_blank_frame` 拦不住它。

    危害比空白帧更隐蔽：黑区里的格子**看起来是空的**，于是整盘棋像是凭空少了一半
    子力 —— 脚本会以为对方刚吃掉自己七八个子。所以抓完图必须过 `frame_usable`。
    """
    gray = _to_gray(img)
    if gray is None:
        return True
    return float((gray < TORN_DARK_LEVEL).mean()) >= TORN_DARK_RATIO


def frame_usable(img: np.ndarray | None) -> bool:
    """这一帧能不能信 —— 抓图之后、识别之前的统一闸门。

    空白帧与残缺帧都要丢掉重抓。返回 `False` 时**不要**把帧传给识别层。
    """
    return not is_blank_frame(img) and not is_torn_frame(img)


@dataclass
class GameWindow:
    """一个已经定位到的游戏窗口。"""

    hwnd: int
    title: str
    client_size: tuple[int, int]
    client_origin: tuple[int, int]
    dpi: int

    @classmethod
    def of(cls, hwnd: int, title: str) -> GameWindow:
        left, top, right, bottom = _client_rect(hwnd)
        point = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(point))
        return cls(
            hwnd=int(hwnd),
            title=title,
            client_size=(right - left, bottom - top),
            client_origin=(int(point.x), int(point.y)),
            dpi=_dpi_of(hwnd),
        )

    # ---------------------------------------------------------------- 状态

    def is_alive(self) -> bool:
        return bool(user32.IsWindow(self.hwnd))

    def size_changed(self, w: int, h: int) -> bool:
        """给定尺寸与缓存的客户区尺寸是否不一致。

        调用方在每帧抓图后拿 `(img.shape[1], img.shape[0])` 来问一次：返回
        `True` 说明窗口被拖动/改过尺寸，缓存的标定与 `BoardGeometry` 已经失效。
        """
        return (int(w), int(h)) != self.client_size

    def refresh(self) -> None:
        """重新查询几何。窗口已关闭则抛 `WindowNotFound`。"""
        if not self.is_alive():
            raise WindowNotFound(f"窗口 {self.title!r} 已关闭")
        left, top, right, bottom = _client_rect(self.hwnd)
        point = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(self.hwnd, ctypes.byref(point)):
            raise WindowNotFound(f"窗口 {self.title!r} 的客户区原点取不到")
        self.client_size = (right - left, bottom - top)
        self.client_origin = (int(point.x), int(point.y))
        self.dpi = _dpi_of(self.hwnd)

    # ---------------------------------------------------------------- 抓图

    def grab(self) -> np.ndarray:
        """抓客户区全图，BGR `(H, W, 3) uint8`，**含屏幕外的部分**。

        每次都按**当前**客户区尺寸分配位图（不读缓存的 `client_size`），这样
        窗口刚被改过尺寸也不会拿到错位的画面；缓存要不要更新由 `refresh()`
        决定，`size_changed()` 靠这个差值来判断标定是否失效。

        ⚠️ 返回的帧**还要过一遍 `frame_usable`**：本方法只在「`PrintWindow`
        返回 0 且抓到空白」时才抛异常，删残缺帧（`is_torn_frame`）是调用方的事。
        """
        if not self.is_alive():
            raise WindowNotFound(f"窗口 {self.title!r} 已关闭")
        left, top, right, bottom = _client_rect(self.hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise WindowNotFound(f"窗口 {self.title!r} 的客户区尺寸无效：{width}x{height}")

        # **必须用屏幕 DC + `CreateDIBSection`，不能拿 `GetWindowDC` 配
        # `CreateCompatibleBitmap`** —— 后者在 DirectComposition 窗口上会得到
        # 一张空位图（`PrintWindow` 照样返回 0，抓出来全白）。实测就是这个差异。
        hdc_screen = user32.GetDC(None)
        if not hdc_screen:
            raise WindowNotFound("取屏幕 DC 失败")
        hdc_mem = None
        hbmp = None
        old = None
        bits = ctypes.c_void_p()
        try:
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            if not hdc_mem:
                raise WindowNotFound("创建内存 DC 失败")

            info = _BITMAPINFO()
            header = info.bmiHeader
            header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            header.biWidth = width
            header.biHeight = -height  # 负值 → top-down，行序与图像一致
            header.biPlanes = 1
            header.biBitCount = 32
            header.biCompression = BI_RGB

            hbmp = gdi32.CreateDIBSection(
                hdc_screen, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(bits), None, 0
            )
            if not hbmp or not bits:
                raise WindowNotFound("创建 DIB 位图失败")
            old = gdi32.SelectObject(hdc_mem, hbmp)

            # `PrintWindow` 在 DirectComposition 窗口上**经常返回 0 但照样画对了**
            # （它只是表示窗口过程没处理 WM_PRINT）。所以返回值只当提示，真正的
            # 判据是抓到的像素：返回 0 **且**抓到空白才认为是真的失败了。
            painted = bool(user32.PrintWindow(self.hwnd, hdc_mem, PW_RENDERFULLCONTENT))
            gdi32.GdiFlush()
        finally:
            if old is not None and hdc_mem:
                gdi32.SelectObject(hdc_mem, old)
                old = None
            # 位图数据指针在 DeleteObject 之后失效，所以要在 finally 里把像素
            # 抄出来——但抄的动作放到 try 成功之后做，见下面的 raw。
            if hdc_mem:
                gdi32.DeleteDC(hdc_mem)
            if hdc_screen:
                user32.ReleaseDC(None, hdc_screen)

        # `bits` 指向 DIB 的像素区。位图对象还没删，直接按地址读。
        raw = ctypes.cast(bits, ctypes.POINTER(ctypes.c_ubyte * (width * height * 4))).contents
        bgra = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(height, width, 4)
        if hbmp:
            gdi32.DeleteObject(hbmp)

        frame = np.ascontiguousarray(bgra[:, :, :3])
        if not painted and is_blank_frame(frame):
            raise WindowNotFound(
                "抓到的画面是空白的（PrintWindow 未渲染）；"
                "JJ象棋 窗口可能被最小化了，请恢复显示后重试"
            )
        return frame
