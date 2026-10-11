"""抓图环境自检：确认本进程能真的抓到 JJ象棋 窗口的画面。

用法：

    uv run python tools/check_capture.py

排查抓图问题时**先跑这个**。它刻意只用 `ctypes` + `struct`，不 import `jjchess`、
不依赖 numpy，因此在 `.venv` 坏掉、依赖没装好、甚至换别的 Python 版本时都能跑，
可以把「环境问题」和「代码问题」干净地分开。

最容易踩的坑是**进程完整性级别**：如果本进程是 Low，`PrintWindow` 对任何
Medium 级别的窗口都返回 0 且 `GetLastError() == 5 (ERROR_ACCESS_DENIED)`。
详见 README 的「运行前提」。
"""

from __future__ import annotations

import ctypes
import struct
import sys
from ctypes import wintypes

TARGET_TITLE = "JJ象棋"
ERROR_ACCESS_DENIED = 5
PROBE_W, PROBE_H = 400, 300

u = ctypes.WinDLL("user32", use_last_error=True)
g = ctypes.WinDLL("gdi32", use_last_error=True)

u.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
u.PrintWindow.restype = wintypes.BOOL
u.GetDC.argtypes = [wintypes.HWND]
u.GetDC.restype = wintypes.HDC
u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
u.ReleaseDC.restype = ctypes.c_int
u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
u.GetWindowTextW.restype = ctypes.c_int
u.IsWindowVisible.argtypes = [wintypes.HWND]
u.IsWindowVisible.restype = wintypes.BOOL
u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
u.GetWindowThreadProcessId.restype = wintypes.DWORD
u.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
u.EnumWindows.restype = wintypes.BOOL

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL

advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
advapi32.GetTokenInformation.restype = wintypes.BOOL

g.CreateCompatibleDC.argtypes = [wintypes.HDC]
g.CreateCompatibleDC.restype = wintypes.HDC
g.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.c_void_p,
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]
g.CreateDIBSection.restype = wintypes.HANDLE
g.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
g.SelectObject.restype = wintypes.HANDLE
g.DeleteObject.argtypes = [wintypes.HANDLE]
g.DeleteObject.restype = wintypes.BOOL
g.DeleteDC.argtypes = [wintypes.HDC]
g.DeleteDC.restype = wintypes.BOOL
g.GdiFlush.restype = wintypes.BOOL

_TOKEN_QUERY = 0x0008
_TOKEN_INTEGRITY_LEVEL = 25  # TokenIntegrityLevel

_INTEGRITY_NAMES = {
    0x0000: "Untrusted",
    0x1000: "Low",
    0x2000: "Medium",
    0x2100: "Medium Plus",
    0x3000: "High",
    0x4000: "System",
    0x5000: "Protected",
}


def title_of(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    u.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def integrity_level() -> int | None:
    """返回本进程的完整性级别 RID（0x1000 = Low，0x2000 = Medium）。"""
    token = wintypes.HANDLE()
    current_process = wintypes.HANDLE(-1)  # GetCurrentProcess() 伪句柄
    if not advapi32.OpenProcessToken(current_process, _TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, _TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token, _TOKEN_INTEGRITY_LEVEL, buf, size.value, ctypes.byref(size)
        ):
            return None
        # TOKEN_MANDATORY_LABEL { SID_AND_ATTRIBUTES { PSID Sid; DWORD Attributes; } }
        sid = ctypes.c_void_p.from_buffer(buf).value
        # SID 的第 8..11 字节是最后一个 SubAuthority（32 位小端），即 RID。
        return ctypes.c_uint32.from_address(sid + 8).value
    finally:
        k32.CloseHandle(token)


def probe(hwnd: int, label: str, w: int = PROBE_W, h: int = PROBE_H) -> tuple[bool, int, float]:
    """对 hwnd 跑一次真实的 PrintWindow(flag=2)，返回 (成功, 错误码, 平均亮度)。"""
    scr = u.GetDC(None)
    mem = g.CreateCompatibleDC(scr)
    bmi = ctypes.create_string_buffer(40 + 16)
    struct.pack_into("<IiiHHIIiiII", bmi, 0, 40, w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    hb = g.CreateDIBSection(scr, bmi, 0, ctypes.byref(bits), None, 0)
    old = g.SelectObject(mem, hb)
    ctypes.set_last_error(0)
    ok = u.PrintWindow(hwnd, mem, 2)
    err = ctypes.get_last_error()
    g.GdiFlush()
    raw = bytes(ctypes.cast(bits, ctypes.POINTER(ctypes.c_ubyte * (w * h * 4))).contents)
    mean = sum(raw[0::4]) / (w * h)
    g.SelectObject(mem, old)
    g.DeleteObject(hb)
    g.DeleteDC(mem)
    u.ReleaseDC(None, scr)
    print(f"  {label:<28} hwnd={hwnd:<10} ok={ok} err={err:<4} 平均亮度={mean:6.2f}")
    return bool(ok), err, mean


def main() -> int:
    print(f"Python {sys.version.split()[0]}  {ctypes.sizeof(ctypes.c_void_p) * 8} 位")

    rid = integrity_level()
    if rid is None:
        print("完整性级别：读取失败（不阻塞后续检查）")
    else:
        name = _INTEGRITY_NAMES.get(rid, "未知")
        print(f"完整性级别：{name}（RID=0x{rid:04X}）")

    targets: list[tuple[int, str]] = []
    if len(sys.argv) > 1:
        targets.append((int(sys.argv[1]), "命令行指定"))

    cb_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def collect(hwnd, _lparam):
        if u.IsWindowVisible(hwnd):
            t = title_of(hwnd)
            if t:
                targets.append((hwnd, t))
        return True

    u.EnumWindows(cb_type(collect), 0)

    print(f"\n对本机 {len(targets)} 个可见窗口各跑一次 PrintWindow(flag=2)：")
    game_hwnd = None
    results = []
    for hwnd, t in targets:
        ok, err, mean = probe(hwnd, t[:28])
        results.append((t, ok, err, mean))
        if t == TARGET_TITLE:
            game_hwnd = (hwnd, ok, err, mean)

    print()
    if game_hwnd is None:
        print(f"✗ 没找到标题为 {TARGET_TITLE!r} 的窗口。请确认 JJ象棋 已在微信里打开。")
        return 1

    hwnd, ok, err, mean = game_hwnd
    if ok and mean > 1.0:
        print(f"✓ 抓图正常：{TARGET_TITLE} 窗口 hwnd={hwnd}，平均亮度 {mean:.2f}（有真实画面）。")
        return 0

    print(f"✗ 抓图失败：{TARGET_TITLE} 窗口 hwnd={hwnd}，ok={ok} err={err} 平均亮度={mean:.2f}。")
    if err == ERROR_ACCESS_DENIED or (not ok and all(not r[1] for r in results)):
        print(
            "  错误码 5 = ERROR_ACCESS_DENIED，几乎所有窗口都抓不到 —— 本进程很可能跑在 Low 完整性级别。"
        )
        if rid is not None and rid < 0x2000:
            print("  已确认本进程就是 Low。请对项目目录执行一次：")
        else:
            print("  请在项目目录执行一次：")
        print('      icacls "<项目目录>" /setintegritylevel Medium')
        print("  然后重开终端再跑一遍本脚本。")
    else:
        print(f"  可能是 {TARGET_TITLE} 窗口被最小化了，恢复显示后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
