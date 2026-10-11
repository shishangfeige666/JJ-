"""全局热键：轮询 `GetAsyncKeyState`，只在按下沿触发。

为什么不用 `RegisterHotKey` / 窗口消息：本脚本的主循环是「抓图 → 识别 → 决策 →
点击」的同步循环，没有消息泵。轮询一次四个键的成本可忽略，而且不需要窗口、
不需要注册，脚本被 Ctrl+C 打断时也不会漏掉注销。
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable

__all__ = ["FakeHotkeys", "Hotkeys"]

#: 键名 → 虚拟键码。`dryrun` 是切换「彩排/实点」，不是「开始」。
_KEYS: dict[str, int] = {
    "pause": 0x77,  # F8
    "step": 0x78,  # F9
    "dryrun": 0x79,  # F10
    "quit": 0x1B,  # ESC
}

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short


def _key_down(vk: int) -> bool:
    """只看最高位（当前是否按下）。

    低位那个「上次调用后曾被按过」的标志位**不能用** —— 它会让一次按键在
    之后的每一帧都重复上报，F8 会疯狂翻转暂停状态。
    """
    return (_user32.GetAsyncKeyState(vk) & 0x8000) != 0


class Hotkeys:
    """真实热键源。`is_down` 可注入，测试不必真的按键。"""

    def __init__(self, is_down: Callable[[int], bool] | None = None) -> None:
        self._is_down = _key_down if is_down is None else is_down
        self._held: set[str] = set()

    def poll(self) -> set[str]:
        """返回本帧**新按下**的键名集合（按住不放不会重复返回）。"""
        events: set[str] = set()
        for name, vk in _KEYS.items():
            if self._is_down(vk):
                if name not in self._held:
                    events.add(name)
                self._held.add(name)
            else:
                self._held.discard(name)
        return events


class FakeHotkeys:
    """按脚本发键的替身：每次 `poll()` 消费一项，一项就是「这一帧按了哪些键」。"""

    def __init__(self, script: list[str | set[str] | None]) -> None:
        self._script = [_normalize(item) for item in script]
        self._index = 0

    def poll(self) -> set[str]:
        if self._index >= len(self._script):
            return set()
        item = self._script[self._index]
        self._index += 1
        return set(item)


def _normalize(item: str | set[str] | None) -> set[str]:
    if item is None:
        return set()
    if isinstance(item, str):
        return {item}
    return set(item)
