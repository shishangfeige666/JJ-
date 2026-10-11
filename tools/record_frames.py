"""从真实 JJ象棋 窗口采集测试用画面帧。

用法：

    uv run python tools/record_frames.py --name lobby --count 1
    uv run python tools/record_frames.py --name mid --count 4 --interval 20 --watch

采集到的 PNG 落到 `tests/fixtures/`，供 Task 11（棋盘定位）与 Task 13（棋子识别）
的测试当输入。**棋盘类帧还需要人工写一份同名 `.fen`**，把图里的局面逐格记下来，
那就是识别测试的期望值来源。

`--watch` 会持续盯着窗口，**画面一变就存一份带时间戳的帧**，用于自动收集中局。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jjchess.window import GameWindow, WindowNotFound, find_window  # noqa: E402

DEFAULT_OUT = Path("tests/fixtures")
WATCH_INTERVAL = 0.5
# 画面变化判据：缩到 1/8 后灰度平均绝对差超过这个值就算变了。
_WATCH_DIFF_THRESHOLD = 1.5


def _save(img: np.ndarray, path: Path) -> None:
    """写 PNG。走 imencode 而不是 imwrite，避免 OpenCV 在非 ASCII 路径上静默失败。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败：{path}")
    path.write_bytes(buf.tobytes())


def _describe(img: np.ndarray) -> str:
    from jjchess.window import is_blank_frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flag = "空白帧！" if is_blank_frame(img) else "有画面"
    return f"{img.shape[1]}x{img.shape[0]}  平均亮度={gray.mean():6.2f}  {flag}"


def _fingerprint(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (180, 320))
    return small.astype(np.int16)


def _run_watch(win: GameWindow, args: argparse.Namespace) -> int:
    print(f"监视中：每 {WATCH_INTERVAL}s 抓一帧，画面变化就存一份。Ctrl+C 结束。")
    prev: np.ndarray | None = None
    saved = 0
    try:
        while True:
            img = win.grab()
            fp = _fingerprint(img)
            changed = prev is None or float(np.abs(fp - prev).mean()) > _WATCH_DIFF_THRESHOLD
            if changed:
                stamp = time.strftime("%H%M%S")
                path = args.out / f"{args.name}_{stamp}.png"
                _save(img, path)
                saved += 1
                print(f"  [{saved:03d}] 画面变化 → {path}  {_describe(img)}")
                prev = fp
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n结束：共存下 {saved} 帧，在 {args.out}")
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从真实 JJ象棋 窗口采集测试帧")
    ap.add_argument("--name", default="frame", help="文件名前缀（opening / lobby / mid …）")
    ap.add_argument("--count", type=int, default=1, help="抓多少帧；>1 时文件名带 _01 序号")
    ap.add_argument("--interval", type=float, default=0.5, help="多帧之间的间隔秒数")
    ap.add_argument("--watch", action="store_true", help="持续监视，画面一变就存一帧")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    args = ap.parse_args(argv)

    try:
        win = find_window("JJ象棋", timeout_s=5.0)
    except WindowNotFound as exc:
        print(f"找不到游戏窗口：{exc}", file=sys.stderr)
        return 1
    print(f"窗口：{win}")

    if args.watch:
        return _run_watch(win, args)

    for i in range(args.count):
        if i:
            time.sleep(args.interval)
        img = win.grab()
        name = f"{args.name}.png" if args.count == 1 else f"{args.name}_{i + 1:02d}.png"
        path = args.out / name
        _save(img, path)
        print(f"  {path}  {_describe(img)}")

    print(f"\n完成：{args.count} 帧已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
