"""标定工具：给真机画面建识别模板，或在自动定位失败时手工标出棋盘位置。

## 为什么需要它

识别依赖两份"真机数据"：

1. **模板集**（`config/templates.npz`）—— 15 类棋子的标准外观。它不是随代码分发的
   常量，而是从**你机器上这一局的开局画面**里现抠出来的。理由：JJ象棋的棋子是
   游戏自带的贴图，换个版本、换台机器（DPI/缩放不同）像素就不一样了；现抠一份
   永远和当前画面严丝合缝。开局局面 32 个子齐全、14 类全都出现，而且每个交叉点
   的答案由 `START_FEN` 给定 —— 等于白送一份带标注的训练集。
2. **棋盘几何**（可选，`config/calibration.json`）—— 正常情况由
   `jjchess.vision.locator` 自动定位，不需要手工标。只有在投影峰值法失败
   （画面被裁、棋盘被道具挡住、换了很怪的分辨率）时才需要它兜底。

## 用法

    uv run python tools/calibrate.py --build      # 抓一帧开局画面，建模板集
    uv run python tools/calibrate.py --manual     # 鼠标点两个交叉点，反推几何
    uv run python tools/calibrate.py --check      # 只用现有模板集做一次全盘自检

`--build` 要求游戏正停在**开局局面**（推荐「人机对战」开局，红方在下方）。
`--manual` 会弹出一个窗口：先点棋盘**左上角**的交叉点，再点**右下角**的交叉点，
然后按任意键退出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jjchess.geometry import BoardGeometry, ScreenMapper
from jjchess.rules.board import PIECE_CHARS, RED, START_FEN, Board
from jjchess.vision import templates as T
from jjchess.vision.locator import BoardNotFound, locate
from jjchess.window import WindowNotFound, find_window, is_blank_frame

#: 手工标定的落盘位置。
DEFAULT_CALIBRATION = Path("config/calibration.json")
#: 模板总览图（人眼核对用）。
DEFAULT_PREVIEW = Path("config/templates_preview.png")
#: 识别结果叠图（人眼核对用）。
DEFAULT_RECOGNITION_PREVIEW = Path("config/recognition_preview.png")
#: 抓图重试次数 —— 微信小程序的画面偶尔会抓到空白帧。
GRAB_ATTEMPTS = 5
#: 低于这个置信度就在自检里点名（实测开局帧最低也有 0.798，0.75 是留了余量的告警线）。
CONF_FLOOR = 0.75
#: 叠图上的类名字号与线宽（画面是 1440x2560，0.5 号字根本看不清）。
LABEL_SCALE = 1.4
LABEL_THICKNESS = 3


def grab_frame(title: str) -> np.ndarray:
    """抓一帧真实画面；连续抓到空白帧就放弃并给出中文提示。

    **空白帧是真会出现的**：实测同一个窗口每 0.5 s 抓一次，会零星抓到整帧全黑
    （平均亮度 0.00），紧接着下一帧就恢复正常。所以抓图必须带自检和重试，
    绝不能把空白帧当成"棋盘上什么子都没有"往下传。
    """
    try:
        window = find_window(title, timeout_s=5.0)
    except WindowNotFound as exc:
        raise SystemExit(f"{exc}\n请先启动 JJ象棋（微信小程序）并让它停在棋盘界面。") from exc
    for attempt in range(1, GRAB_ATTEMPTS + 1):
        img = window.grab()
        if img is not None and not is_blank_frame(img):
            print(f"抓到画面：{img.shape[1]}x{img.shape[0]}（第 {attempt} 次尝试）")
            return img
        print(f"第 {attempt} 次抓到空白帧，重试……")
    raise SystemExit("连续抓到空白帧。请确认游戏窗口没有被最小化，且棋盘完整显示在屏幕上。")


def check(
    img: np.ndarray, geom: BoardGeometry, board: Board, mapper: ScreenMapper, title: str
) -> int:
    """按已知局面逐格核对识别结果，返回判错的格子数。

    只比对**名字**够不够？不够 —— 名字对了但置信度很低的格子，说明模板对该格
    本身就不适用（光照、遮挡），下一帧很可能翻车。所以低置信度单独列出来。
    最后再把结果画在画面上存成 `config/recognition_preview.png` 供人眼核对
    （画面上写的是 ASCII 类名，`cv2.putText` 画不了中文）。
    """
    suite = T.TemplateSet.load(T.DEFAULT_TEMPLATES_PATH)
    wrong: list[str] = []
    shaky: list[str] = []
    overlay = img.copy()
    for row in range(10):
        for col in range(9):
            rank, file = mapper.to_board(row, col)
            expected = PIECE_CHARS[board.at(rank, file)]
            got, conf = suite.match(T.make_patch(img, geom, row, col))
            if got != expected:
                wrong.append(f"  ({row},{col}) 期望 {expected} 实得 {got} 置信度 {conf:.3f}")
            elif conf < CONF_FLOOR:
                shaky.append(f"  ({row},{col}) {got} 置信度仅 {conf:.3f}")
            center_x, center_y = geom.cell_to_px(row, col)
            origin = (round(center_x) - 18, round(center_y) + 12)
            color = (
                (0, 0, 255)
                if got != expected
                else ((0, 165, 255) if conf < CONF_FLOOR else (0, 200, 0))
            )
            cv2.putText(
                overlay,
                got,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE,
                (0, 0, 0),
                LABEL_THICKNESS + 3,
            )
            cv2.putText(
                overlay, got, origin, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, color, LABEL_THICKNESS
            )

    print(f"【{title}】棋盘几何：{geom}")
    if wrong:
        print(f"识别自检：{len(wrong)}/90 格判错")
        print("\n".join(wrong))
    else:
        print("识别自检：90 格全部正确 ✓")
    if shaky:
        print(f"另有 {len(shaky)} 格置信度低于 {CONF_FLOOR}：")
        print("\n".join(shaky))
    ok, buffer = cv2.imencode(".png", overlay)
    if ok:
        DEFAULT_RECOGNITION_PREVIEW.write_bytes(buffer.tobytes())
        print(f"识别结果叠图：{DEFAULT_RECOGNITION_PREVIEW}（绿=正确、橙=置信度低、红=判错）")
    return len(wrong)


def build(title: str) -> int:
    """抓一帧开局画面 → 建模板集 → 立刻用它自检一遍。"""
    img = grab_frame(title)
    try:
        geom = locate(img)
    except BoardNotFound as exc:
        print(f"自动定位失败：{exc}")
        print("请先运行 `uv run python tools/calibrate.py --manual` 手工标定棋盘位置。")
        return 1

    board = Board.from_fen(START_FEN)
    mapper = ScreenMapper(RED)
    suite = T.build_templates(img, geom, board, mapper)
    suite.save(T.DEFAULT_TEMPLATES_PATH)
    suite.save_png(DEFAULT_PREVIEW)
    print(
        f"模板集已写入 {T.DEFAULT_TEMPLATES_PATH}（{suite.data.shape[0]} 类 {suite.data.shape[1]}x{suite.data.shape[2]}）"
    )
    print(f"模板总览图：{DEFAULT_PREVIEW}")

    wrong = check(img, geom, board, mapper, "开局帧")
    if wrong:
        print("模板自检没过 —— 请确认游戏停在**开局局面**且红方在下方。")
        return 1
    print("模板构建完成，可以开跑了。")
    return 0


def manual(title: str, out: Path) -> int:
    """鼠标点两个交叉点，反推 `BoardGeometry`。"""
    img = grab_frame(title)
    points: list[tuple[int, int]] = []
    display = img.copy()
    window_name = "calibrate - click top-left then bottom-right intersection"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        points.append((x, y))
        cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(
            display, str(len(points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)
    print("请依次点击：①棋盘左上角交叉点 ②棋盘右下角交叉点；然后按任意键结束。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 2:
        print(f"只收到 {len(points)} 个点，已取消。")
        return 1
    (x0, y0), (x1, y1) = points
    dx = (x1 - x0) / 8.0
    dy = (y1 - y0) / 9.0
    geom = BoardGeometry((float(x0), float(y0)), dx, dy, 0.85 * min(dx, dy))
    if not geom.is_plausible():
        print(f"这两个点推出来的几何不合理（dx={dx:.1f} dy={dy:.1f}）—— 是不是点错了？")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(geom.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"棋盘几何已写入 {out}")
    print(f"  {geom}")
    return 0


def reload_and_check(title: str, calib: Path) -> int:
    """只用现有模板集 + 现有几何做一次全盘自检（不重建）。"""
    if not T.DEFAULT_TEMPLATES_PATH.is_file():
        print(f"没有 {T.DEFAULT_TEMPLATES_PATH}，请先跑 --build。")
        return 1
    img = grab_frame(title)
    if calib.is_file():
        geom = BoardGeometry.from_json(json.loads(calib.read_text(encoding="utf-8")))
        print(f"使用手工标定的几何：{calib}")
    else:
        try:
            geom = locate(img)
        except BoardNotFound as exc:
            print(f"自动定位失败：{exc}")
            return 1
    return 1 if check(img, geom, Board.from_fen(START_FEN), ScreenMapper(RED), "自检") else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JJ象棋识别模板与棋盘几何标定")
    parser.add_argument("--build", action="store_true", help="抓一帧开局画面并重建模板集")
    parser.add_argument("--manual", action="store_true", help="鼠标点两个交叉点手工标定棋盘几何")
    parser.add_argument("--check", action="store_true", help="用现有模板集做一次全盘自检")
    parser.add_argument("--title", default="JJ象棋", help="游戏窗口标题（默认 JJ象棋）")
    parser.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION, help="手工标定的输出路径")
    args = parser.parse_args(argv)

    if args.manual:
        return manual(args.title, args.out)
    if args.build:
        return build(args.title)
    if args.check:
        return reload_and_check(args.title, args.out)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
