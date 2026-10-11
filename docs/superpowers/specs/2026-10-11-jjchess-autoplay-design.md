# JJ象棋 自动下棋脚本 —— 设计规格

- 日期：2026-10-11
- 状态：待评审
- 目标平台：Windows 11 + 微信电脑版小程序（WeChatAppEx.exe，窗口标题 `JJ象棋`）

---

## 1. 目标与成功标准

做一个 Python 程序，对微信小游戏「JJ象棋」实现**全自动托管下棋**：自动找到游戏窗口 → 自动识别棋盘上 32 个棋子的位置 → 用皮卡鱼（Pikafish）算出最佳着法 → 自动操作鼠标完成走子 → 循环直到对局结束；同时提供一个实时可视化面板用于观察识别与思考过程，并在对局结束后导出中文记谱。

**成功标准**

| # | 标准 | 验收方式 |
|---|---|---|
| S1 | 开局标准局面识别 32 子全对 | `tests/test_vision.py` 对 fixture 帧断言 100% |
| S2 | 中局任意合法局面识别正确 | 录制多帧 fixture 断言 100% |
| S3 | 规则层正确 | perft(1..4) = 44 / 1920 / 79666 / 3290240 |
| S4 | 皮卡鱼能算出正确杀棋着法 | `tests/test_uci.py` 用已知杀局断言 bestmove |
| S5 | 单步端到端（识别→思考→点击→校验）< 4 秒 | 运行时日志计时 |
| S6 | 全程可中断 | `F8` 暂停即时生效，`ESC` 退出 |
| S7 | 拟人化思考延迟生效 | `--human-delay` 开启后每步额外停顿落在 `[min_s, max_s]`；且**延迟期间 `F8`/`ESC` 仍即时响应** |
| S8 | 对局记录可导出 | 结束后生成中文记谱 txt，着法序列与对局完全一致，可被 `tests/test_record.py` 断言 |

**非目标（YAGNI，明确不做）**

- 不做残局闯关 / 揭棋 / 翻翻棋等其它玩法
- 不做多开、不做后台多账号
- 不做胜率统计、不做云库开局库
- 不做安卓模拟器 / 手机 adb 支持（本机是微信电脑版原生窗口）
- 不做标准 PGN 格式导出（象棋无统一 PGN 标准，只出中文记谱 + ICCS 行）

---

## 2. 已实测确认的环境事实

这些是本次调研**实际跑出来的结果**，不是假设：

| 项 | 实测值 |
|---|---|
| 游戏进程 / 窗口 | `WeChatAppEx.exe`，PID 29928，标题 `JJ象棋`，`hwnd=4983344` |
| 客户区尺寸 | **1440 × 2560**，DPI = 96（无缩放问题） |
| 窗口位置 | 屏幕坐标 `(-1440, -279)` —— **顶部 279px 在显示器外** |
| 显示器 | DISPLAY1 2560×1440 主屏；DISPLAY2 1440×2560 竖屏 |
| **抓图方法** | `PrintWindow(hwnd, hdc, 2)`（`PW_RENDERFULLCONTENT`）**实测可用**，返回完整彩色 1440×2560，含屏幕外部分 ✅ |
| 不可用方法 | `PrintWindow(hwnd, hdc, 0)` → 全白（Chromium/DirectComposition 窗口的典型行为）❌ |
| 兜底方法 | `GetWindowDC` + `BitBlt` → 可得，但**只能拿到屏幕内可见部分** |
| CPU | Intel i9-14900K，24 核 / 32 线程 |
| 引擎 | Pikafish **2026-09-06**（universal binary，自动探测 CPU 指令集）<br>`Pikafish.2026-09-06.7z` = 53,603,424 B<br>sha256 `41952bbfe2520faceb5902c69e6ab4845cc999841d2b49a95cc1be7867a25e5b` |
| **引擎已下载并跑通** | `vendor/pikafish/pikafish.exe`（6.94 MB）+ `vendor/pikafish/pikafish.nnue`（50.7 MB）<br>⚠️ **NNUE 权重是独立文件**，必须与 exe 同目录，否则引擎无法加载 |
| 引擎实测性能 | 开局 `go depth 18`：430 万节点 / **297 ms** = **14.5 Mnps**（16 线程） |
| 下载通道 | `api.github.com` 返回 **403**（本机有代理 `127.0.0.1:7897`）；release CDN `curl -r 0-1023` 返回 **HTTP 206** ✅ 但**直连仅 ~25 KB/s（需 35 分钟）**<br>镜像 `https://gh-proxy.com/<原始URL>` 实测 **21.6 MB/s**，51 MB **3 秒下完** ✅（配合 sha256 校验，镜像不可信也无妨） |
| 解压工具 | `bsdtar 3.6.2 - libarchive 3.6.2`（`C:\Windows\System32\tar.exe`），原生支持读 `.7z` ✅ |
| 阿里 PyPI 镜像 | `https://mirrors.aliyun.com/pypi/simple/` 实测 HTTP 200 ✅ |
| uv | `0.12.18` |
| 注意 | 系统 `python` 是 Anaconda 3.6.5（太老）→ 由 uv 固定独立 Python 3.12<br>PowerShell 是 **5.1**（非 pwsh 7），且 ExecutionPolicy **禁止直接运行 .ps1** |

> ⚠️ 副作用提醒：窗口顶部 279px 在屏幕外。抓图无影响（`PrintWindow` 拿得到），但**点击**只能点到屏幕内区域。棋盘完整落在屏幕内，因此走子不受影响；启动时会做一次几何校验，若棋盘范围越界则提示用户把窗口移回屏内。

---

## 3. 总体架构

```
                 ┌──────────────────────────────────────────┐
                 │              main.py  (主循环)            │
                 └──────────────────────────────────────────┘
                        │
   ┌────────────┬───────┴────────┬──────────────┬─────────────┐
   ▼            ▼                ▼              ▼             ▼
┌────────┐ ┌──────────┐   ┌───────────┐  ┌───────────┐ ┌──────────┐
│window  │ │ vision   │   │  rules    │  │  engine   │ │ control  │
│抓图     │→│ 定位+识别 │→ │ 规则/合法性│→ │ Pikafish  │→│ 鼠标点击  │
└────────┘ └──────────┘   └───────────┘  └───────────┘ └──────────┘
                 ▲              ▲                            │
                 │              │                            │
                 │        ┌─────┴──────┐                     │
                 └────────│  session   │◀────────────────────┘
                          │ 会话状态机  │  (局面校验 / 合法着法解释器)
                          └─────┬──────┘
                                ▼
                          ┌───────────┐
                          │ ui/panel  │  实时可视化面板 (cv2.imshow)
                          └───────────┘
```

数据流（单向）：

```
像素帧 ──locator──▶ BoardGeometry ──recognizer──▶ Board(np 10×9)
                                                      │
                          FEN ◀──rules/board───────────┘
                           │
                     engine/uci ──▶ "bestmove h2e2"
                           │
                  rules 校验该着法合法
                           │
              geometry.cell_to_px ──▶ control.click_px ×2
                           │
                    抓图复检 ──▶ 一致? ── 是 ──▶ 等对手 ──▶ 循环
                                     └─ 否 ──▶ 重试 1 次 ──▶ 仍否 ──▶ 熔断告警
```

---

## 4. 模块设计

### 4.1 `src/jjchess/window.py` —— 窗口发现与抓图

职责：找到游戏窗口并把它的客户区像素变成 numpy 数组。**不关心棋局**。

```python
@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    client_size: tuple[int, int]      # (w, h) = (1440, 2560)
    client_origin: tuple[int, int]    # 屏幕坐标 (-1440, -279)

def find_game_window(title_substr: str = "JJ象棋") -> WindowInfo: ...
class WindowCapture:
    def __init__(self, info: WindowInfo, method: Literal["printwindow","screen"] = "printwindow"): ...
    def grab(self) -> np.ndarray:      # (H, W, 3) uint8, BGR
        ...
```

- 用 `ctypes` 调 `EnumWindows` / `GetWindowTextW` / `GetClientRect` / `ClientToScreen`，**零额外依赖**
- `grab()` 实现：`CreateCompatibleDC` → `CreateDIBSection`(32bpp, top-down) → `PrintWindow(hwnd, memdc, 2)` → `np.frombuffer(bits, np.uint8).reshape(h, w, 4)[:, :, :3]`
- **全白/全黑帧自检**：若标准差 < 2 判定抓图失败，抛 `CaptureError`；调用方退避重试；连续 5 次失败则自动切 `method="screen"`（BitBlt）并告警
- 依赖：`numpy`、`ctypes`（stdlib）

### 4.2 `src/jjchess/geometry.py` —— 棋盘几何

职责：像素坐标 ↔ 棋盘坐标 的**唯一**换算入口。

```python
@dataclass(frozen=True)
class BoardGeometry:
    origin: tuple[float, float]   # 屏幕坐标下，交叉点 (row=0, col=0) 的中心像素
    dx: float                     # 相邻纵线间距（列方向）
    dy: float                     # 相邻横线间距（行方向）
    cell_px: float                # 棋子圆盘直径估计（用于点击抖动与 ROI 尺寸）

    def cell_to_px(self, row: int, col: int) -> tuple[float, float]: ...
    def px_to_cell(self, x: float, y: float) -> tuple[int, int]: ...   # 最近邻
    def offset_to_screen(self) -> "BoardGeometry": ...                 # 客户区坐标 → 屏幕坐标
    def save(self, path) / @classmethod load(cls, path): ...
```

棋盘坐标约定：`row ∈ [0,9]` **从上到下**，`col ∈ [0,8]` 从左到右（屏幕视角）。持久化到 `config/calibration.json`。

### 4.3 `src/jjchess/vision/locator.py` —— 棋盘自动定位

职责：从一帧图像自动求出 `BoardGeometry`。这是**免手工标定**的关键。

算法（投影峰值法）：

1. 灰度化，取中部 ROI：`y ∈ [0.15H, 0.80H]`、`x ∈ [0.01W, 0.99W]`（避开顶部头像栏与底部按钮栏）
2. `cv2.adaptiveThreshold` 取暗线掩膜（棋盘为浅木色，网格线为深棕）
3. **纵向投影**：每列统计暗像素数 → 平滑 → 取峰值 → 9 条纵线
4. **横向投影**：同上 → 10 条横线
5. **等距性校验**：计算相邻峰间距，若变异系数 > 5% 则用 RANSAC 式拟合剔除异常峰，重新等距排布
6. 输出 `origin = (vlines[0], hlines[0])`、`dx = mean(diff(vlines))`、`dy = mean(diff(hlines))`、`cell_px ≈ 0.85 * min(dx, dy)`

**失败兜底**：定位失败 → 提示"自动定位失败，请运行 `uv run python tools/calibrate.py` 手工标定"；手工标定让用户点击棋盘左上角与右下角两个交叉点即可反推 `origin/dx/dy`。

### 4.4 `src/jjchess/vision/templates.py` + `recognizer.py` —— 棋子识别

**棋类集合（15 类）**：`K帅/将 A仕/士 B相/象 N马 R车 C炮 P兵/卒` × `{r红, b黑}` + `.` 空点 = 15。
（用 FEN 标准字符 `B`/`N`，`E`/`H` 只作为 `PIECE_CHAR_ALIASES` 里的输入别名保留。）

**模板获取**（`tools/calibrate.py`，一次性）：

1. 让用户把游戏停在**开局标准局面**（或脚本自动检测到开局局面）
2. 按已知初始布局，在每个交叉点切 ROI，得到 14 类模板（同类多格 → 取**得分最高**的那一格，不是中位数）
3. 从未被占据的交叉点切 `empty` 模板（取多格中位数——棋盘木纹均匀，交叉点形态一致）
4. 落盘 `config/templates.npz`（键：类名 → `(72,72,3) uint8`）

**单格识别**：

```
ROI 边长 = round(1.30 * cell_px)   # 略大于圆盘，含少量木纹背景
→ cv2.resize 到 72×72
→ 灰度 + CLAHE 归一化（抵消选中高亮/环境光）
→ 对 15 个模板做 cv2.matchTemplate(TM_CCOEFF_NORMED) 取 1×1 响应
→ argmax
```

**判空用拉普拉斯方差，不是空模板相关**（原方案已实测证伪）：
先算这一格的拉普拉斯方差 `lap`，`lap < EMPTY_LAPLACIAN = 3000` 就判空
（`conf = 1/(1+(lap/EMPTY_LAPLACIAN)**2)`），否则再和 14 类棋子比。
实测空点 250~1149、棋子 8809~18627，差 7.7 倍；而「空点 vs 空模板」的相关性本身只有 0.65，
分数散落在 0.046~0.926，根本分不开。

**匹配窗口要缩到核心**：`MATCH_CORE_CELLS = 0.85`（只取中心 85% 做相关）。
棋子选中时的**光圈**正好落在 ROI 外圈，全尺寸匹配会被它严重干扰 ——
真机上量到「黑砲」那格 `top1 = 0.722 / top2 = 0.686`（只差 0.036，随时翻车），
缩到 0.85 后变成 `0.839 / 0.693`；两个 fixture 的分类正确率都是 1.000。
（缩到 0.95 差距掉回 0.103，1.05 以上光圈又进得来，所以取 0.85。）

**置信度与兜底**：`conf < 0.60` 标 `unknown`，置信度矩阵 `(10,9)` 一并返回；
面板上按 绿(>0.75)/黄(0.60~0.75)/红(<0.60) 着色标注。
阈值从最初的 0.75/0.90 下调，是因为实测棋子 top1 最低 0.798、空点 conf 最低 0.872，
原阈值会把每一格都标成不可信。

```python
@dataclass
class Recognition:
    board: "Board"
    confidence: np.ndarray        # (10, 9) float32
    unknown: list[tuple[int,int]]
```

### 4.5 `src/jjchess/rules/` —— 中国象棋规则

**`board.py`** —— `Board` = `np.int8 (10, 9)`。编码：`+1..+7` 红 `K A E H R C P`，`-1..-7` 黑，`0` 空。

XQ-FEN（标准，红在下，rank 0 在 FEN 最后一行）：

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
```

**`moves.py`** —— 走法生成与合法性：

- `generate(board, side) -> list[Move]`：车（直线，遇子止）、马（**蹩马腿**：先直后斜，直向邻格须空）、象/相（田字，**塞象眼**：田心须空，**不可过河**）、士/仕（九宫斜走）、将/帅（九宫直走，另加**将帅照面**判负）、炮（移动同车；吃子须**隔且仅隔一个炮架**）、兵/卒（向前；**过河后可横走**，永不后退）
- `is_attacked(board, sq, by_side)` / `in_check(board, side)`
- `is_legal(board, move)`：走完后己方不被将军
- `game_status(board, side)` → `ONGOING | CHECKMATE | STALEMATE | CHECK`
- `make(board, move)` / `unmake(board, move, captured)`（就地修改，供搜索与 perft 用）
- `Move` 用 ICCS 字符串表示：`"h2e2"`（file `a..i` + rank `0..9`）

**`notation.py`** —— ICCS ↔ 中文纵线记谱：`"h2e2"` → `"炮二平五"`，用于日志与面板可读性。

**性能取舍**：不做位棋盘（bitboard）。中国象棋 90 格，numpy 数组 + 预生成偏移表足够；perft(4) 目标秒级。真正的搜索交给皮卡鱼，本模块只做合法性校验与状态判定。

### 4.6 `src/jjchess/engine/uci.py` —— 皮卡鱼 UCI 封装

```python
@dataclass
class Analysis:
    bestmove: str          # "h2e2"
    score_cp: int | None   # 厘兵分
    mate_in: int | None
    depth: int
    sel_depth: int
    nps: int
    pv: list[str]          # 后续着法序列，ICCS

class UciEngine:
    def __init__(self, exe: Path, threads: int = 8, hash_mb: int = 512, multipv: int = 1): ...
    def __enter__/__exit__        # 进程生命周期
    def analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis: ...
    def quit(self) -> None: ...
```

- `subprocess.Popen`，`stdin/stdout` 文本管道，行缓冲
- 握手：`uci` → 等 `uciok`；`setoption name Threads value N`；`setoption name Hash value M`；`isready` → 等 `readyok`
- 求招：`position fen <fen>` + `go movetime <ms>`；持续读 `info ...` 行保留**最后一条**完整 `info`（含 `depth/score/pv`），读到 `bestmove <m>` 结束
- ⚠️ **实现硬约束（实测踩坑得出）**：**绝不能发完 `go` 就发 `quit`**。Pikafish 主线程收到 `quit` 会立即终止正在进行的搜索，导致只搜到 depth 1 就返回垃圾着法（实测 `bestmove a3a4`）。
  正确做法：**阻塞读 stdout 直到出现 `bestmove` 行**，再从该行取第一个 token（后可能跟 `ponder <move>`，须忽略）
- **严格校验（实测已确认）**：送非法 FEN（如将帅照面）→ 引擎**直接以 exit code 1 自杀**。因此送引擎前必须先过 `rules` 合法性检查；wrapper 需捕获进程意外退出并重启引擎（最多 3 次）
- **`EvalFile`**：引擎需同目录的 `pikafish.nnue`。握手后发 `setoption name EvalFile value <abs path>` 更稳妥（默认即按 exe 同目录查找）
- 坐标映射（**关键**，见 §5.3）

**`tools/fetch_pikafish.py`** —— 引擎获取：

1. `curl -L --retry 5 -C -` 下载 `Pikafish.2026-09-06.7z` 到 `vendor/`
   优先走镜像 `https://gh-proxy.com/<原始URL>`（实测 21.6 MB/s）；镜像失败则回退直连
2. 校验 sha256 = `41952bbfe2520faceb5902c69e6ab4845cc999841d2b49a95cc1be7867a25e5b`
   （**镜像不可信，但哈希可信** —— 校验通过即保证文件与官方发布完全一致）
3. `tar.exe -xf` 解压（libarchive 原生读 7z）
4. 安装**两个**文件到 `vendor/pikafish/`：
   - `Pikafish-Windows-x86-64-universal.exe` → `vendor/pikafish/pikafish.exe`
   - `pikafish.nnue`（50.7 MB，**必需**）→ `vendor/pikafish/pikafish.nnue`
5. 自检：启动引擎发 `uci`，收到 `uciok` 即成功；再 `go depth 10` 确认能出 `bestmove`（这一步同时验证 NNUE 加载成功）

**降级**：若引擎文件缺失或握手失败，`engine/fallback.py` 提供一个浅层 α-β（2~3 层 + 子力价值表）保证管线仍可端到端运行与测试；同时在面板上醒目标红"未使用皮卡鱼"。

### 4.7 `src/jjchess/control.py` —— 鼠标执行

```python
class Mouse:
    def __init__(self, dry_run: bool = False, jitter_px: int = 6): ...
    def click_client(self, x: float, y: float, origin: tuple[int,int]) -> None: ...
    def play_move(self, mv: Move, geom: BoardGeometry) -> None: ...
```

- `SetCursorPos` + `SendInput`（`MOUSEEVENTF_LEFTDOWN/LEFTUP`，绝对坐标 `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`，归一化到 `0..65535`）
- **拟人化（点击层面）**：落点加 ±`jitter_px` 随机偏移；按下→抬起间隔 `60~140ms`；起止两点之间 `180~350ms`
- `dry_run=True` 只打印坐标不点击（`--dry-run`），用于安全联调

**拟人化（思考层面）`--human-delay`** —— 避免"每步都秒回"暴露脚本特征：

- CLI：`--human-delay [MIN-MAX]`，裸写 `--human-delay` 等价 `--human-delay 0.5-2.0`（单位秒，浮点）
- 配置：`"human_delay": { "enabled": false, "min_s": 0.5, "max_s": 2.0 }`；`enabled=false` 时 CLI 传入的区间也视为关闭
- 实现在 `session.py` 的 `THINKING → PLAYING` 之间：`analyse()` 返回后再停 `random.uniform(min_s, max_s)`
- ⚠️ **该停顿必须可中断**（S6 要求）：不能一次 `time.sleep(2.0)`，而要**切成 ≤50ms 的片段轮询暂停/退出标志**，保证 `F8`/`ESC` 在延迟期间仍即时生效
- 停顿期间面板状态显示 `THINKING (human delay 1.3s)`，让使用者看得见它在"装思考"

### 4.8 `src/jjchess/session.py` —— 会话状态机（**鲁棒性核心**）

```
IDLE ──发现开局局面──▶ MY_TURN ──思考──▶ THINKING ──▶ PLAYING ──▶ VERIFY
  ▲                                                                  │
  │                                                            一致 ─┘─ 否 ─▶ 重试(1) ─▶ PAUSED(告警)
  │                                                                  │
  └──────────── OPPONENT_TURN ◀──等待对方走子──────────────────────── 是
                                    │
                              检测到终局 ──▶ GAME_OVER
```

**我方执色判定**：`--side red|black|auto`（默认 `auto`）。`auto` = 开局局面下看屏幕**最下一行（row=9）**的中宫位是红帅还是黑将 → 决定执色；同时决定是否上下翻转映射。

**对手走子检测 + 合法着法解释器**（消除识别噪声的关键设计）：

```
prev = 上一帧已确认的稳定局面
cur  = 本帧识别结果

若 cur == prev           → 局面未变，继续等待
否则：
  for m in generate(prev, side_to_move):
      if diff(make(prev, m), cur) <= TOLERANCE_CELLS:      # TOLERANCE_CELLS = 1
          → 采纳 m，并把 cur 修正为 make(prev, m)          # 吞掉单格误识别
          break
  else:
      → 无法用任何合法着法解释 ⇒ 判为识别异常
        · 重抓 3 次取多数投票
        · 仍失败 ⇒ 暂停 + 面板标红 + 蜂鸣告警
```

这一步同时解决了三个问题：① 单格误识别被自动纠正；② 顺便识别出"对手走了哪一步"（用于日志和记谱）；③ 拒绝把非法局面喂给皮卡鱼（避免引擎 `CRITICAL ERROR` 自杀）。

**我方走子校验**：点击后等待画面稳定（连续 2 帧相同），比对是否 ≈ `make(prev, my_move)`；不符 → 重试 1 次 → 仍不符则 `PAUSED`。

**终局检测**：局面连续 12 秒不变 且 非我方回合，或检测到底部弹出按钮区域亮度突增 → 判定对局结束，停止并打印对局记录。

### 4.9 `src/jjchess/record.py` —— 对局记录与中文记谱导出

```python
@dataclass
class MoveRecord:
    ply: int                # 从 1 开始的半步序号
    side: str               # "red" | "black"
    iccs: str               # "h2e2"
    chinese: str            # "炮二平五"
    score_cp: int | None    # 我方着法才有；对手着法为 None
    mate_in: int | None
    depth: int | None
    think_ms: int | None

class GameRecorder:
    def __init__(self, my_side: str, start_fen: str, started_at: datetime): ...
    def add(self, board_before: Board, mv: Move,
            analysis: Analysis | None = None, think_ms: int | None = None) -> MoveRecord: ...
    def to_text(self) -> str            # 中文记谱 txt 全文
    def iccs_line(self) -> str          # "h2e2 h9g7 h0g2 ..." 单行，便于程序复用
    def save(self, path: Path) -> None  # 写 UTF-8 文件
```

- 中文记谱复用已存在的 `rules/notation.py`（面板 PV 显示也在用它），不重复实现
- 记谱规则：**红方用汉字数字一~九（从红方右手边数起），黑方用阿拉伯数字 1~9（从黑方右手边数起）**；同一纵线两个同名子用"前/后"区分
- 导出路径：`<debug-dir>/game-YYYYmmdd-HHMMSS.txt`；未指定 `--debug-dir` 时写到 `records/`
- 终局时（`GAME_OVER`）自动保存；`ESC` 手动退出时也保存一次，避免丢记录
- 导出格式示例：

```
JJ象棋 对局记录
开始时间: 2026-10-11 15:30:22
我方执色: 红
初始局面: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
结束原因: 用户退出

 1. 红 炮二平五   (h2e2)  +27   d22  1203ms
 1. 黑 马8进7     (h9g7)
 2. 红 马二进三   (h0g2)  +19   d21  1198ms

ICCS: h2e2 h9g7 h0g2
```

### 4.10 `src/jjchess/ui/panel.py` —— 实时可视化面板

复用 OpenCV（零额外依赖），单窗口合成图，主循环内 `cv2.imshow` + `cv2.waitKey(1)`：

- **左半**：棋盘 ROI 缩放图，叠加每个交叉点的识别结果（棋子字符）与置信度颜色框（绿/黄/红）
- **右半**：文字信息 —— 我方执色、当前轮到谁、状态机状态、当前 FEN、引擎 `depth / score / nps / bestmove`、PV 前 8 步（中文记谱）、最近一次点击的像素坐标、抓图 FPS
- 底部状态条：`RUNNING / PAUSED / DRY-RUN / 告警信息`

**全局热键**（`GetAsyncKeyState` 轮询，不抢占焦点、不需要管理员权限）：

| 键 | 作用 |
|---|---|
| `F8` | 暂停 / 继续 |
| `F9` | 单步（只走一步后自动暂停） |
| `F10` | 切换 dry-run（只打印不点击） |
| `ESC` | 立即退出 |

### 4.11 `src/jjchess/main.py` —— 组装与入口

`argparse` CLI：

```
uv run jjchess [--side auto|red|black] [--movetime 1200] [--threads 8] [--hash 512]
               [--dry-run] [--no-panel] [--calibrate] [--debug-dir DIR]
               [--human-delay [MIN-MAX]] [--no-record]
```

主循环：`grab → locate(首帧) → recognize → session.step → panel.render → hotkeys` 循环，目标 4~6 FPS。

---

## 5. 关键算法与数据契约

### 5.1 像素 ↔ 棋盘坐标

```
px_x = origin_x + col * dx
px_y = origin_y + row * dy
```

反解用最近邻并校验残差 `< 0.4 * min(dx,dy)`，否则判为无效点击目标。

### 5.2 棋盘格编码

```
row: 0 = 屏幕最上（黑方底线，标准朝向）
     9 = 屏幕最下（红方底线，标准朝向）
col: 0 = 屏幕最左, 8 = 屏幕最右
```

### 5.3 ICCS 坐标映射（与皮卡鱼通信）

皮卡鱼用 `a0..i9`：`a` = 红方视角最左纵线，`rank 0` = 红方底线。

**我方执红（棋盘未翻转，红在下方）**

```
file = chr(ord('a') + col)
rank = 9 - row
```

**我方执黑（小程序会把棋盘 180° 翻转，黑在下方）**

```
file = chr(ord('i') - col)
rank = row
```

> 验证：标准开局黑方视角屏幕 `(row=0, col=0)` 应对应 `i0`（红方右车）。公式给 `file='i'-0='i'`, `rank=0` ⇒ `i0` ✓

> **✅ 已实测验证（`_probe/uci_probe.ps1`）**：构造"一步杀"局面
> `4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1`（红帅 d0、红车 a8、红车 h5、黑将 e9）
> 按本约定预测唯一杀着为 `h5h9`（车 h5→h9 借 a8 车控第 8 横线成杀）
> → 引擎返回 `info depth 12 ... score mate 1 ... pv h5h9` / `bestmove h5h9` **完全吻合** ✓
> 同时实测开局 `go depth 18` 返回 `bestmove c3c4`（兵七进一），PV `c3c4 h7e7 h0g2 h9g7 i0h0 i9h9 …` 全部为合法着法格式。

### 5.4 局面 → 引擎的完整链路

```
识别的 10×9 数组
  → Board.from_array()
  → Board.to_fen()                       # "…RNBAKABNR w - - 0 1"
  → rules.sanity_check(board, side)      # 双王存在、不在被将的假死状态、子力数合理
  → UciEngine.analyse(fen, movetime)
  → bestmove iccs
  → rules.parse_iccs() + is_legal()      # 引擎着法必须合法，否则丢弃并要求重算
  → geometry.cell_to_px() ×2
  → Mouse.play_move()
```

### 5.5 配置文件

`config/default.json`（可提交，含默认值）

```json
{
  "engine":  { "path": "vendor/pikafish/pikafish.exe", "nnue": "vendor/pikafish/pikafish.nnue",
               "movetime_ms": 1200, "threads": 8, "hash_mb": 512 },
  "vision":  { "conf_min": 0.60, "conf_good": 0.75 },
  "session": { "tolerance_cells": 1, "opponent_idle_sec": 0.0,
               "human_delay": { "enabled": false, "min_s": 0.5, "max_s": 2.0 } },
  "record":  true
}
```

没有出现在这里、只作为模块常量的：窗口标题 `WINDOW_TITLE = "JJ象棋"`（`src/jjchess/main.py`）、
棋盘 ROI 比例 `ROI_TOP/ROI_BOTTOM = 0.15/0.80`、`ROI_LEFT/ROI_RIGHT = 0.01/0.99`、
识别尺度 `ROI_SCALE = 1.30` / `SEARCH_SCALE = 1.90` / `TEMPLATE_PX = 72` / `MATCH_CORE_CELLS = 0.85` /
`EMPTY_LAPLACIAN = 3000`（`src/jjchess/vision/templates.py`）、
走子校验额度 `VERIFY_TIMEOUT_S = 5.0`、棋盘丢失额度 `BOARD_LOST_TIMEOUT_S = 10.0`（`src/jjchess/session.py`）、
画面稳定闸门 `SETTLE_SEC = 0.50`、定期复查棋盘 `RELOCATE_EVERY_S = 2.0`（`src/jjchess/main.py`）。
这些没做成配置项，是因为它们要么是「跟游戏画面绑定、换游戏才改」的量，
要么是调参时反复量过、写死并配了注释与测试比暴露给用户更安全。

`config/calibration.json` 与 `config/templates.npz` 为**运行时生成**，不入库，由 `tools/calibrate.py` 产出。

---

## 6. 项目结构与依赖

```
JJ/
├── pyproject.toml            # uv 管理；[tool.uv.index] 指向阿里云
├── .python-version           # 3.12
├── README.md                 # 使用说明（中文）
├── config/
│   ├── default.json
│   ├── calibration.json      # 生成物
│   └── templates.npz         # 生成物
├── vendor/
│   └── pikafish/pikafish.exe # tools/fetch_pikafish.py 下载
├── docs/superpowers/specs/2026-10-11-jjchess-autoplay-design.md
├── src/jjchess/
│   ├── __init__.py
│   ├── __main__.py
│   ├── window.py
│   ├── geometry.py
│   ├── control.py
│   ├── session.py
│   ├── record.py
│   ├── main.py
│   ├── vision/{__init__,locator,templates,recognizer}.py
│   ├── rules/{__init__,board,moves,notation}.py
│   ├── engine/{__init__,uci,fallback}.py
│   └── ui/{__init__,panel,hotkeys}.py
├── tools/{fetch_pikafish.py,calibrate.py,record_frames.py}
├── records/                  # 对局记谱导出（生成物）
└── tests/
    ├── test_geometry.py      ├── test_fen.py
    ├── test_moves.py         ├── test_notation.py
    ├── test_vision.py        ├── test_uci.py
    ├── test_record.py        ├── test_session.py
    └── fixtures/*.png
```

**运行依赖**（`uv add`）

| 包 | 用途 |
|---|---|
| `numpy` | 帧缓冲、棋盘数组、模板运算 |
| `opencv-python` | 图像处理、模板匹配、可视化面板窗口 |

**开发依赖**（`uv add --dev`）：`pytest`、`ruff`

其余全部 stdlib：`ctypes`（窗口/鼠标）、`subprocess`（引擎）、`argparse`、`json`、`dataclasses`、`pathlib`。

**阿里云镜像配置**（写进 `pyproject.toml`）

```toml
[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
```

---

## 7. 测试策略

| 测试 | 内容 |
|---|---|
| `test_moves.py` | **perft(1..4) = 44 / 1920 / 79666 / 3290240**（中国象棋公认基准值）。专项：蹩马腿、塞象眼、象不过河、炮需炮架、兵过河横走、将帅照面、九宫限制 |
| `test_fen.py` | 开局 FEN 往返；若干名局残局 FEN 往返；非法 FEN 拒绝 |
| `test_notation.py` | ICCS ↔ 中文记谱 双向往返；同线双子"前/后"消歧 |
| `test_geometry.py` | `cell_to_px` / `px_to_cell` 互逆；越界拒绝 |
| `test_vision.py` | fixture 帧（开局 + 中局若干）识别 100%；含"选中高亮"帧的鲁棒性 |
| `test_uci.py` | `skipif(not engine_present)`；已知杀局断言 bestmove；断言非法 FEN 时 wrapper 不崩溃（重启引擎） |
| `test_session.py` | 合法着法解释器：构造"单格误识别"输入，断言被正确纠正；构造"无法解释"输入，断言进入告警。**可中断睡眠**：暂停标志置位后提前返回。**终局判定**：轮到我方时长期静止 → `PAUSED` 而非 `GAME_OVER`。**全白帧不改变已确认局面** |
| `test_config.py` | `Config` 覆盖优先级；`parse_human_delay` 区间解析（裸开关 / `"0.5-2.0"` / 非法串拒绝） |
| `test_record.py` | `GameRecorder` 逐步累积后 `to_text()` 断言着法/记谱/ICCS 行完全一致；中文记谱与 UTF-8 落盘 |
| `test_locator.py` | 大厅帧（无棋盘）必须抛 `BoardNotFound`；合成网格图能定位；等距拟合能剔除异常峰 |
| `test_window.py` | 找不到窗口抛中文 `WindowNotFound`；`is_blank_frame` 对纯白/纯黑为真、对真实画面为假 |
| `test_control.py` | `dry_run` 不真点；抖动落在范围内；ICCS→屏幕坐标换算正确（易错点） |
| `test_panel.py` | 合成图尺寸固定；有帧/无帧都不崩；热键边沿去重 |
| 端到端 | `--dry-run` + 录制帧序列，跑完整管线不断言点击 |

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| Chromium 窗口 `PrintWindow` 偶发返回空白帧 | `is_blank_frame` + `is_torn_frame` 双检（撕裂帧的暗像素比达 0.2993），坏帧跳过重抓 `GRAB_ATTEMPTS = 3` 次；仍失败计入 `bad_streak`，连续 `MAX_BAD_STREAK = 10` 次退出 |
| 窗口被拖动 / 小程序尺寸变化 | 每帧校验 hwnd 与客户区尺寸；变化则重新定位棋盘。另加每 `RELOCATE_EVERY_S = 2.0` 秒定期复查（`locate()` 实测 10.9~12.7 ms，代价可忽略），定位失败就当作棋盘已消失 |
| 棋子选中高亮、走子动画导致误识别 | 只取盘心区 + CLAHE 归一化；动画期等画面稳定（`FrameSettle`，同一 FEN 连续 `SETTLE_SEC = 0.50` 秒）才识别 |
| 单格识别错误导致非法局面 | 合法着法解释器自动纠正；送引擎前 `sanity_check`。`interpret()` 额外要求**落点精确匹配** —— 一颗子走完一定在落点上，只放宽其余格子，否则容忍度会变成模型漂移的通道（真机踩过） |
| 皮卡鱼严格校验导致进程自杀 | 先校验后发送；wrapper 检测进程退出并自动重启（最多 3 次）。**已实测确认**：非法 FEN → exit code 1 |
| 引擎缺少 NNUE 权重无法加载 | `pikafish.nnue`（50.7 MB）与 exe 必须同目录；`fetch_pikafish.py` 安装**两个**文件；自检用 `go depth 10` 确认能出 `bestmove` |
| UCI 读取时序错误导致只搜到 depth 1 | **发完 `go` 不可立即 `quit`**；必须阻塞读到 `bestmove` 行 |
| 皮卡鱼下载失败（GitHub 403 / 直连仅 25 KB/s 需 35 分钟） | 优先镜像 `gh-proxy.com`（实测 21.6 MB/s，3 秒完成）；`curl --retry 5 -C -` 断点续传；**sha256 强校验**保证镜像内容与官方一致；失败时降级到浅层 α-β 并在面板标红 |
| 点击落空 / 走子动画没放完 | 点击后按**时间**给额度复检（`VERIFY_TIMEOUT_S = 5.0` 秒），不是「重试 1 次」—— 主循环一轮才 20 ms，两次校验只隔 40 ms，连棋子滑过去的动画都没放完。吃子动画实测要 2~3 秒才停稳；仍对不上就熔断为 PAUSED |
| 对局结束后进程空转、记谱不落盘 | 终局判断放在**主循环开头**（循环体里有 `continue` 的早退分支，放末尾会被绕过去）；`GAME_OVER` 即 `break` 并走 `_finish()` 落盘记谱 |
| 鼠标被抢占 / 误操作 | `F8` 即时暂停；`--dry-run` 全程不点击 |
| 窗口顶部 279px 在屏外 | 抓图不受影响（`PW_RENDERFULLCONTENT` 已实测可拿到屏外部分）；启动时校验棋盘是否完全在屏幕内，否则提示移动窗口 |

---

## 9. 评审结论（2026-10-11 已确认）

1. 模块划分与目录结构 —— ✅ 认可
2. 识别方案（模板匹配 + 合法着法解释器纠错）—— ✅ 认可（相比"训练 CNN 分类"更轻量、可解释、无需标注数据）
3. 思考时间默认 **1200ms**（可 `--movetime` 调整）—— ✅ 认可。实测本机 **14.5 Mnps**（16 线程），1200ms 可搜到 depth 20+，远超人类；对局总时长 30 分钟，按 60 步估算共耗约 72 秒，余量充足
4. **拟人化思考延迟** —— ✅ **纳入**，`--human-delay`，见 §4.7
5. **记谱导出** —— ✅ **纳入**，默认开启，`--no-record` 可关，见 §4.9
