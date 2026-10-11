# JJ象棋自动下棋脚本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做一个 Python 程序，自动抓取微信小游戏「JJ象棋」窗口画面 → 识别棋盘 32 子 → 用皮卡鱼算出最佳着法 → 自动点击鼠标走子 → 循环至终局，并提供实时可视化面板与中文记谱导出。

**Architecture:** 五层单向依赖。① **纯函数规则内核** `rules/`（无 I/O，板坐标 `[rank, file]`，rank 0 = 红方底线）→ ② **引擎层** `engine/`（皮卡鱼 UCI + 浅层 α-β 兜底，只吃 FEN 吐 ICCS 着法）→ ③ **视觉层** `vision/`（PrintWindow 抓图 → 投影峰值定位 → 模板匹配识别）→ ④ **会话状态机** `session/`（用「合法着法解释器」把视觉噪声吸收成合法着法，是鲁棒性核心）→ ⑤ **执行层** `control.py`（拟人化鼠标）+ `ui/`（OpenCV 面板）+ `record.py`（记谱）。

**Tech Stack:** Python 3.12（uv 管理，阿里云 PyPI 镜像）、numpy、opencv-python、ctypes（Win32 抓图/鼠标/热键）、Pikafish 2026-09-06（UCI）、pytest、ruff。除 numpy/opencv 外全部 stdlib。

**Spec:** `docs/superpowers/specs/2026-10-11-jjchess-autoplay-design.md`

## Global Constraints

以下为项目级硬约束，**每个任务都隐含包含本节**。数值全部逐字来自 spec。

- **运行环境**：Windows 11；Python **3.12**（本机已有，`uv python list --only-installed` 可见 `cpython-3.12.14` 与 `Python312\python.exe` 3.12.10）——**不需要从 GitHub 下载解释器**。
- **包管理**：`uv`（0.12.18）。运行依赖仅 `numpy`、`opencv-python`；dev 依赖仅 `pytest`、`ruff`。其余一律 stdlib。
- **镜像源**：`pyproject.toml` 写入 `[[tool.uv.index]] name="aliyun" url="https://mirrors.aliyun.com/pypi/simple/" default=true`。所有 `uv add` 走该源。
- **平台**：只用 Windows API（`ctypes.windll.user32/gdi32`）；无跨平台要求。
- **语言**：所有面向用户的输出、日志、异常消息、代码注释、README 均为**中文**。
- **Shell**：本机只有 PowerShell 5.1（无 pwsh 7），且默认 ExecutionPolicy 禁止跑 `.ps1`。任何脚本用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <path>` 调用。
- **棋盘坐标**：`row ∈ [0,9]` 屏幕从上到下，`col ∈ [0,8]` 屏幕从左到右。
- **板坐标**：`rank ∈ [0,9]`，`rank 0` = **红方底线**；`file ∈ [0,8]`，`file 0` = **a 纵线**（红方视角最左）。
- **屏幕↔板映射**：执红 `rank = 9 - row, file = col`；执黑（小程序 180° 翻转）`rank = row, file = 8 - col`。
- **ICCS**：`chr(ord('a') + file) + str(rank)`，如 `a0`、`e9`。着法为 4 字符串 `"h2e2"`。
- **开局 FEN**：`rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1`
- **引擎路径**：`vendor/pikafish/pikafish.exe` 与 `vendor/pikafish/pikafish.nnue`（**两个文件都必须存在**）。
- **引擎默认参数**：`--movetime 1200 --threads 8 --hash 512`。
- **引擎交互铁律**：发完 `go` **绝不可立即发 `quit`**；必须阻塞读 stdout 直到出现 `bestmove` 行。非法 FEN 会让引擎 **exit code 1 自杀**，wrapper 必须检测并自动重启（≤3 次）。
- **视觉常量**：`conf_min = 0.60`、`conf_good = 0.75`、判空用**拉普拉斯方差** `EMPTY_LAPLACIAN = 3000`（不是 empty 模板相关，见 Task 12 的 Ruling）、ROI 边长 `1.30 * cell_px`、搜索尺度 `1.90`、模板 `72×72` 灰度+CLAHE、匹配窗口 `MATCH_CORE_CELLS = 0.85`（避开选中光圈）、匹配法 `cv2.TM_CCOEFF_NORMED`。
- **纠错常量**：`TOLERANCE_CELLS = 1`。
- **拟人化**：`human_delay` 默认 `{enabled: false, min_s: 0.5, max_s: 2.0}`；延迟必须切成 **≤50ms 片段**轮询暂停标志（保证 S6 可中断）。鼠标落点抖动 ±6px，按下→抬起 `60~140ms`，两点间 `180~350ms`。
- **git 提交信息前缀**：`feat:` / `test:` / `docs:` / `chore:`。

## Review Focus

以下 5 类输入/失效模式 spec 未明说、也无任务测试覆盖，但最可能让使用者踩坑。**每一项都已在下方所属任务中加了对应测试**（见各任务标注 `[RF-n]`）。

- **RF-1 游戏没开 / 窗口标题变了**：`window.py` 找不到 `JJ象棋` 窗口时，必须抛带中文说明的 `WindowNotFound` 并友好退出，不能抛 `AttributeError` 或退化成死循环。→ Task 8
- **RF-2 抓图返回全白帧**：窗口最小化或被遮挡时 `PrintWindow` 会返回纯白图。必须能检测出空白帧并重试/降级/告警；**绝不能把全白帧当成"棋盘 32 子全空"** 而去走非法着法。→ Task 8 + Task 15
- **RF-3 画面里根本没有棋盘**：停在「对战」大厅菜单、或弹出广告时，locator 必须判定失败并等待，不能把菜单纹理当网格线定位出一堆垃圾坐标。→ Task 10
- **RF-4 局面长时间不变**：对方长考 / 网络卡顿时局面会静止很久。**轮到我方时局面不变必须判 PAUSED 告警，不能判 GAME_OVER**；~~只有「非我方回合 + 超时」或「底部按钮区亮度突增」才可判终局~~ —— **后半句已被真机推翻**：对局进行中对方单步能想两分钟以上，「非我方回合 + 超时」也只能**暂停**；终局只认「棋盘从画面消失」。→ Task 15 / Task 19 Ruling
- **RF-5 窗口被拖动 / 改尺寸 / 换显示器**：已缓存的 `calibration.json` 与 `BoardGeometry` 立即失效。必须每帧校验 hwnd 与客户区尺寸，变化则重新定位，不能继续对着旧坐标点击。→ Task 8 + Task 17

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | uv 项目定义；阿里云 index；依赖与 ruff/pytest 配置 |
| `.python-version` | `3.12` |
| `src/jjchess/rules/board.py` | `Board` 数据结构（`(10,9) int8`，`[rank, file]`）+ FEN 解析/序列化 + 棋子编码常量 |
| `src/jjchess/rules/moves.py` | 着法生成、`make/unmake`、将军/将帅照面判定、`game_status`、`perft` |
| `src/jjchess/rules/notation.py` | ICCS ↔ 中文纵线记谱双向转换（含同线双子「前/后」消歧） |
| `src/jjchess/geometry.py` | `BoardGeometry`（origin/dx/dy/cell_px）+ 像素↔格互转 + 客户区→屏幕偏移 + JSON 持久化 |
| `src/jjchess/config.py` | `Config` dataclass、`config/default.json` 加载、CLI 覆盖合并、`parse_human_delay` |
| `src/jjchess/window.py` | `find_window` 按标题找窗；`GameWindow.grab()` 用 `PrintWindow(hwnd,dc,2)` 抓客户区；空白帧检测 |
| `src/jjchess/vision/locator.py` | 投影峰值法从帧自动求 `BoardGeometry`；失败抛 `BoardNotFound` |
| `src/jjchess/vision/templates.py` | 15 类模板集 `TemplateSet`（`.npz` 存取）+ 从带标注帧构建模板 |
| `src/jjchess/vision/recognizer.py` | 逐格 ROI → 72×72 灰度 CLAHE → 两阶段模板匹配 → `Recognition` |
| `src/jjchess/engine/uci.py` | 皮卡鱼 UCI 封装（阻塞读至 `bestmove`、自杀重启、`Analysis`） |
| `src/jjchess/engine/fallback.py` | 浅层 α-β 兜底引擎（接口与 `UciEngine` 一致） |
| `src/jjchess/control.py` | `SetCursorPos`+`SendInput` 拟人化点击；`dry_run` 只打印 |
| `src/jjchess/record.py` | `GameRecorder` 逐步累积着法 → 中文记谱 txt + ICCS 行 |
| `src/jjchess/session.py` | 状态机 + 合法着法解释器 + 可中断睡眠 |
| `src/jjchess/ui/panel.py` | OpenCV 单窗口合成图（左棋盘叠加 / 右信息栏 / 底状态条） |
| `src/jjchess/ui/hotkeys.py` | `GetAsyncKeyState` 轮询 F8/F9/F10/ESC |
| `src/jjchess/main.py` | argparse CLI + 依赖组装 + 主循环 |
| `src/jjchess/__main__.py` | `python -m jjchess` 入口 |
| `tools/fetch_pikafish.py` | 下载（gh-proxy 镜像）+ sha256 校验 + `tar.exe` 解压 + 安装 exe 与 nnue |
| `tools/record_frames.py` | 录制真机帧到 `tests/fixtures/`（供识别测试与调参） |
| `tools/calibrate.py` | 手工两点标定 + 从开局帧构建 `config/templates.npz` |
| `tests/` | 见各任务；`tests/fixtures/*.png` 为真机截图，**需提交进 git** |

---

### Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`、`.python-version`、`src/jjchess/__init__.py`、`src/jjchess/__main__.py`、`README.md`
- Modify: `.gitignore`（追加 `records/`、`.ruff_cache/`、`.pytest_cache/`）
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 `uv run pytest` 与 `uv run jjchess`（此刻 `--help` 只要不崩即可）

- [ ] **Step 1: 用本机已有的 3.12 建项目**

```powershell
cd C:\Users\admin\Documents\deepseek-harness\JJ
uv init --lib --name jjchess --python 3.12 --no-workspace
```

若 `uv init` 生成的布局不是 `src/`，手工建 `src/jjchess/` 并在 `pyproject.toml` 写 `[tool.hatch.build.targets.wheel] packages = ["src/jjchess"]`。

- [ ] **Step 2: 写 `pyproject.toml`（阿里云源 + 脚本入口 + ruff/pytest 配置）**

```toml
[project.scripts]
jjchess = "jjchess.main:main"

[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 3: 加依赖（走阿里云源）**

```powershell
uv add numpy opencv-python
uv add --dev pytest ruff
```

**验证镜像确实生效**（这是用户明确要求，必须证明）：

```powershell
uv add --verbose numpy 2>&1 | Select-String "aliyun"
```

期望：输出中出现 `mirrors.aliyun.com`。若没有，检查 `[[tool.uv.index]]` 是否被写在 `[project]` 段之内（必须在顶层）。

- [ ] **Step 4: 写失败测试 `tests/test_smoke.py`（`[RF-1]` 之外的基础可用性）**

```python
import numpy as np
import cv2

def test_deps_importable():
    assert np.zeros((2, 2)).shape == (2, 2)
    assert cv2.__version__

def test_package_importable():
    import jjchess
    assert jjchess.__version__ == "0.1.0"
```

- [ ] **Step 5: 跑测试确认失败**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess'` 或 `AttributeError: __version__`

- [ ] **Step 6: 建包与 `__main__.py`**

`src/jjchess/__init__.py` 内容：`__version__ = "0.1.0"`。
`src/jjchess/__main__.py` 内容：`from jjchess.main import main` / `raise SystemExit(main())`。
`src/jjchess/main.py` 此刻只放最小 `def main(argv=None) -> int: return 0`（Task 17 再补全）。

- [ ] **Step 7: 跑测试确认通过**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: 2 passed

- [ ] **Step 8: 补 `.gitignore` 并提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "chore: uv 项目脚手架（阿里云镜像 + numpy/opencv + pytest/ruff）"
```

---

### Task 2: `rules/board.py` —— 棋盘与 FEN

**Files:**
- Create: `src/jjchess/rules/__init__.py`、`src/jjchess/rules/board.py`
- Test: `tests/test_fen.py`

**Interfaces:**
- Consumes: 无（纯 stdlib + numpy）
- Produces:
  - 常量 `RED = 1`、`BLACK = -1`、`K,A,E,H,R,C,P = 1..7`
  - `PIECE_CHARS: dict[int,str]`：`{1:"K",2:"A",3:"E",4:"H",5:"R",6:"C",7:"P", -1:"k",…, 0:"."}`
  - `CHAR_VALUES: dict[str,int]`（上表反向）
  - `START_FEN: str`
  - `@dataclass Board(grid: np.ndarray, side: int = RED)`，`grid` 形状 `(10,9)` dtype `int8`，索引 `[rank, file]`
  - `Board.from_fen(fen: str) -> Board`、`Board.to_fen() -> str`、`Board.copy() -> Board`
  - `Board.at(rank: int, file: int) -> int`、`Board.set(rank, file, value) -> None`
  - `Board.pieces(side: int) -> list[tuple[int,int]]`（返回 `(rank, file)`）
  - `square(rank: int, file: int) -> str`、`parse_square(sq: str) -> tuple[int,int]`

- [ ] **Step 1: 写失败测试 `tests/test_fen.py`**

断言全部写死为字面值：

```python
def test_start_fen_roundtrip():
    assert Board.from_fen(START_FEN).to_fen() == START_FEN

def test_start_position_has_32_pieces():
    b = Board.from_fen(START_FEN)
    assert len(b.pieces(RED)) == 16 and len(b.pieces(BLACK)) == 16

def test_红方底线在_rank_0():
    b = Board.from_fen(START_FEN)
    assert b.at(0, 0) == R          # a0 = 红车
    assert b.at(0, 4) == K          # e0 = 红帅
    assert b.at(9, 4) == -K         # e9 = 黑将
    assert b.at(3, 0) == P          # a3 = 红兵
    assert b.at(6, 0) == -P         # a6 = 黑卒

def test_square_iccs():
    assert square(0, 0) == "a0"
    assert square(9, 4) == "e9"
    assert parse_square("h2") == (2, 7)

def test_非法_fen_被拒绝():
    for bad in ["", "9/9/9/9/9/9/9/9/9/9 w - - 0 1", "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x", "rnbakabnr/8/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"]:
        with pytest.raises(ValueError):
            Board.from_fen(bad)
```

（第一条非法 FEN 缺双方将帅；第二条 `x` 不是合法走子方；第三条某行只有 8 格。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_fen.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.rules'`

- [ ] **Step 3: 实现 `src/jjchess/rules/board.py`**

关键实现要点（签名与常量已定，实现自行完成）：
- `from_fen` 分割为 6 段（不足 6 段补 `- - 0 1`）；第 1 段必须切出 **恰好 10 行**、每行展开后 **恰好 9 列**，否则 `raise ValueError(f"非法 FEN：棋盘行数/列数不对：{fen!r}")`。
- **FEN 第 i 行（i=0 在最前）对应 `rank = 9 - i`**；行内从左到右对应 `file = 0..8`。
- `to_fen` 反向；每行末尾的连续空格压成数字（`9` 表示全空行），空棋盘不得输出 `000000000`。
- 走子方段：`w` → `RED`，`b` → `BLACK`，其它 → `ValueError`。
- `from_fen` 必须校验**双方各有且仅有一个将/帅**，否则 `ValueError`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_fen.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(rules): Board 数据结构与 XQ-FEN 解析/序列化"
```

---

### Task 3: `rules/moves.py` —— 着法生成与 perft

**Files:**
- Create: `src/jjchess/rules/moves.py`
- Test: `tests/test_moves.py`

**Interfaces:**
- Consumes: Task 2 的 `Board`、`RED/BLACK/K/A/E/H/R/C/P`、`square`、`parse_square`
- Produces:
  - `parse_move(mv: str) -> tuple[tuple[int,int], tuple[int,int]]`（`"h2e2"` → `((2,7),(2,4))`）
  - `format_move(src: tuple[int,int], dst: tuple[int,int]) -> str`
  - `generate(board: Board) -> list[str]`（按 `board.side` 生成**伪合法**着法，ICCS 字符串）
  - `is_attacked(board: Board, rank: int, file: int, by_side: int) -> bool`
  - `in_check(board: Board, side: int) -> bool`
  - `legal_moves(board: Board) -> list[str]`（过滤掉走完自己被将军的着法）
  - `make(board: Board, mv: str) -> int`（就地改 `grid` 并翻转 `side`；返回被吃子，`0` 表示无）
  - `unmake(board: Board, mv: str, captured: int) -> None`
  - `game_status(board: Board) -> str` → `"ONGOING" | "CHECK" | "CHECKMATE" | "STALEMATE"`（无着可走且被将军 = `CHECKMATE`；无着可走且未被将军 = `STALEMATE`）
  - `perft(board: Board, depth: int) -> int`

- [ ] **Step 1: 写失败测试 `tests/test_moves.py`**

```python
@pytest.mark.parametrize("depth,expected", [(1,44),(2,1920),(3,79666),(4,3290240)])
def test_perft_startpos(depth, expected):
    assert perft(Board.from_fen(START_FEN), depth) == expected

def test_蹩马腿():
    # b0 红马，b1 被己方子占住 -> 马只能走 a2/c2/d1/d2 中的 d1/d2 方向被堵
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/1P7/1N2K4 w - - 0 1")
    mvs = {m for m in generate(b) if m.startswith("b0")}
    assert mvs == {"b0a2", "b0c2", "b0d1"}   # b0b1 方向被 b1 的兵堵住 -> 无 b0? 的直向跳

def test_炮必须隔一个炮架才能吃():
    b = Board.from_fen("4k4/9/9/9/9/9/4r4/9/9/3K1C3 w - - 0 1")
    # 红炮 c0 与黑车 e0 之间无架 -> 不能吃
    assert "c0e0" not in generate(b)

def test_象不过河():
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/2E1K4 w - - 0 1")
    assert all(not m.startswith("c0") or int(m[3]) <= 4 for m in generate(b))

def test_兵过河后能横走():
    b = Board.from_fen("4k4/9/9/9/9/9/9/1P7/9/4K4 w - - 0 1")   # b2 兵未过河
    assert "b2a2" not in generate(b) and "b2c2" not in generate(b)
    b2 = Board.from_fen("4k4/9/9/9/9/9/1P7/9/9/4K4 w - - 0 1")  # b4 兵已过河
    assert "b4a4" in generate(b2) and "b4c4" in generate(b2)

def test_将帅照面非法():
    b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1")
    assert in_check(b, RED)   # 中路无遮挡 -> 红帅被黑将"照面"攻击

def test_一步杀判定():
    b = Board.from_fen("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1")
    assert "h5h9" in legal_moves(b)
    make(b, "h5h9")
    assert game_status(b) == "CHECKMATE"
```

这些 FEN 均已在 `_probe/uci_probe.ps1` 中实测能被皮卡鱼接受（`h5h9` 为唯一杀着），可直接复用。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_moves.py -v`
Expected: FAIL —— `ImportError: cannot import name 'generate'`

- [ ] **Step 3: 实现走法生成规则（逐条落实 spec §4.5）**

- 车：四方向直线延伸，遇子止；敌子可吃。
- 马：8 个落点；**先直后斜**——先走的那个正交邻格必须为空（蹩马腿）。
- 象/相：4 个田字点；**田心必须为空**（塞象眼）；红方 `rank <= 4`，黑方 `rank >= 5`（不过河）。
- 士/仕：九宫四斜角；九宫 = `file 3..5`，红 `rank 0..2` / 黑 `rank 7..9`。
- 将/帅：九宫四正方向；**另外**，`is_attacked` 对将/帅返回真当同纵线两将之间无任何子（将帅照面）——实现为 `generate` 里拒绝走出照面局面的着法即可，两者等价且更容易测。
- 炮：不吃子时同车；吃子时该方向**有且仅有一个**炮架。
- 兵/卒：向前一格；**过河后可横走**（红 `rank >= 5`、黑 `rank <= 4`）；永不后退。
- `_in_palace(rank, file, side)`、`_on_own_side(rank, side)` 抽成小工具函数复用。

- [ ] **Step 4: 跑 perft 确认结点数完全匹配**

Run: `uv run pytest tests/test_moves.py -v`
Expected: 全部 PASS，perft(4) = **3290240**（这是本任务最强的正确性保证；数字不符必须逐层 debug 到 depth 1 = 44 为止）

- [ ] **Step 5: 用皮卡鱼交叉验证 perft（可选但推荐）**

```powershell
# 皮卡鱼支持 perft
$in = "uci`nposition fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1`ngo perft 4`nquit`n"
$in | & .\vendor\pikafish\pikafish.exe | Select-String "Nodes searched"
```
Expected: 与测试同值。不一致则以皮卡鱼为准并修 `moves.py`。

- [ ] **Step 6: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(rules): 走法生成/将军判定/perft（perft(4)=3290240 通过）"
```

---

### Task 4: `rules/notation.py` —— 中文纵线记谱

**Files:**
- Create: `src/jjchess/rules/notation.py`
- Test: `tests/test_notation.py`

**Interfaces:**
- Consumes: Task 2 `Board`/`square`/`parse_square`，Task 3 `parse_move`/`format_move`/`generate`
- Produces:
  - `to_chinese(board_before: Board, mv: str) -> str` —— `to_chinese(start, "h2e2") == "炮二平五"`
  - `from_chinese(board_before: Board, text: str) -> str | None` —— 反解，歧义或非法返回 `None`
  - `PIECE_NAMES_RED: dict[int,str]` = `{K:"帅",A:"仕",E:"相",H:"马",R:"车",C:"炮",P:"兵"}`
  - `PIECE_NAMES_BLACK: dict[int,str]` = `{K:"将",A:"士",E:"象",H:"马",R:"车",C:"炮",P:"卒"}`
  - `CN_DIGITS = "一二三四五六七八九"`

- [ ] **Step 1: 写失败测试 `tests/test_notation.py`**

```python
def test_炮二平五():
    assert to_chinese(Board.from_fen(START_FEN), "h2e2") == "炮二平五"

def test_马8进7():
    b = Board.from_fen(START_FEN)
    make(b, "h2e2")                      # 红先走一步，轮到黑
    assert to_chinese(b, "h9g7") == "马8进7"

def test_兵七进一():
    assert to_chinese(Board.from_fen(START_FEN), "c3c4") == "兵七进一"

def test_同线双子用前后消歧():
    # a 线两个红车：a0 与 a5 -> 靠前者(a5)称"前车"，靠后者(a0)称"后车"
    b = Board.from_fen("4k4/9/9/9/R8/9/9/9/9/R3K4 w - - 0 1")
    assert to_chinese(b, "a5a6") == "前车进一"
    assert to_chinese(b, "a0a1") == "后车进一"

@pytest.mark.parametrize("mv", ["h2e2", "b2e2", "h2h6", "c3c4", "b0c2", "a0a1"])
def test_往返一致(mv):
    b = Board.from_fen(START_FEN)
    assert from_chinese(b, to_chinese(b, mv)) == mv

def test_胡说八道返回None():
    b = Board.from_fen(START_FEN)
    assert from_chinese(b, "炮九平五") is None
    assert from_chinese(b, "不是着法") is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_notation.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.rules.notation'`

- [ ] **Step 3: 实现记谱规则**

- **纵线编号**：红方从**红方右手边**数起，用汉字 `一二三四五六七八九` → `file 8` 为「一」，`file 0` 为「九」，即 `num = 9 - file`。黑方从**黑方右手边**数起，用阿拉伯数字 `1..9` → `num = file + 1`。
- **动作**：同纵线移动为「进/退」（红 `rank` 增大为「进」，黑 `rank` 减小为「进」）；换纵线为「平」。
- **数值**：车/炮/兵/将 走直线时数值 = 移动的格数；马/象/士 走斜线时数值 = **目标纵线号**。
- **消歧**：同一纵线上有 ≥2 个同类同色子时，用「前/后」替代纵线号。三个及以上兵在同一线时用「一二三」等序号（此场景罕见，实现为 `前/中/后` 或数字前缀均可，测试不覆盖，注释说明即可）。
- `from_chinese` 用「枚举 `generate(board)` 全部着法 → 各自 `to_chinese` 比对」实现，天然覆盖所有歧义情形；无匹配返回 `None`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_notation.py -v`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(rules): ICCS ↔ 中文纵线记谱双向转换"
```

---

### Task 5: `geometry.py` —— 棋盘几何

**Files:**
- Create: `src/jjchess/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass(frozen=True) class BoardGeometry`：字段 `origin: tuple[float,float]`（**客户区坐标**下 `(screen_row=0, screen_col=0)` 交叉点中心）、`dx: float`、`dy: float`、`cell_px: float`
  - `cell_to_px(self, row: int, col: int) -> tuple[float,float]`
  - `px_to_cell(self, x: float, y: float) -> tuple[int,int]`
  - `with_offset(self, ox: float, oy: float) -> BoardGeometry`（客户区坐标 → 屏幕坐标）
  - `is_plausible(self) -> bool`（`dx > 5`、`dy > 5`、`|dx-dy| / max(dx,dy) < 0.2`、`0.5 <= cell_px/min(dx,dy) <= 1.0`）
  - `to_json(self) -> dict` / `from_json(cls, d: dict) -> BoardGeometry`
  - `class ScreenMapper`：`__init__(self, my_side: int)`；`to_board(self, row: int, col: int) -> tuple[int,int]`（返回 `(rank, file)`）；`to_screen(self, rank: int, file: int) -> tuple[int,int]`

- [ ] **Step 1: 写失败测试 `tests/test_geometry.py`**

```python
def test_互逆():
    g = BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)
    for row in range(10):
        for col in range(9):
            assert g.px_to_cell(*g.cell_to_px(row, col)) == (row, col)

def test_越界拒绝():
    g = BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)
    with pytest.raises(ValueError):
        g.cell_to_px(10, 0)
    with pytest.raises(ValueError):
        g.cell_to_px(0, 9)

def test_with_offset():
    g = BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)
    assert g.with_offset(-1440.0, -279.0).cell_to_px(0, 0) == (-1340.0, -219.0)

def test_json_往返():
    g = BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)
    assert BoardGeometry.from_json(g.to_json()) == g

def test_执红映射():
    m = ScreenMapper(RED)
    assert m.to_board(9, 0) == (0, 0)      # 屏幕左下角 = a0 = 红车
    assert m.to_board(0, 0) == (9, 0)      # 屏幕左上角 = a9
    assert m.to_screen(0, 0) == (9, 0)

def test_执黑映射_棋盘180翻转():
    m = ScreenMapper(BLACK)
    assert m.to_board(9, 0) == (9, 8)      # 黑方视角左下角 = a9
    assert m.to_board(0, 0) == (0, 8)      # 左上角 = i0
    for row in range(10):
        for col in range(9):
            assert m.to_screen(*m.to_board(row, col)) == (row, col)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.geometry'`

- [ ] **Step 3: 实现 `geometry.py`**

`cell_to_px(row, col) = (origin[0] + col*dx, origin[1] + row*dy)`（`origin` 是 `(x, y)`，测试里的 `(100.0, 60.0)` 即 `x=100, y=60`）。
`px_to_cell` 用 `round((x-origin[0])/dx)` 并做 `0..8 / 0..9` 越界 `ValueError`。
**注意 `origin` 的轴序**：字段是 `(x, y)`，但名字里的 `row=0,col=0` 指语义。测试已把语义钉死，照测试写即可。
`ScreenMapper` 严格实现 Global Constraints 的映射公式，并对越界输入 `raise ValueError`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: BoardGeometry 像素↔格互转与执色映射"
```

---

### Task 6: `config.py` —— 配置加载与 `--human-delay` 解析

**Files:**
- Create: `src/jjchess/config.py`、`config/default.json`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass class Config`，字段：`engine_path: Path`、`nnue_path: Path`、`movetime_ms: int`、`threads: int`、`hash_mb: int`、`conf_min: float`、`conf_good: float`、`tolerance_cells: int`、`human_delay: tuple[float,float] | None`、`opponent_idle_sec: float`、`debug_dir: Path | None`、`record: bool`
  - `DEFAULTS: Config`（`movetime_ms=1200, threads=8, hash_mb=512, conf_min=0.60, conf_good=0.75, tolerance_cells=1, human_delay=None, opponent_idle_sec=0.0, record=True`）—— 置信度阈值从 0.75/0.90 下调见 Task 15 的 Ruling
  - `load_config(path: Path | None, overrides: dict) -> Config`
  - `parse_human_delay(arg: str | None) -> tuple[float,float] | None`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
def test_裸开关用默认区间():
    assert parse_human_delay("") == (0.5, 2.0)

def test_显式区间():
    assert parse_human_delay("1.5-3") == (1.5, 3.0)

def test_未启用返回None():
    assert parse_human_delay(None) is None

@pytest.mark.parametrize("bad", ["abc", "1.5", "3-1", "-1-2", "1-2-3", "0-0"])
def test_非法输入拒绝(bad):
    with pytest.raises(ValueError):
        parse_human_delay(bad)

def test_覆盖优先级_cli大于文件():
    cfg = load_config(Path("config/default.json"), {"movetime_ms": 700})
    assert cfg.movetime_ms == 700
    assert cfg.threads == 8

def test_缺失配置文件走默认值(tmp_path):
    cfg = load_config(tmp_path / "nope.json", {})
    assert cfg.movetime_ms == 1200 and cfg.record is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.config'`

- [ ] **Step 3: 实现 `config.py` 与 `config/default.json`**

`config/default.json` 内容按 spec §5.5：
```json
{
  "engine": { "path": "vendor/pikafish/pikafish.exe", "nnue": "vendor/pikafish/pikafish.nnue",
              "movetime_ms": 1200, "threads": 8, "hash_mb": 512 },
  "vision": { "conf_min": 0.60, "conf_good": 0.75 },
  "session": { "tolerance_cells": 1, "opponent_idle_sec": 0.0,
               "human_delay": { "enabled": false, "min_s": 0.5, "max_s": 2.0 } },
  "record": true
}
```
`parse_human_delay`：`None` → `None`；`""`（裸开关）→ `(0.5, 2.0)`；否则按 `-` 切成两段转 float，要求 `min > 0` 且 `min < max`，否则 `raise ValueError("--human-delay 格式应为 MIN-MAX，例如 0.5-2.0")`。
`load_config`：文件不存在或 `path is None` 用 `DEFAULTS`；`overrides` 里值为 `None` 的键跳过（表示 CLI 未指定）；`human_delay` 在文件里 `enabled=false` 时取 `None`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: Config 加载与 --human-delay 区间解析"
```

---

### Task 7: `engine/uci.py` —— 皮卡鱼 UCI 封装

**Files:**
- Create: `src/jjchess/engine/__init__.py`、`src/jjchess/engine/uci.py`
- Test: `tests/test_uci.py`

**Interfaces:**
- Consumes: Task 2 `START_FEN`
- Produces:
  - `@dataclass class Analysis`：`bestmove: str`、`score_cp: int | None`、`mate_in: int | None`、`depth: int`、`sel_depth: int`、`nps: int`、`pv: list[str]`
  - `class EngineError(RuntimeError)`
  - `class UciEngine`：`__init__(self, exe: Path, threads: int = 8, hash_mb: int = 512, multipv: int = 1, nnue: Path | None = None)`、`__enter__`/`__exit__`、`analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis`、`restart(self) -> None`、`quit(self) -> None`、`restarts: int`
  - `def engine_present(exe: Path | None = None) -> bool`

- [ ] **Step 1: 写失败测试 `tests/test_uci.py`**

```python
ENGINE = Path("vendor/pikafish/pikafish.exe")
pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="皮卡鱼未就绪")

def test_握手与求招():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse(START_FEN, movetime_ms=800)
    assert len(a.bestmove) == 4 and a.bestmove[0] in "abcdefghi"
    assert a.depth >= 8 and a.pv and a.pv[0] == a.bestmove

def test_一步杀():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1", movetime_ms=600)
    assert a.bestmove == "h5h9" and a.mate_in == 1

def test_发go后不能提前quit_必须等到bestmove():
    # 回归测试：曾经把 quit 紧跟 go，导致只搜到 depth 1 返回垃圾着法
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse(START_FEN, movetime_ms=1500)
    assert a.depth >= 12, f"搜索被提前终止，depth 只有 {a.depth}"

def test_非法fen不崩溃_自动重启():
    with UciEngine(ENGINE, threads=4, hash_mb=128) as e:
        a = e.analyse("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1", movetime_ms=300)   # 将帅照面
        assert e.restarts >= 1
        b = e.analyse(START_FEN, movetime_ms=400)     # 重启后仍能正常工作
    assert len(b.bestmove) == 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_uci.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.engine'`

- [ ] **Step 3: 实现 `UciEngine`**

- `subprocess.Popen([str(exe)], cwd=exe.parent, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)`。
- 握手：发 `uci` → 读行直到 `uciok`（超时 **20s**）；再依次发 `setoption name Threads value N`、`setoption name Hash value M`、`setoption name EvalFile value <abs nnue path>`（`nnue` 非空时）、`isready` → 读直到 `readyok`（超时 30s）。
- `analyse`：发 `position fen <fen>` + `go movetime <ms>`，然后**阻塞读行直到 `startswith("bestmove")`**（超时 `movetime_ms/1000 + 10` 秒）。
  - 每读到 `info ` 开头且含 ` depth ` 的行，用正则抽 `depth`、`seldepth`、`score cp <n>` / `score mate <n>`、`nps`、`pv ...`，**覆盖式保留最后一条**。
  - `bestmove` 行取 `split()[1]`（后面可能跟 `ponder xxx`，必须忽略）。
  - 读过程中若 `p.poll() is not None`（进程已退）或读到 `info string CRITICAL ERROR` → 调 `restart()` 并重试一次；仍失败则 `raise EngineError`。
- `restart`：`quit()` 后重开进程并重新握手，`self.restarts += 1`；`restarts > 3` 时 `raise EngineError("皮卡鱼连续崩溃 3 次，已放弃")`。
- `__exit__` 发 `quit` 并 `wait(timeout=5)`，超时则 `kill()`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_uci.py -v`
Expected: 4 passed（每个 <3 秒）

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(engine): 皮卡鱼 UCI 封装（阻塞读 bestmove + 崩溃自愈）"
```

---

### Task 8: `engine/fallback.py` + `tools/fetch_pikafish.py`

**Files:**
- Create: `src/jjchess/engine/fallback.py`、`tools/fetch_pikafish.py`
- Test: `tests/test_fallback.py`

**Interfaces:**
- Consumes: Task 3 `legal_moves`/`make`/`unmake`/`game_status`、Task 7 `Analysis`
- Produces:
  - `class FallbackEngine`：`__init__(self, depth: int = 3)`、`analyse(self, fen: str, movetime_ms: int = 1200) -> Analysis`（签名与 `UciEngine.analyse` 一致，便于互换）、`__enter__`/`__exit__`
  - `MATERIAL: dict[int,int]` = `{K:10000, R:900, C:450, H:400, A:200, E:200, P:100}`

- [ ] **Step 1: 写失败测试 `tests/test_fallback.py`**

```python
def test_找到一步杀():
    e = FallbackEngine(depth=2)
    a = e.analyse("4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1")
    assert a.bestmove == "h5h9" and a.mate_in == 1

def test_返回合法着法():
    e = FallbackEngine(depth=2)
    a = e.analyse(START_FEN)
    assert a.bestmove in legal_moves(Board.from_fen(START_FEN))

def test_分析对象字段完整():
    a = FallbackEngine(depth=2).analyse(START_FEN)
    assert isinstance(a.pv, list) and a.depth == 2 and a.nps >= 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_fallback.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.engine.fallback'`

- [ ] **Step 3: 实现浅层 α-β**

- 评估：子力价值表求和（红正黑负）**加上**极小位置分（兵越靠前越高）。
- 搜索：negamax + α-β，**按被吃子价值排序着法**（提高剪枝率，让 depth 3 在 1 秒内跑完）。
- 检测到 `game_status == "CHECKMATE"` 返回 `mate_in`（层数换算）与 `±30000` 分。
- `nps` 用实际耗时与结点数算；`pv` 记录主变前 8 步。
- 若 `depth=3` 在开局超过 `movetime_ms`，递减到实际完成的层数并如实填 `depth`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_fallback.py -v`
Expected: 3 passed

- [ ] **Step 5: 写 `tools/fetch_pikafish.py`（幂等下载安装）**

行为：
1. 目标 `vendor/pikafish/pikafish.exe` 与 `pikafish.nnue` 都已存在且 `engine_present()` 为真 → 打印「已就绪」直接返回 0。
2. 下载 `Pikafish.2026-09-06.7z`，URL **优先** `https://gh-proxy.com/https://github.com/official-pikafish/Pikafish/releases/download/Pikafish-2026-09-06/Pikafish.2026-09-06.7z`，失败回退直连。
   用 `subprocess` 调 `C:\Windows\System32\curl.exe -L --retry 5 --retry-delay 3 -C - --connect-timeout 30 -o <out> <url>`。
3. 校验 `size == 53603424` 且 `sha256 == 41952bbfe2520faceb5902c69e6ab4845cc999841d2b49a95cc1be7867a25e5b`，不符则 `raise SystemExit("下载文件校验失败，请重试")` 并删除坏文件。
4. 解压用 `tar.exe -xf <7z> -C vendor/extract`（bsdtar 3.6.2 原生支持 7z）。
5. 复制 `Pikafish-Windows-x86-64-universal.exe` → `vendor/pikafish/pikafish.exe`、`pikafish.nnue` → `vendor/pikafish/pikafish.nnue`。
6. 自检：用 `UciEngine` 跑 `analyse(START_FEN, movetime_ms=300)`，成功则打印 `id name` 与 bestmove 并返回 0，否则返回 1。

- [ ] **Step 6: 实跑验证（引擎已在本机，应直接短路返回）**

Run: `uv run python tools/fetch_pikafish.py`
Expected: 打印引擎已就绪与自检结果，exit 0

- [ ] **Step 7: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(engine): 浅层 α-β 兜底引擎 + 皮卡鱼下载校验脚本"
```

---

### Task 9: `window.py` —— 窗口发现与抓图

**Files:**
- Create: `src/jjchess/window.py`
- Test: `tests/test_window.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `class WindowNotFound(RuntimeError)`
  - `def find_window(title: str = "JJ象棋", timeout_s: float = 0.0) -> GameWindow`（`timeout_s > 0` 时轮询等待；超时抛 `WindowNotFound`）
  - `def is_blank_frame(img: np.ndarray) -> bool`
  - `@dataclass class GameWindow`：`hwnd: int`、`title: str`、`client_size: tuple[int,int]`、`client_origin: tuple[int,int]`、`dpi: int`
    - `refresh(self) -> None`（重新查询几何，若 hwnd 已失效抛 `WindowNotFound`）
    - `grab(self) -> np.ndarray`（BGR `(H,W,3) uint8`，**客户区全图，含屏幕外部分**）
    - `is_alive(self) -> bool`
    - `size_changed(self, w: int, h: int) -> bool`

- [ ] **Step 1: 写失败测试 `tests/test_window.py`**

```python
def test_找不到窗口抛中文异常():
    with pytest.raises(WindowNotFound) as ei:
        find_window("绝对不存在的窗口标题xyz", timeout_s=0.3)
    assert "找到" in str(ei.value) or "未" in str(ei.value)      # [RF-1]

def test_空白帧检测_纯白为真():
    assert is_blank_frame(np.full((2560, 1440, 3), 255, np.uint8)) is True

def test_空白帧检测_纯黑为真():
    assert is_blank_frame(np.zeros((2560, 1440, 3), np.uint8)) is True

def test_空白帧检测_真实画面为假():
    img = np.random.randint(0, 255, (400, 300, 3), np.uint8)     # [RF-2]
    assert is_blank_frame(img) is False

@pytest.mark.skipif(not _game_running(), reason="JJ象棋未运行")
def test_抓真窗口():
    w = find_window("JJ象棋", timeout_s=2.0)
    img = w.grab()
    assert img.shape == (w.client_size[1], w.client_size[0], 3)
    assert not is_blank_frame(img)
```

其中 `_game_running()` 用 `find_window` 试探（捕获 `WindowNotFound` 返回 False）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_window.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.window'`

- [ ] **Step 3: 实现 `window.py`**

- 全部用 `ctypes.windll.user32` / `gdi32`，显式声明 `argtypes`/`restype`（Win64 句柄是 64 位，**不声明会截断**）。
- `find_window`：`EnumWindows` 回调 + `GetWindowTextW` 精确匹配 `title`（回退 `子串匹配`），跳过不可见窗口。
- 几何：`GetClientRect` 取 `client_size`；`ClientToScreen(hwnd, POINT(0,0))` 取 `client_origin`（**这是屏幕坐标中客户区左上角**，实测主屏竖屏时为 `(-1440, -279)`）；DPI 用 `GetDpiForWindow`（失败回退 96）。
- `grab()` 必须用：
  ```
  hdc_win = GetWindowDC(hwnd)
  hdc_mem = CreateCompatibleDC(hdc_win)
  hbmp    = CreateCompatibleBitmap(hdc_win, w, h)
  SelectObject(hdc_mem, hbmp)
  PrintWindow(hwnd, hdc_mem, 2)     # 2 = PW_RENDERFULLCONTENT  ← 关键，不可用 0
  GetDIBits(...)  →  BGR ndarray
  释放：DeleteObject(hbmp) / DeleteDC(hdc_mem) / ReleaseDC(hwnd, hdc_win)
  ```
  行序注意 `biHeight` 取负值得到 top-down 的 `H×W` 数组。
- `is_blank_frame`：转灰度后判 `std < 1.0`（纯色）或「≥99% 像素灰度为 255」，二者之一为真即判空白。
- `refresh()` 里若 `client_size` 与上次不同，更新字段（调用方负责据此重定位棋盘 —— 见 Task 17 的 `[RF-5]`）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_window.py -v`
Expected: 4~5 passed

- [ ] **Step 5: 手工确认抓到的是完整棋盘画面**

```powershell
uv run python -c "import cv2; from jjchess.window import find_window; w=find_window('JJ象棋',2.0); img=w.grab(); cv2.imwrite('_probe/live_grab.png', img); print(w.client_size, img.shape)"
```
然后用 read_image 查看 `_probe/live_grab.png`，确认**顶部 279px（屏外部分）内容存在且不是白色**。

- [ ] **Step 6: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: 窗口发现与 PW_RENDERFULLCONTENT 抓图（含空白帧检测）"
```

---

### Task 10: `tools/record_frames.py` + 采集真机 fixture

**Files:**
- Create: `tools/record_frames.py`
- Create（生成物，**需提交**）：`tests/fixtures/*.png`
- Test: 无单测（工具脚本）；由 Task 12 的识别测试消费其产物

**Interfaces:**
- Consumes: Task 9 `find_window`/`GameWindow`
- Produces: `tests/fixtures/` 下的 PNG，命名约定：
  - `opening.png` —— **开局标准局面**（32 子齐全）
  - `lobby.png` —— 大厅菜单帧（**无棋盘**，供 Task 11 的 `[RF-3]` 测试用）
  - `mid_<NN>.png` —— 中局若干帧
  - 同目录 `opening.fen` —— 与 `opening.png` 对应的实际 FEN（人工确认后写入）

- [ ] **Step 1: 写 `tools/record_frames.py`**

CLI：`uv run python tools/record_frames.py --name opening --count 1 [--interval 0.5]`，行为：`find_window("JJ象棋")` → 循环 `grab()` → `cv2.imwrite(f"tests/fixtures/{name}.png", img)`（`count>1` 时依次 `{name}_{i:02d}.png`）→ 打印尺寸与是否空白帧。加 `--watch` 选项：每 0.5s 抓一帧，**画面发生变化就存一份带时间戳的帧**，用于自动收集中局。

- [ ] **Step 2: 采集 `lobby.png`（现在就能做）**

```powershell
uv run python tools/record_frames.py --name lobby --count 1
```
用 read_image 确认里面是「对战」大厅菜单、**没有棋盘**。

- [ ] **Step 3: 采集 `opening.png`（需要用户配合）**

⚠️ **这一步需要人在微信里操作**：请用户把 JJ象棋切到**一局对弈的开局局面**（推荐进「人机对战」——可以自己跟自己下、也不影响别人计分），保持棋盘完整可见、**不要最小化窗口**。然后：

```powershell
uv run python tools/record_frames.py --name opening --count 1
```

用 read_image 确认：32 个棋子齐全、红方在下方、与 `START_FEN` 一致。确认后在 `tests/fixtures/opening.fen` 写入 `START_FEN` 的字面值。

- [ ] **Step 4: 采集 3~5 张中局帧**

对局中每隔一会儿运行一次（或直接用 `--watch`）：

```powershell
uv run python tools/record_frames.py --name mid --count 4 --interval 20 --watch
```

`--watch` 存下的帧人工挑 3~5 张，重命名为 `mid_01.png` …，并**为每张手工把对应 FEN 写进 `tests/fixtures/mid_NN.fen`**（用棋盘截图对照 `START_FEN` 逐格记录；这是 Task 12 识别测试的期望值来源）。

- [ ] **Step 5: 提交 fixture**

```powershell
git add tests/fixtures tools/record_frames.py
git -c user.name=dev -c user.email=dev@local commit -m "test: 采集真机 fixture 帧（开局/大厅/中局）与录制工具"
```

---

### Task 11: `vision/locator.py` —— 棋盘自动定位

**Files:**
- Create: `src/jjchess/vision/__init__.py`、`src/jjchess/vision/locator.py`
- Test: `tests/test_locator.py`

**Interfaces:**
- Consumes: Task 5 `BoardGeometry`；`tests/fixtures/opening.png`
- Produces:
  - `class BoardNotFound(RuntimeError)`
  - `def locate(img: np.ndarray) -> BoardGeometry`（失败抛 `BoardNotFound`，消息为中文并提示运行 `tools/calibrate.py`）
  - `def find_lines(profile: np.ndarray, count: int) -> list[float]`（供单测直接验证等距拟合）

- [ ] **Step 1: 写失败测试 `tests/test_locator.py`**

```python
def test_大厅帧判定无棋盘():
    img = cv2.imread("tests/fixtures/lobby.png")           # [RF-3]
    with pytest.raises(BoardNotFound):
        locate(img)

def test_合成棋盘能定位():
    img = _render_fake_board()          # 见 Step 1 说明
    g = locate(img)
    assert g.is_plausible()

def test_等距拟合能剔除异常峰():
    prof = np.zeros(400, np.float32)
    for i, x in enumerate([10, 50, 90, 130, 170, 210, 250, 290, 330]):
        prof[x] = 100
    prof[95] = 80                        # 一个假峰
    assert np.allclose(find_lines(prof, 9), [10, 50, 90, 130, 170, 210, 250, 290, 330], atol=3)

def test_真实开局帧能定位():
    img = cv2.imread("tests/fixtures/opening.png")
    g = locate(img)
    assert g.is_plausible()
    assert 60 < g.cell_px < 200
```

`_render_fake_board()` 是本测试文件内的辅助函数：用 numpy 画一张浅木色底 + 深色 9×10 等距网格线的合成图（约 900×1000），用于在没有真机 fixture 时也能验证算法。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_locator.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.vision'`

- [ ] **Step 3: 实现投影峰值法（spec §4.3）**

1. `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`；取 ROI `y ∈ [0.15H, 0.80H]`、`x ∈ [0.01W, 0.99W]`。
2. `mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)` —— 深色网格线变白。
3. 纵向投影 `prof_x = mask.sum(axis=0) / 255`；横向 `prof_y = mask.sum(axis=1) / 255`。
4. `find_lines(profile, count)`：`cv2.GaussianBlur` 平滑 → 峰检测（局部极大且高于 `0.3 * max`）→ 若峰数 > count，取最高的 count 个；再对峰位置做**等距最小二乘拟合**（`spacing = median(diff)`，用 `origin + i*spacing` 重排，剔除与拟合值偏差 > `0.35*spacing` 的峰并重拟合，最多 3 轮）。
5. 等距性校验：相邻间距变异系数 > 5% 时同样走上面的重拟合；最终仍不满足则 `raise BoardNotFound`。
6. `origin = (vlines[0], hlines[0])`、`dx = mean(diff(vlines))`、`dy = mean(diff(hlines))`、`cell_px = 0.85 * min(dx, dy)`。
7. 用 `BoardGeometry.is_plausible()` 做最后把关，不通过 → `BoardNotFound`。
8. `[RF-3]`：`BoardNotFound` 的判定要**足够早**——若 9 条纵线峰或 10 条横线峰的数量不足（< count），直接抛，不要硬凑。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_locator.py -v`
Expected: 4 passed

- [ ] **Step 5: 目视确认定位结果**

```powershell
uv run python -c "import cv2;from jjchess.vision.locator import locate;from jjchess.window import find_window;w=find_window('JJ象棋',2.0);img=w.grab();g=locate(img);print(g);import numpy as np;vis=img.copy();[cv2.circle(vis,(int(g.cell_to_px(r,c)[0]),int(g.cell_to_px(r,c)[1])),6,(0,0,255),-1) for r in range(10) for c in range(9)];cv2.imwrite('_probe/located.png',vis)"
```
read_image 查看 `_probe/located.png`，确认 90 个红点**精确落在每个交叉点上**。

- [ ] **Step 6: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(vision): 投影峰值法自动定位棋盘（失败抛 BoardNotFound）"
```

---

### Task 12: `vision/templates.py` + `tools/calibrate.py` —— 模板构建

**Files:**
- Create: `src/jjchess/vision/templates.py`、`tools/calibrate.py`
- Create（生成物，**不提交**，已在 `.gitignore`）：`config/templates.npz`
- Test: `tests/test_templates.py`

**Interfaces:**
- Consumes: Task 2 `Board`/`CHAR_VALUES`/`PIECE_CHARS`、Task 5 `BoardGeometry`、Task 6 `Config`
- Produces:
  - `CLASS_NAMES: list[str]` = `[".", "K","A","E","H","R","C","P", "k","a","e","h","r","c","p"]`（15 类；`.` = 空）
  - `ROI_SCALE = 1.30`、`TEMPLATE_PX = 72`
  - `def make_patch(img: np.ndarray, geom: BoardGeometry, row: int, col: int) -> np.ndarray`（返回 `(72,72) uint8` 灰度，已完成 CLAHE）
  - `@dataclass class TemplateSet`：`names: list[str]`、`data: np.ndarray`（`(15,72,72) uint8`）
    - `save(self, path: Path) -> None` / `classmethod load(cls, path: Path) -> TemplateSet`
    - `match(self, patch: np.ndarray) -> tuple[str, float]`（返回 `(类名, 置信度)`；**内部先比 `.`，`>0.9` 直接返回**）
  - `def build_templates(img: np.ndarray, geom: BoardGeometry, board: Board, mapper: "ScreenMapper") -> TemplateSet`
  - `def load_or_build(cfg: Config, img, geom, board, mapper) -> TemplateSet`

- [ ] **Step 1: 写失败测试 `tests/test_templates.py`**

```python
def test_类名有15个且含empty():
    assert len(CLASS_NAMES) == 15 and CLASS_NAMES[0] == "."

def test_从开局帧构建模板并全部匹配正确():
    img = cv2.imread("tests/fixtures/opening.png")
    geom = locate(img)
    board = Board.from_fen(Path("tests/fixtures/opening.fen").read_text().strip())
    ts = build_templates(img, geom, board, ScreenMapper(RED))
    assert ts.data.shape == (15, 72, 72)
    wrong = []
    for row in range(10):
        for col in range(9):
            rank, file = ScreenMapper(RED).to_board(row, col)
            expected = PIECE_CHARS[board.at(rank, file)]
            got, conf = ts.match(make_patch(img, geom, row, col))
            if got != expected or conf < 0.9:
                wrong.append((row, col, expected, got, round(conf, 3)))
    assert wrong == []

def test_模板集npz往返(tmp_path):
    ts = build_templates(...)          # 复用上面的 fixture
    ts.save(tmp_path / "t.npz")
    again = TemplateSet.load(tmp_path / "t.npz")
    assert again.names == ts.names and np.array_equal(again.data, ts.data)

def test_空点被剪枝为empty():
    img = cv2.imread("tests/fixtures/opening.png")
    geom = locate(img)
    # f0 在开局是空点
    name, conf = build_templates(...).match(make_patch(img, geom, 9, 5))
    assert name == "." and conf > 0.9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_templates.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.vision.templates'`

- [ ] **Step 3: 实现 `templates.py`**

- `make_patch`：以 `geom.cell_to_px(row, col)` 为中心，边长 `round(ROI_SCALE * geom.cell_px)` 的正方形；越界部分用图像边缘像素填充（`cv2.copyMakeBorder` 的 `BORDER_REPLICATE`）；`cv2.resize` 到 `72×72`；`cvtColor → GRAY`；`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(...)`。
- `build_templates`：对 90 个交叉点算 `patch`，按 `PIECE_CHARS[board.at(rank,file)]` 分桶；**每桶取逐像素中位数**（`np.median(axis=0).astype(np.uint8)`）作为该类模板 —— 抗单格噪声。为空桶（如开局没有的类）填入该类的**全棋盘同色中位数**或全 `0` 并在 `names` 中保留位置（识别时若置信度低会落到 `unknown`）。
- `match`：`cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)` 在尺寸相同时返回 `1×1`，取 `.ravel()[0]`。**先与 `names.index(".")` 的模板比，`>0.9` 立即返回 `(".", score)`**。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 4 passed

- [ ] **Step 5: 写 `tools/calibrate.py`（手工兜底 + 生成 `config/templates.npz`）**

两个功能：
1. `--build`（默认）：抓当前帧 → `locate()`；失败则进入手工模式；成功后用 `opening.fen`（或 `--fen`）构建模板并写入 `config/templates.npz`。
2. `--manual`：`cv2.imshow` 显示抓到的帧，提示用户**依次点击左上角交叉点与右下角交叉点**（鼠标回调 `cv2.setMouseCallback`），反推 `origin/dx/dy/cell_px` 并写入 `config/calibration.json`（`BoardGeometry.to_json()`）。

- [ ] **Step 6: 实跑并目视验证**

```powershell
uv run python tools/calibrate.py --build
uv run python -c "from jjchess.vision.templates import *;ts=TemplateSet.load(__import__('pathlib').Path('config/templates.npz'));print(ts.names, ts.data.shape, ts.data.mean())"
```
Expected: 15 类、`(15,72,72)`、均值在 60~200 之间（不是全黑也不是全白）

- [ ] **Step 7: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(vision): 15 类模板集构建/存取/匹配 + 手工标定工具"
```

---

### Task 13: `vision/recognizer.py` —— 棋子识别

**Files:**
- Create: `src/jjchess/vision/recognizer.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: Task 2 `Board`、Task 5 `BoardGeometry`/`ScreenMapper`、Task 12 `TemplateSet`/`make_patch`
- Produces:
  - `@dataclass class Recognition`：`board: Board`（**已是板坐标**）、`confidence: np.ndarray`（`(10,9) float32`，**屏幕格坐标**）、`unknown: list[tuple[int,int]]`（**屏幕 `(row,col)`**，面板叠加要用）
  - `def recognize(img: np.ndarray, geom: BoardGeometry, templates: TemplateSet, mapper: "ScreenMapper", side_to_move: int = RED) -> Recognition`

- [ ] **Step 1: 写失败测试 `tests/test_vision.py`**

```python
@pytest.fixture(scope="module")
def opening():
    img = cv2.imread("tests/fixtures/opening.png")
    geom = locate(img)
    fen = Path("tests/fixtures/opening.fen").read_text().strip()
    return img, geom, TemplateSet.load(Path("config/templates.npz")), Board.from_fen(fen)

def test_开局32子全对(opening):
    img, geom, ts, expect = opening
    rec = recognize(img, geom, ts, ScreenMapper(RED))
    assert rec.board.grid.tolist() == expect.grid.tolist()
    assert rec.unknown == []
    assert rec.confidence.min() > 0.75   # 实测定在 0.872（MATCH_CORE_CELLS = 0.85）

@pytest.mark.parametrize("name", ["mid_01", "mid_02", "mid_03"])
def test_中局识别正确(name):
    img = cv2.imread(f"tests/fixtures/{name}.png")
    geom = locate(img)
    ts = TemplateSet.load(Path("config/templates.npz"))
    expect = Board.from_fen(Path(f"tests/fixtures/{name}.fen").read_text().strip())
    rec = recognize(img, geom, ts, ScreenMapper(RED))
    assert rec.board.grid.tolist() == expect.grid.tolist()

def test_选中高亮帧仍能识别(opening):        # 见 spec §7「含选中高亮帧的鲁棒性」
    img, geom, ts, expect = opening
    hi = img.copy()
    x, y = geom.cell_to_px(0, 1)             # 给一个棋子加一圈高亮
    cv2.circle(hi, (int(x), int(y)), int(geom.cell_px * 0.6), (0, 215, 255), 8)
    rec = recognize(hi, geom, ts, ScreenMapper(RED))
    assert rec.board.grid.tolist() == expect.grid.tolist()

def test_执黑视角识别(opening):
    img, geom, ts, expect = opening
    flipped = cv2.rotate(img, cv2.ROTATE_180)
    g2 = locate(flipped)
    rec = recognize(flipped, g2, ts, ScreenMapper(BLACK))
    assert rec.board.grid.tolist() == expect.grid.tolist()

def test_低置信度进unknown(opening):
    img, geom, ts, _ = opening
    noise = img.copy()
    x, y = geom.cell_to_px(5, 4)
    cv2.rectangle(noise, (int(x-40), int(y-40)), (int(x+40), int(y+40)), (255, 0, 255), -1)
    rec = recognize(noise, geom, ts, ScreenMapper(RED))
    assert (5, 4) in rec.unknown
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_vision.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.vision.recognizer'`

- [ ] **Step 3: 实现 `recognize`**

对屏幕 `(row, col)` 遍历 90 格：`make_patch` → `templates.match` → `CHAR_VALUES[名字]`（`.` 映射为 `0`）；`(rank, file) = mapper.to_board(row, col)`，写进 `board.grid[rank, file]`。置信度写 `confidence[row, col]`；`conf < conf_min(0.60)` 的格子**置为 `0`（空）**并加入 `unknown`。返回的 `board.side` 设为 `side_to_move`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_vision.py -v`
Expected: 全 PASS（若开局帧有 1~2 格错，说明模板或 ROI 需调，**必须先修到 100% 再往下走**）

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(vision): 逐格模板匹配识别（含执黑翻转与低置信兜底）"
```

---

### Task 14: `control.py` —— 拟人化鼠标

**Files:**
- Create: `src/jjchess/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: Task 5 `BoardGeometry`、Task 3 `parse_move`
- Produces:
  - `class Mouse`：`__init__(self, dry_run: bool = False, jitter_px: int = 6)`、`clicks: list[tuple[float,float]]`（记录每次实际点击的屏幕坐标，`dry_run` 时也记）、`click_screen(self, x: float, y: float) -> None`、`play_move(self, mv: str, geom: BoardGeometry) -> tuple[tuple[float,float], tuple[float,float]]`（`geom` 必须是**已加过屏幕偏移**的）、`move_to(self, x, y)`
  - `def _send_click(self, x: int, y: int) -> None`（真发 `SendInput`）

- [ ] **Step 1: 写失败测试 `tests/test_control.py`**

```python
def _geom():
    return BoardGeometry(origin=(100.0, 60.0), dx=80.0, dy=80.0, cell_px=68.0)

def test_dry_run不真点():
    m = Mouse(dry_run=True)
    m.play_move("e0e1", _geom())
    assert len(m.clicks) == 2

def test_落点抖动在范围内():
    m = Mouse(dry_run=True, jitter_px=6)
    for _ in range(50):
        (x1, y1), (x2, y2) = m.play_move("e0e1", _geom())
        assert abs(x1 - 100.0) <= 6 and abs(y1 - 60.0) <= 6      # e0 -> (row9,col4) = (420,780)
        assert abs(x2 - 100.0) <= 6

def test_点击坐标对应正确交叉点():
    m = Mouse(dry_run=True, jitter_px=0)
    (x1, y1), (x2, y2) = m.play_move("h2e2", _geom())
    assert (x1, y1) == (100.0 + 7*80.0, 60.0 + 2*80.0)     # h2 -> (row=7, col=7)
    assert (x2, y2) == (100.0 + 4*80.0, 60.0 + 2*80.0)     # e2 -> (row=7, col=4)

def test_非法着法拒绝():
    with pytest.raises(ValueError):
        Mouse(dry_run=True).play_move("zzzz", _geom())
```

⚠️ 注意测试里的坐标换算：`h2` = `rank 2, file 7` → 屏幕 `row = 9-2 = 7, col = 7`。**`play_move` 收到的是 ICCS，必须先 `parse_move` 得到板坐标，再用 `ScreenMapper.to_screen` 转屏幕格**。上面 `test_点击坐标对应正确交叉点` 的期望值请按此自行核对并写死（这是本任务最容易写错的地方）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_control.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.control'`

- [ ] **Step 3: 实现 `control.py`**

- `play_move` 流程：`parse_move(mv)` → 起点/终点各 `cell_to_px` → 各加 `random.uniform(-jitter, +jitter)` → `move_to(src)` → `click` → `time.sleep(random.uniform(0.18, 0.35))` → `move_to(dst)` → `click`。
- `_send_click`：`SetCursorPos(x, y)` 后构造 `INPUT` 结构发 `SendInput`（`MOUSEEVENTF_LEFTDOWN` / `LEFTUP`），按下与抬起之间 `time.sleep(random.uniform(0.06, 0.14))`。
  - 绝对坐标需按虚拟桌面归一化到 `0..65535` 并带 `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`。**虚拟桌面左上角与尺寸用 `GetSystemMetrics(SM_XVIRTUALSCREEN/SM_YVIRTUALSCREEN/SM_CXVIRTUALSCREEN/SM_CYVIRTUALSCREEN)` 取**（本机虚拟桌面起点是 `(-1440, -279)`，用 0 当原点会点错）。
  - ⚠️ 使用 `SetCursorPos` 时坐标是**屏幕物理像素**，不要再做归一化 —— 二选一即可，实现里固定用 `SetCursorPos` + `SendInput` 发按键事件（不设 `MOUSEEVENTF_ABSOLUTE`，让 `SendInput` 在当前位置点击）。
- `dry_run=True`：只 `print(f"[dry-run] 点击 ({x:.0f}, {y:.0f})")` 并记录，不调任何 Win32 写操作。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_control.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: 拟人化鼠标执行（dry-run 安全模式）"
```

---

### Task 15: `record.py` —— 对局记录与中文记谱导出

**Files:**
- Create: `src/jjchess/record.py`
- Test: `tests/test_record.py`

**Interfaces:**
- Consumes: Task 2 `Board`/`START_FEN`、Task 3 `make`、Task 4 `to_chinese`、Task 7 `Analysis`
- Produces:
  - `@dataclass class MoveRecord`：`ply: int`、`side: int`、`iccs: str`、`chinese: str`、`score_cp: int | None`、`mate_in: int | None`、`depth: int | None`、`think_ms: int | None`
  - `class GameRecorder`：`__init__(self, my_side: int, start_fen: str, started_at: datetime)`、`moves: list[MoveRecord]`、`result: str = "未知"`、`add(self, board_before: Board, mv: str, analysis: Analysis | None = None, think_ms: int | None = None) -> MoveRecord`、`to_text(self) -> str`、`iccs_line(self) -> str`、`save(self, path: Path) -> None`

- [ ] **Step 1: 写失败测试 `tests/test_record.py`**

```python
def _rec():
    return GameRecorder(RED, START_FEN, datetime(2026, 10, 11, 15, 30, 22))

def test_记谱内容正确():
    r = _rec()
    b = Board.from_fen(START_FEN)
    a = Analysis(bestmove="h2e2", score_cp=27, mate_in=None, depth=22, sel_depth=30, nps=14_000_000, pv=["h2e2","h9g7"])
    r.add(b, "h2e2", a, think_ms=1203)
    make(b, "h2e2")
    r.add(b, "h9g7")
    assert r.iccs_line() == "h2e2 h9g7"
    assert [m.chinese for m in r.moves] == ["炮二平五", "马8进7"]
    assert r.moves[0].score_cp == 27 and r.moves[0].depth == 22 and r.moves[0].think_ms == 1203
    assert r.moves[1].score_cp is None

def test_文本格式():
    r = _rec()
    b = Board.from_fen(START_FEN)
    r.add(b, "h2e2")
    t = r.to_text()
    assert "JJ象棋 对局记录" in t
    assert "开始时间: 2026-10-11 15:30:22" in t
    assert "我方执色: 红" in t
    assert f"初始局面: {START_FEN}" in t
    assert " 1. 红 炮二平五   (h2e2)" in t
    assert t.rstrip().endswith("ICCS: h2e2")

def test_保存为utf8(tmp_path):
    r = _rec(); r.add(Board.from_fen(START_FEN), "h2e2")
    p = tmp_path / "g.txt"; r.save(p)
    assert "炮二平五" in p.read_text(encoding="utf-8")

def test_黑方着法标注为黑():
    r = _rec()
    b = Board.from_fen(START_FEN); make(b, "h2e2")
    rec = r.add(b, "h9g7")
    assert rec.side == BLACK and " 1. 黑 马8进7" in r.to_text()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_record.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.record'`

- [ ] **Step 3: 实现 `record.py`**

`add` 用 `to_chinese(board_before, mv)` 取记谱；`ply = len(self.moves) + 1`；`side = board_before.side`。
`to_text` 逐行拼接（格式见 spec §4.9 的示例），**红黑分列时用 `(ply + 1) // 2` 作回合号**；只有我方着法带 `score_cp/mate_in/depth/think_ms` 后缀，对手着法省略。
`save` 用 `path.write_text(self.to_text(), encoding="utf-8")`，`path.parent.mkdir(parents=True, exist_ok=True)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_record.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: GameRecorder 对局记录与中文记谱导出"
```

---

### Task 16: `session.py` —— 状态机与合法着法解释器（**鲁棒性核心**）

**Files:**
- Create: `src/jjchess/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: Task 2 `Board`、Task 3 `legal_moves`/`make`/`game_status`、Task 5 `ScreenMapper`、Task 6 `Config`、Task 7 `Analysis`、Task 14 `Mouse`、Task 15 `GameRecorder`
- Produces:
  - `class State(str, Enum)`：`IDLE / MY_TURN / THINKING / PLAYING / VERIFY / OPPONENT_TURN / PAUSED / GAME_OVER`
  - `def interruptible_sleep(seconds: float, stop: Callable[[], bool], slice_s: float = 0.05) -> bool`（返回 `True` 表示睡满，`False` 表示被中断）
  - `def interpret(prev: Board, cur: Board, tolerance: int = 1) -> str | None` —— **纯函数，可独立测试**
  - `@dataclass class SessionDeps`：`board_provider: Callable[[], Board | None]`、`engine`、`mouse`、`mapper: ScreenMapper`、`recorder: GameRecorder | None`、`config: Config`
  - `class Session`：`__init__(self, deps: SessionDeps)`、`state: State`、`board: Board | None`、`last_analysis: Analysis | None`、`message: str`、`pause()`、`resume()`、`request_stop()`、`should_stop: bool`、`step(self) -> None`

- [ ] **Step 1: 写失败测试 `tests/test_session.py`**

```python
# ---- 合法着法解释器 ----
def test_完全一致返回None():
    b = Board.from_fen(START_FEN)
    assert interpret(b, b.copy()) is None

def test_识别出对手着法():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy(); make(cur, "h2e2")
    assert interpret(prev, cur) == "h2e2"

def test_单格误识别被纠正():                      # [RF] 核心纠错能力
    prev = Board.from_fen(START_FEN)
    cur = prev.copy(); make(cur, "h2e2")          # 真实着法
    cur.grid[0, 0] = 0                            # 但 a0 的红车被误识别成空
    assert interpret(prev, cur) == "h2e2"

def test_无法解释返回None():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy()
    cur.grid[0, 0] = 0; cur.grid[1, 0] = 0; cur.grid[2, 0] = 0; cur.grid[0, 3] = 0
    assert interpret(prev, cur) is None

def test_不修改入参():
    prev = Board.from_fen(START_FEN)
    cur = prev.copy(); make(cur, "h2e2")
    want = prev.grid.tolist()
    interpret(prev, cur)
    assert prev.grid.tolist() == want

# ---- 可中断睡眠（S6 / S7）----
def test_睡眠可被提前中断():
    flag = {"v": False}
    t0 = time.monotonic()
    threading.Timer(0.2, lambda: flag.__setitem__("v", True)).start()
    ok = interruptible_sleep(3.0, lambda: flag["v"], slice_s=0.05)
    assert ok is False and time.monotonic() - t0 < 1.0

def test_睡眠睡满返回True():
    t0 = time.monotonic()
    assert interruptible_sleep(0.3, lambda: False, slice_s=0.05) is True
    assert 0.25 < time.monotonic() - t0 < 1.0

# ---- 终局判定（[RF-4]）----
def test_轮到我方时局面不变_不进终局():
    s = _make_session()                      # 站在 MY_TURN 的会话
    s._idle_since = time.monotonic() - 1000  # 假装局面已静止很久
    s.step()
    assert s.state is not State.GAME_OVER

def test_非我方回合静止超时_进终局():
    s = _make_session(state=State.OPPONENT_TURN)
    s._idle_since = time.monotonic() - 1000
    s.step()
    assert s.state is State.GAME_OVER

# ---- [RF-2] 全白帧不得改变局面 ----
def test_全白帧不改变已确认局面():
    s = _make_session()                      # board_provider 返回 None 模拟抓图失败
    before = s.board
    for _ in range(10):
        s.step()
    assert s.board is before
```

`_make_session()` 是本测试文件内的辅助构造器：注入假的 `board_provider` / `engine`（返回固定 `Analysis`）/ `Mouse(dry_run=True)` / `Config`，把会话推进到指定状态。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.session'`

- [ ] **Step 3: 实现 `interpret`（纯函数，先写这个）**

```
mv_candidates = legal_moves(prev)                  # 已过滤自将着法
for mv in mv_candidates:
    trial = prev.copy()
    make(trial, mv)
    if count_diff(trial, cur) <= tolerance:        # tolerance = 1
        return mv
return None
```

`count_diff` 统计 `trial.grid != cur.grid` 的格数。**只在 `legal_moves` 里搜**，天然排除了「识别出的非法局面」——这正是 spec 说的"拒绝把非法局面喂给皮卡鱼"。

- [ ] **Step 4: 实现状态机**

- `MY_TURN`：若未 `my_side` 判定过，用首帧 `mapper` 判定（`auto` 模式）。
- `THINKING`：`analyse(board.to_fen(), cfg.movetime_ms)`；记录 `think_ms`；若有 `human_delay`，用 `interruptible_sleep(random.uniform(*human_delay), lambda: self._pause_requested or self.should_stop)`，被中断则回到 `MY_TURN` 且**不走子**。面板 `message` 设为 `f"THINKING (human delay {d:.1f}s)"`。
- `PLAYING`：`mouse.play_move(mv, geom_screen)`；随后 `recorder.add(...)`。
- `VERIFY`：重抓一帧（`board_provider()` 返回 `None` 时按失败处理）→ 与 `make(prev, mv)` 比对，**落点必须精确匹配**（一颗子走完一定在落点上，只放宽其余格子）；一致 → `OPPONENT_TURN`；不一致 → 在 `VERIFY_TIMEOUT_S = 5.0` 秒内继续等（游戏里棋子是滑过去的，吃子动画实测要 2~3 秒才停稳），超时仍不一致 → `PAUSED` 并 `message = "点击后局面校验失败，已暂停"`。
- `OPPONENT_TURN`：`cur = board_provider()`；`cur is None`（抓图失败/白帧）→ **直接 return，不改动任何状态**（`[RF-2]`）；`interpret(prev, cur)` 得着法 → 采纳并 `recorder.add` → `MY_TURN`。
  连续 3 次 `interpret` 返回 `None` 且 `cur != prev` → `PAUSED` 并 `message = "无法解释局面变化，已暂停（请检查识别）"`。
- `IDLE`：`board_provider()` 返回的局面与 `START_FEN` 一致（或棋子数 ≥ 28 且双方将帅在位）→ 判定开局，转 `MY_TURN`。
- **终局判定**：维护 `self._idle_since`（局面最后一次变化的时刻）。
  - （**已被真机推翻，见 Task 19 Ruling「静止≠终局」**）`state is OPPONENT_TURN` 且 `now - _idle_since > cfg.opponent_idle_sec` → 只**暂停**，绝不 `GAME_OVER`；且 `opponent_idle_sec <= 0`（默认）时整项关闭。
  - `state is MY_TURN` 且超时 → **不得进 `GAME_OVER`**；改为 `PAUSED`，`message = "轮到我方但局面异常静止，请检查"`（`[RF-4]`）。
- `pause()` 置 `_pause_requested`；`step()` 开头若已暂停则立刻 return（保证 `F8` 即时生效）。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_session.py -v`
Expected: 全 PASS

- [ ] **Step 6: 跑全量测试**

Run: `uv run pytest -v`
Expected: 全绿

- [ ] **Step 7: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(session): 状态机 + 合法着法解释器纠错 + 可中断拟人延迟"
```

---

### Task 17: `ui/panel.py` + `ui/hotkeys.py` —— 面板与热键

**Files:**
- Create: `src/jjchess/ui/__init__.py`、`src/jjchess/ui/panel.py`、`src/jjchess/ui/hotkeys.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: Task 2 `PIECE_CHARS`、Task 5 `BoardGeometry`、Task 13 `Recognition`、Task 16 `Session`/`State`
- Produces:
  - `class Panel`：`__init__(self, enabled: bool = True, width: int = 1500, height: int = 900)`、`render(self, frame: np.ndarray | None, geom: BoardGeometry | None, rec: Recognition | None, session: Session) -> np.ndarray`（**始终返回合成图 ndarray**，便于无头测试）、`close(self) -> None`
  - `class Hotkeys`：`poll(self) -> set[str]`（返回 `{"pause","step","dryrun","quit"}` 的子集）、`class FakeHotkeys`（测试用，构造时给定一串待发按键）
  - `def compose(frame, geom, rec, session, width, height) -> np.ndarray`（**纯函数，可无头测试**）

- [ ] **Step 1: 写失败测试 `tests/test_panel.py`**

```python
def test_合成图尺寸固定():
    out = compose(None, None, None, _fake_session(), 1500, 900)
    assert out.shape == (900, 1500, 3)

def test_有帧时左半区非空():
    frame = cv2.imread("tests/fixtures/opening.png")
    geom = locate(frame)
    out = compose(frame, geom, _fake_recognition(), _fake_session(), 1500, 900)
    assert out[:, :900].std() > 5          # 左半不是纯色

def test_右半含关键信息文字():
    s = _fake_session(state=State.THINKING, message="THINKING (human delay 1.3s)")
    out = compose(None, None, None, s, 1500, 900)
    assert out[:, 900:].std() > 5          # 右半确实画了字

def test_无帧时不崩溃():
    assert compose(None, None, None, _fake_session(), 1500, 900) is not None

def test_热键轮询去重():
    hk = FakeHotkeys(["pause", "pause", "quit"])
    assert hk.poll() == {"pause"} and hk.poll() == {"pause"} and hk.poll() == {"quit"}
```

`_fake_session()` / `_fake_recognition()` 为本文件内的轻量替身（用 `types.SimpleNamespace` 或 Task 16 的测试构造器）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_panel.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'jjchess.ui'`

- [ ] **Step 3: 实现 `compose` 与 `Panel`**

- 左半：`frame` 缩放到 `(width*0.6, height)`；`geom` 非空时在每个交叉点画识别字符与置信度色框（`>0.75` 绿 / `0.60~0.75` 黄 / `<0.60` 红）；`rec.unknown` 的格子画红叉。
- 右半：用 `cv2.putText` 逐行输出中文信息（**OpenCV 的 `putText` 不支持中文**，用 `cv2.freetype`？—— 不引入新依赖，改为：中文用 PIL？也不行）。
  **决策**：`cv2.putText` 只画 ASCII。中文信息（执色、状态说明）用 `cv2.putText` 的 ASCII 等价形式输出（如 `SIDE: RED`、`STATE: THINKING`），中文仅出现在**终端日志与导出的记谱文件**里。这样零新依赖，且 panel 的信息在调试场景足够。
- 底部状态条：`RUNNING / PAUSED / DRY-RUN / GAME OVER` + `session.message`（ASCII 化）。
- `Panel.render` 在 `enabled=True` 时调 `cv2.imshow("JJ象棋 自动下棋", out)` + `cv2.waitKey(1)`。
- `Hotkeys.poll`：用 `GetAsyncKeyState` 读 `VK_F8(0x77)`、`VK_F9(0x78)`、`VK_F10(0x79)`、`VK_ESCAPE(0x1B)`，**只在按下沿触发**（记录上帧状态做边沿检测），同一帧内去重。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_panel.py -v`
Expected: 5 passed

- [ ] **Step 5: 目视确认面板**

```powershell
uv run python -c "import cv2,numpy as np;from jjchess.ui.panel import compose;from jjchess.window import find_window;from jjchess.vision.locator import locate;from jjchess.vision.templates import TemplateSet;from jjchess.vision.recognizer import recognize;from jjchess.geometry import ScreenMapper;from pathlib import Path;w=find_window('JJ象棋',2.0);img=w.grab();g=locate(img);ts=TemplateSet.load(Path('config/templates.npz'));rec=recognize(img,g,ts,ScreenMapper(1));import types;out=compose(img,g,rec,types.SimpleNamespace(state='IDLE',message='',board=rec.board,last_analysis=None),1500,900);cv2.imwrite('_probe/panel.png',out)"
```
read_image 查看 `_probe/panel.png`：确认左侧棋盘上每个棋子都被标注且**标注与棋子一致**。

- [ ] **Step 6: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat(ui): OpenCV 实时面板与全局热键（F8/F9/F10/ESC）"
```

---

### Task 18: `main.py` —— 组装、CLI 与主循环

**Files:**
- Create: 补全 `src/jjchess/main.py`
- Modify: `src/jjchess/__main__.py`（Task 1 已建，此处确认无误）
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: 前面全部任务
- Produces: `def build_parser() -> argparse.ArgumentParser`、`def main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: 写失败测试 `tests/test_cli.py`**

```python
def test_help_不崩溃(capsys):
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["--help"])
    assert ei.value.code == 0
    assert "--human-delay" in capsys.readouterr().out

def test_默认值():
    a = build_parser().parse_args([])
    assert a.side == "auto" and a.movetime == 1200 and a.threads == 8 and a.hash == 512
    assert a.dry_run is False and a.no_panel is False and a.no_record is False

def test_human_delay裸开关():
    assert parse_human_delay(build_parser().parse_args(["--human-delay"]).human_delay) == (0.5, 2.0)

def test_human_delay带区间():
    assert parse_human_delay(build_parser().parse_args(["--human-delay", "1-4"]).human_delay) == (1.0, 4.0)

def test_非法side被拒(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--side", "green"])

def test_找不到窗口时返回非零并打印中文提示(monkeypatch, capsys):
    monkeypatch.setattr("jjchess.main.find_window",
                        lambda *a, **k: (_ for _ in ()).throw(WindowNotFound("未找到标题为「JJ象棋」的窗口，请先打开微信小游戏")))
    assert main(["--no-panel"]) == 2
    assert "未找到" in capsys.readouterr().err
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL —— `AttributeError: module 'jjchess.main' has no attribute 'build_parser'`

- [ ] **Step 3: 实现 CLI**

```
uv run jjchess [--side auto|red|black] [--movetime 1200] [--threads 8] [--hash 512]
               [--dry-run] [--no-panel] [--calibrate] [--debug-dir DIR]
               [--human-delay [MIN-MAX]] [--no-record] [--config PATH]
               [--rounds N]        # 后加：连下几局；0（默认）= 一直下
```
`--human-delay` 用 `nargs="?"`, `const=""`（裸开关传空串，交给 `parse_human_delay` 解释）；`--side` 用 `choices=["auto","red","black"]`。

- [ ] **Step 4: 实现主循环**

```
1. cfg = load_config(args.config, vars(args) 映射)
2. window = find_window("JJ象棋", timeout_s=10)      # WindowNotFound -> stderr 中文提示, return 2
3. 定位 + 识别循环，直到拿到开局局面：
     img = window.grab()
     if is_blank_frame(img): 退避重试（0.2s，最多 5 次），仍失败则切换提示
     geom = locate(img)  # BoardNotFound -> 等待 1s 重试（[RF-3] 大厅/菜单场景）
     rec  = recognize(...)
     [RF-5] 每轮比对 window.client_size；与上次不同 -> window.refresh() 并重新 locate()
4. mapper = ScreenMapper(my_side)   # auto 时由开局局面判定
5. session = Session(...); panel = Panel(enabled=not args.no_panel); hotkeys = Hotkeys()
6. while not session.should_stop:
     for k in hotkeys.poll():
         F8 -> session.pause()/resume();  F9 -> session.step_once();  F10 -> mouse.dry_run 取反;  ESC -> session.request_stop()
     img = window.grab(); session.step()
     panel.render(img, geom, rec, session)
7. 退出时：recorder.save(...)（除非 --no-record）；打印中文总结
8. return 0
```
`--calibrate` 时直接转交 `tools/calibrate.py` 的逻辑（导入其 `main`）后返回。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 6 passed

- [ ] **Step 6: 跑全量测试 + ruff**

```powershell
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```
Expected: 全绿

- [ ] **Step 7: 提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "feat: CLI 与主循环组装（含窗口/棋盘失效自动重定位）"
```

---

### Task 19: 端到端 dry-run 实战演练

**Files:**
- Create: `README.md`（补全使用说明）
- Test: 手工演练（无自动化断言；证据留存于 `_probe/`）

- [ ] **Step 1: 先跑 dry-run，全程不点击**

请用户把 JJ象棋 切到「人机对战」的开局局面，然后：

```powershell
uv run jjchess --dry-run --no-panel --debug-dir _probe/e2e
```
Expected（观察 60 秒）：
- 识别出开局局面，判定我方执色
- 我方回合打印引擎 bestmove 与 `[dry-run] 点击 (...)` 坐标
- **鼠标完全没动**
- 终端无异常堆栈

- [ ] **Step 2: 用面板确认识别叠加正确**

```powershell
uv run jjchess --dry-run --debug-dir _probe/e2e
```
Expected：面板左半每个棋子标注与实际一致，右半显示 DEPTH/SCORE/BESTMOVE 持续更新，底部状态条为 `DRY-RUN`。按 `F8` 应立即变 `PAUSED`，`ESC` 立即退出。

- [ ] **Step 3: 验证拟人化延迟与记谱**

```powershell
uv run jjchess --dry-run --human-delay 1-3 --debug-dir _probe/e2e
```
Expected：每步在引擎返回后额外停 1~3 秒，面板短暂显示 `THINKING (human delay x.xs)`；此期间按 `F8` **立即**响应。退出后 `_probe/e2e/game-*.txt` 存在且记谱与实际着法一致。

- [ ] **Step 4: 真机实走一局（去掉 --dry-run）**

```powershell
uv run jjchess --human-delay --debug-dir _probe/e2e
```
⚠️ 提示用户：**鼠标会被脚本接管**，需要临时别动鼠标；随时可按 `F8` 暂停、`ESC` 退出。
Expected：脚本自动走完我方每一步；对方（人机 AI）走完后脚本能自动识别并继续；终局或 `ESC` 后生成记谱文件。

- [ ] **Step 5: 写 `README.md` 使用说明（中文）**

内容：环境要求、`uv sync`、`python tools/fetch_pikafish.py`、`tools/calibrate.py --build`、各 CLI 参数表、热键表、常见问题（找不到窗口 / 定位失败 / 引擎缺失 / 识别不准怎么办）。

- [ ] **Step 6: 最终提交**

```powershell
git add -A
git -c user.name=dev -c user.email=dev@local commit -m "docs: 使用说明与端到端 dry-run 验证记录"
```

---

## 自审记录

**1. Spec 覆盖**：§1 目标 → Task 1/18；§4.1 window → T9；§4.2 geometry → T5；§4.3 locator → T11；§4.4 templates/recognizer → T12/T13；§4.5 rules → T2/T3/T4；§4.6 uci + fetch → T7/T8；§4.7 control + human-delay → T14/T16；§4.8 session → T16；§4.9 record → T15；§4.10 panel → T17；§4.11 main → T18；§5.5 配置 → T6；§6 结构/依赖 → T1；§7 测试 → 各任务；§8 风险 → 各任务对策。**无遗漏**。

**2. 与 spec 的偏差（已刻意为之，均属填补 spec 未指派的空白）**：
- 新增 `src/jjchess/config.py`：spec §5.5 定义了配置文件但**没有模块负责加载**，且 `parse_human_delay` / `interruptible_sleep` 无归属。→ 前者入 `config.py`，后者入 `session.py`（它才是睡觉的那一方）。
- spec §4.4 的 `Recognition.board` 需要知道执色；本计划把 `mapper` 作为 `recognize()` 参数传入，`Recognition.board` 仍是 `Board`，形状不变。
- `Board.grid` 索引明确为 `[rank, file]`（rank 0 = 红方底线），使 `rules/` 完全与屏幕无关；spec §5.3 的屏幕映射公式原样搬到 `geometry.ScreenMapper`。二者等价（已逐条验算）。
- spec §7 的 `test_record.py` 原含「`--human-delay` 解析 / 可中断睡眠」两项，已按其真实归属拆到 `test_config.py` 与 `test_session.py`。
- 面板中文渲染：OpenCV `cv2.putText` 不支持中文，为守住「不引入新依赖」的约束，面板上的文字改用 ASCII 等价形式（`SIDE: RED` / `STATE: THINKING`），中文只出现在终端日志与导出的记谱文件中。

**3. 类型一致性**：`Analysis` 在 T7 定义、T8/T15/T16/T17 消费，字段名一致；`Board.at/set/pieces/copy/from_fen/to_fen` 在 T2 定义后被 T3/T4/T12/T13/T15/T16 一致使用；`BoardGeometry.cell_to_px/px_to_cell/with_offset/to_json/from_json/is_plausible` 在 T5 定义后被 T9/T11/T12/T13/T14/T17 一致使用；`TemplateSet.load/match`、`Recognition.board/confidence/unknown`、`GameRecorder.add/to_text/iccs_line/save`、`Session.state/board/last_analysis/message/step/pause/resume/request_stop/should_stop` 均单次定义、多处引用，无改名。

**4. Review Focus**：RF-1（T9 测试）、RF-2（T9 `is_blank_frame` 三条测试 + T16 `test_全白帧不改变已确认局面`）、RF-3（T11 `test_大厅帧判定无棋盘`）、RF-4（T16 `test_非我方回合静止超时_只暂停不终局`、`test_轮到我方时局面不变_暂停等人`、`test_默认不检查静止_对方想多久都等`）、RF-5（T9 `size_changed` + T18 Step 4 的重定位循环）。五项全部有测试落位。

**5. 比例**：本计划约 900 行，spec 约 580 行，比例合理；代码块只出现在「测试断言」与「算法不确定处」（perft 规则、投影峰拟合、UCI 读取时序、绝对坐标归一化），未逐行转写实现。
