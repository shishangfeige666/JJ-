# JJ象棋自动下棋脚本

微信小游戏「JJ象棋」的自动对弈程序。抓取游戏窗口画面 → 识别棋盘上的 32 个棋子 →
用[皮卡鱼（Pikafish）](https://github.com/official-pikafish/Pikafish)引擎计算最佳着法
→ 自动点击鼠标走子 → 循环至终局，并提供实时可视化面板与中文记谱导出。

## 环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 11（只用 Win32 API，无跨平台支持） |
| Python | 3.12（由 uv 管理） |
| 包管理 | [uv](https://docs.astral.sh/uv/)（实测 0.12.18） |
| 游戏 | 微信电脑版打开「JJ象棋」小程序，窗口标题为 `JJ象棋` |

## 安装

```powershell
uv sync
```

依赖从**阿里云 PyPI 镜像**安装（已写入 `pyproject.toml` 的 `[[tool.uv.index]]`）。

引擎二进制不随仓库分发，需单独下载：

```powershell
uv run python tools/fetch_pikafish.py
```

它会从镜像下载 Pikafish 发布包、校验 sha256、并把 `pikafish.exe` 与 `pikafish.nnue`
两个文件安装到 `vendor/pikafish/`（**两个文件都必须存在**，NNUE 权重是独立文件）。

## 运行前提：进程完整性级别必须是 Medium

抓图和自动点击都要跨越 Windows 的 UIPI 边界去操作游戏窗口（`WeChatAppEx.exe`，
Medium 完整性级别）。**如果本程序跑在 Low 完整性级别，`PrintWindow` 会对任何窗口
返回 `ERROR_ACCESS_DENIED`（错误码 5）并抓到全零，`SendInput` 点击也会被静默丢弃。**

本机实测：项目目录曾被 DSH 沙箱标记为

```
Mandatory Label\Low Mandatory Level:(OI)(CI)(NW)
```

该标签被目录下所有文件继承，于是 `.venv\Scripts\python.exe` 也以 Low 级别启动。
修复（对项目目录本身执行一次即可，子项会跟随）：

```powershell
icacls "<项目目录>" /setintegritylevel Medium
```

自检（**JJ象棋 窗口开着**时跑）：

```powershell
uv run python tools/check_capture.py
```

它会打印本进程的完整性级别、实测一次 `SendInput`，再对每个可见窗口实测一次
`PrintWindow(flag=2)`。正常输出结尾是：

```
✓ 抓图正常：JJ象棋 窗口 hwnd=…，平均亮度 …（有真实画面）。
✓ 鼠标注入正常。抓图与点击两条链路都通了。
```

如果几乎所有窗口都 `ok=0 err=5`（`ERROR_ACCESS_DENIED`），就是进程还在 Low 级别——
重跑上面的 `icacls` 即可。如果只有 `SendInput：失败`，原因相同：Low 完整性进程
往 Medium 窗口注入鼠标事件会被 UIPI 拦下，症状是「脚本看得见棋盘却点不动」。

`SendInput` 的自检发的是一个**零位移鼠标移动**（等价于没动），不会碰到屏幕上的任何东西；
本工具绝不用真实左键点击做自检。

`tools/check_capture.py` 刻意只用 `ctypes` + `struct`、不 import `jjchess`、不依赖
numpy，所以在依赖装坏或换别的 Python 版本时也能跑，能把「环境问题」和「代码问题」
干净地分开。

## 使用

```powershell
uv run jjchess                 # 自动识别执子方并开始对弈
uv run jjchess --dry-run       # 只计算不点击（安全演练）
uv run jjchess --help          # 查看全部参数
```

第一次跑真机之前，建议先彩排一遍确认识别正常：

```powershell
uv run jjchess --dry-run --debug-dir debug
```

彩排**不会点鼠标**，所以画面上不会真的走子，校验环节会提示
「彩排：没真的点鼠标，局面不会变，已暂停」—— 这是预期行为，按 `F10` 关掉彩排即可
继续。彩排期间 `F10` 可以随时在「真点」和「不点」之间切换，用来试跑那些没把握的局面。

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--side {auto,red,black}` | `auto` | 我方执色。`auto` 从画面上「哪一方的子在自己这半边」自动判定 |
| `--movetime 毫秒` | `1200` | 每步思考时间。实战 1200 ms 已经比陪练 AI 强不少，调大更稳但更慢 |
| `--threads N` | `8` | 引擎线程数。按 CPU 核数给，给到 8 以上收益就很小了 |
| `--hash MB` | `512` | 引擎置换表大小 |
| `--dry-run` | 关 | 彩排：照常识别与思考，但不真的点鼠标 |
| `--no-panel` | 关 | 不显示实时可视化面板（无人值守时用，终端仍会逐行打印状态） |
| `--no-record` | 关 | 不导出中文记谱文件 |
| `--human-delay [MIN-MAX]` | 关 | 拟人化思考延迟（秒）。裸写 `--human-delay` 等价于 `0.5-2.0`，也可写 `--human-delay 1.2-3` |
| `--debug-dir 目录` | 无 | 记谱与故障帧写到这里；不指定时记谱落在 `records/` |
| `--calibrate` | — | 等价于跑一次 `tools/calibrate.py --build` 然后退出 |
| `--config 路径` | `config/default.json` | 配置文件。优先级：命令行 > 配置文件 > 内置默认值 |
| `--engine 路径` | `vendor/pikafish/pikafish.exe` | 引擎可执行文件 |
| `--nnue 路径` | `vendor/pikafish/pikafish.nnue` | NNUE 权重（**必须和 exe 放一起**） |

### 热键

全局热键，**游戏窗口有焦点时也能按**：

| 键 | 作用 |
|---|---|
| `F8` | 暂停 / 继续。暂停不丢局面认知，恢复后接着走 |
| `F9` | 单步：只推进一步状态机（用来盯着慢放看它到底在干什么） |
| `F10` | 切换彩排模式（真点 ⇄ 不点），跟着走完整局也不会碰到鼠标 |
| `ESC` | 退出，并把这一局的中文记谱保存下来 |

### 可视化面板

默认开启，1500×900 的独立窗口：

- **左侧**：当前帧叠加识别结果 —— 每格一个点，绿 = 可信（置信度 ≥ `conf_good`）、
  黄 = 勉强认出来（≥ `conf_min`）、红 = 认不出来（已按「空」处理，同时计入 `unknown`）；
  棋子上直接标出识别到的字。
- **右侧**：`STATE` / `SIDE` / `LOW`（最低置信度）/ `MSG` / `BEST`（引擎打分）/
  `PV`（主变）/ `FEN`。
- **底部状态条**：左绿右红表示轮到谁，「DRY-RUN」表示彩排模式。

⚠️ 面板上的文字是 **ASCII**（`SIDE: RED`、`STATE: THINKING`）—— OpenCV 的
`cv2.putText` 画不了中文，硬写会变方块。中文只出现在**终端日志**和**导出的记谱文件**里。

### 中文记谱导出

每一步都会写进 `<debug-dir>/game-YYYYmmdd-HHMMSS.txt`（不指定 `--debug-dir` 时是
`records/`），内容形如：

```
JJ象棋自动下棋记谱
开始时间：2026-10-11 17:19:03
我方执色：红
开始局面：rnbakabnr/9/1c5c1/...

  1. 红  炮二平五    c2e2    +18   深度 22
     黑  马8进7     h9g7    -12   深度 20
  2. 红  马二进三    b0c2     +9   深度 21
...
结果：胜（对局结束）
```

「中文着法」用标准记谱（`炮二平五` / `马8进7`），后面跟 ICCS 坐标、引擎打分和搜索深度。
不想要记谱就加 `--no-record`。

## 标定

正常情况下**不需要**手工标定：程序会自己定位棋盘、自己认棋子。模板集
（`config/templates.npz`）是个生成物，不入库 —— 第一次运行时如果它不存在，程序会
等一个标准开局画面，从那一帧现场采集 15 类模板。

要单独重建、或想看看识别到底对不对：

```powershell
uv run python tools/calibrate.py --build   # 抓一帧开局画面重建模板集
uv run python tools/calibrate.py --check   # 用现有模板集做一次全盘自检
uv run python tools/calibrate.py --manual  # 鼠标点两个交叉点手工标定几何
```

- `--build` 会同时写出 `config/templates_preview.png`（15 类模板一览）和
  `config/recognition_preview.png`（全盘识别结果 + 置信度着色）。**先看这两张图**
  再判断识别是不是真的准了。
- `--check` 把识别结果和当前画面并排画出来，逐格核对最快。
- 模板集只对**同一套棋子皮肤**有效。换皮肤、改窗口尺寸/缩放之后重建一次。

## 常见问题

**Q：启动就报「未找到标题为「JJ象棋」的窗口」。**
程序按窗口标题找游戏。请确认：微信电脑版已经打开 JJ象棋小程序（不是只在手机上看）、
小程序窗口没有最小化。窗口标题必须正好是 `JJ象棋` —— 用
`uv run python tools/check_capture.py` 可以看到当前所有可见窗口的实际标题。
如果你的标题不一样（换版本、换渠道），改 `src/jjchess/main.py` 顶部的
`WINDOW_TITLE` 常量即可。

**Q：脚本看得见棋盘但点不动鼠标 / 抓图全黑。**
进程完整性级别不是 Medium。见上面「运行前提」，跑一次
`icacls "<项目目录>" /setintegritylevel Medium`，再用 `tools/check_capture.py` 自检。

**Q：终端说「没找到皮卡鱼引擎 …，退回内置简易引擎（棋力很低）」。**
`vendor/pikafish/` 下缺文件（引擎二进制不入库）。跑一次
`uv run python tools/fetch_pikafish.py` 自动下载并校验 sha256。
注意 **`pikafish.exe` 和 `pikafish.nnue` 必须都在**：NNUE 权重是独立文件，
少了它引擎起不来。内置的简易引擎只是「没引擎也能跑通流程」的兜底，棋力约等于初学者。

**Q：一直显示「还没看到棋盘（可能停在大厅/菜单），等待对局开始…」。**
程序在等一个棋盘画面。请手动进到对局里（大厅、战绩页都识别不出棋盘）。
如果已经在对局里还这样，多半是棋盘定位失败 —— 加 `--debug-dir debug` 跑一遍，
看终端里有没有「棋盘定位：BoardGeometry(...)」。

**Q：提示「点击后局面校验失败，已暂停」。**
鼠标点下去之后画面没变成预期的样子。常见原因：游戏窗口被别的窗口挡住、
游戏在弹对话框、或者点击落在了别的显示器上。`--debug-dir` 会存下暂停那一刻的画面
（`frame-*-paused.png`），并同时打印 `[画面]`（识别看到的）和 `[模型]`（程序以为的）
两个 FEN，对着图一看就知道差在哪。按 `F8` 可以继续。

**Q：识别偶尔认错一个子。**
先看面板右侧的 `LOW`。如果 `LOW` 长期低于 `conf_min`，调 `config/default.json` 里的
`vision.conf_min` / `vision.conf_good`，或者用 `--calibrate` 重建模板集。
程序对零星识别误差有容忍（`session.tolerance_cells`），但**局面整体跑偏时会自动暂停
而不是硬走下去**，这是有意的。

**Q：怎么让它慢一点、别那么像机器？**
`--human-delay 1.5-4`。延迟期间 `F8` / `ESC` 依然立刻响应。

**Q：游戏结束之后脚本没退出。**
现在会：棋盘从画面上消失超过 10 秒就判定终局，写出记谱并退出。
如果只是窗口被挡住几秒又露出来，不会误判 —— 读到棋盘就把失联计时清零。

**Q：对方一直在想棋，脚本会不会等不及？**
不会。**「局面长时间不变」永远不判终局** —— 对手长考是常态（JJ象棋陪练单步能想两分钟
以上），棋盘没动推不出对局结束。默认连「静止多久就暂停」这项检查都是关的
（`config/default.json` 里的 `session.opponent_idle_sec = 0.0`），对方想多久就等多久。
如果你要无人值守、想加个看门狗，把它设成秒数（比如 `300`）：超时**只会暂停**
并提示「对方长时间没走棋」，按 `F8` 继续等 —— 绝不会把这局棋收掉。
真正的终局信号只有一个：**棋盘从画面上消失**（结算弹窗会把整个棋盘撤掉）。

**Q：跑起来 CPU 占用很高。**
把 `--threads` 降到 4、`--hash` 降到 128 试试。识别那一侧的开销很小，吃 CPU 的是引擎。

## 开发

```powershell
uv run pytest                  # 跑测试
uv run ruff check .            # 代码检查
uv run ruff format .           # 代码格式化
```

真机帧是识别层唯一的可信依据，所以 `tests/fixtures/` 里存的是**真机截图**而不是合成图：

- `opening.png` + `opening.fen`：标准开局，用来钉死「32 子全对」。
- `mid_01.png` + `mid_01.fen`：真机中局，**带光圈**（黑砲吃了马落在 h0，
  游戏给这颗子套了一圈绿光）。这是「只比棋子中心」那个修复的回归证据 ——
  整格比对时这一帧会漏掉 h0 的黑砲。
- `blank_frame.png` / `torn_frame.png`：两种坏帧（全黑、撕裂）。
  `PrintWindow` 偶发返回这两种画面，必须识别出来并丢掉，**绝不能当成「棋盘空了」**。

要补新的真机帧：

```powershell
uv run python tools/record_frames.py --name mid --watch
```

它会盯着游戏窗口，画面一变就存一帧。存下来的 `.png` 需要人工核对并配一份同名的
`.fen`，然后加进 `tests/test_vision.py` 的 `MID_CASES`。

## 许可证

本项目为学习与研究用途。皮卡鱼引擎以 GPL 授权，见 `vendor/pikafish/` 内的许可证文件。
请勿用于任何违反游戏用户协议或影响他人游戏体验的场景。
