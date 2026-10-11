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

它会打印本进程的完整性级别，并对每个可见窗口实测一次 `PrintWindow(flag=2)`。
正常输出结尾是 `✓ 抓图正常：JJ象棋 窗口 hwnd=…`；如果几乎所有窗口都 `ok=0 err=5`
（`ERROR_ACCESS_DENIED`），就是进程还在 Low 级别——重跑上面的 `icacls` 即可。

`tools/check_capture.py` 刻意只用 `ctypes` + `struct`、不 import `jjchess`、不依赖
numpy，所以在依赖装坏或换别的 Python 版本时也能跑，能把「环境问题」和「代码问题」
干净地分开。

## 使用

```powershell
uv run jjchess                 # 自动识别执子方并开始对弈
uv run jjchess --dry-run       # 只计算不点击（安全演练）
uv run jjchess --help          # 查看全部参数
```

## 开发

```powershell
uv run pytest                  # 跑测试
uv run ruff check .            # 代码检查
```

## 许可证

本项目为学习与研究用途。皮卡鱼引擎以 GPL 授权，见 `vendor/pikafish/` 内的许可证文件。
请勿用于任何违反游戏用户协议或影响他人游戏体验的场景。
