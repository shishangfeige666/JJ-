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
