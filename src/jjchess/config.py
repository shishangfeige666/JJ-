"""配置：`config/default.json` 的加载、CLI 覆盖与 `--human-delay` 区间解析。

覆盖优先级是 **CLI > 配置文件 > `DEFAULTS`**。`overrides` 里值为 `None`
的键表示"CLI 没指定这个选项"，直接跳过，这样 `main.py` 可以把整个 argparse
结果丢进来而不用先过滤。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

#: 裸 `--human-delay`（不带区间）时用的默认区间，单位秒。
DEFAULT_HUMAN_DELAY = (0.5, 2.0)
#: 仓库里默认配置文件的相对路径。
DEFAULT_CONFIG_PATH = Path("config/default.json")

_HUMAN_DELAY_HINT = "--human-delay 格式应为 MIN-MAX，例如 0.5-2.0"


@dataclass
class Config:
    """全部可调参数。字段名与 CLI 的 `--flag`（下划线化）一一对应。"""

    engine_path: Path = Path("vendor/pikafish/pikafish.exe")
    nnue_path: Path = Path("vendor/pikafish/pikafish.nnue")
    movetime_ms: int = 1200
    threads: int = 8
    hash_mb: int = 512
    conf_min: float = 0.60
    conf_good: float = 0.75
    tolerance_cells: int = 1
    human_delay: tuple[float, float] | None = None
    gameover_idle_sec: float = 12.0
    debug_dir: Path | None = None
    record: bool = True


#: 缺省配置（配置文件缺失或字段缺省时的兜底）。
DEFAULTS = Config()

#: 允许被 `overrides` 覆盖的字段名。
_FIELD_NAMES = frozenset(f.name for f in fields(Config))


def parse_human_delay(arg: str | None) -> tuple[float, float] | None:
    """解析 `--human-delay` 的取值。

    - `None` —— 没传这个开关，返回 `None`（不拟人化）。
    - `""` —— 裸开关，用 `DEFAULT_HUMAN_DELAY`。
    - `"MIN-MAX"` —— 显式区间，要求 `MIN > 0` 且 `MIN < MAX`。
    """
    if arg is None:
        return None
    if arg.strip() == "":
        return DEFAULT_HUMAN_DELAY

    parts = arg.split("-")
    if len(parts) != 2:
        raise ValueError(_HUMAN_DELAY_HINT)
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(_HUMAN_DELAY_HINT) from exc
    if not (low > 0 and low < high):
        raise ValueError(_HUMAN_DELAY_HINT)
    return (low, high)


def _from_sections(data: dict) -> Config:
    """把配置文件的三段结构摊平成 `Config`。"""
    engine = data.get("engine", {})
    vision = data.get("vision", {})
    session = data.get("session", {})

    delay = session.get("human_delay") or {}
    human_delay = (float(delay["min_s"]), float(delay["max_s"])) if delay.get("enabled") else None

    debug_dir = session.get("debug_dir")

    return Config(
        engine_path=Path(engine.get("path", DEFAULTS.engine_path)),
        nnue_path=Path(engine.get("nnue", DEFAULTS.nnue_path)),
        movetime_ms=int(engine.get("movetime_ms", DEFAULTS.movetime_ms)),
        threads=int(engine.get("threads", DEFAULTS.threads)),
        hash_mb=int(engine.get("hash_mb", DEFAULTS.hash_mb)),
        conf_min=float(vision.get("conf_min", DEFAULTS.conf_min)),
        conf_good=float(vision.get("conf_good", DEFAULTS.conf_good)),
        tolerance_cells=int(session.get("tolerance_cells", DEFAULTS.tolerance_cells)),
        human_delay=human_delay,
        gameover_idle_sec=float(session.get("gameover_idle_sec", DEFAULTS.gameover_idle_sec)),
        debug_dir=Path(debug_dir) if debug_dir else None,
        record=bool(data.get("record", DEFAULTS.record)),
    )


def load_config(path: Path | None, overrides: dict) -> Config:
    """加载配置并应用 CLI 覆盖。

    `path` 为 `None` 或文件不存在时完全走 `DEFAULTS`（不报错——首次运行
    还没写配置文件是正常情况）。`overrides` 中值为 `None` 的键跳过；
    出现 `Config` 没有的键（多半是 CLI 拼错）则抛 `ValueError`。
    """
    data: dict = {}
    if path is not None and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))

    cfg = _from_sections(data)

    for key, value in overrides.items():
        if value is None:
            continue
        if key not in _FIELD_NAMES:
            raise ValueError(f"未知配置项：{key!r}")
        setattr(cfg, key, value)

    return cfg
