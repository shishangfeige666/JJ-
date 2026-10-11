"""配置加载与 `--human-delay` 解析测试。"""

from pathlib import Path

import pytest

from jjchess.config import DEFAULTS, load_config, parse_human_delay

#: 仓库里的真实配置文件，相对 CWD（pytest 的 rootdir）。
DEFAULT_JSON = Path("config/default.json")


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
    cfg = load_config(DEFAULT_JSON, {"movetime_ms": 700})
    assert cfg.movetime_ms == 700
    assert cfg.threads == 8


def test_缺失配置文件走默认值(tmp_path):
    cfg = load_config(tmp_path / "nope.json", {})
    assert cfg.movetime_ms == 1200 and cfg.record is True


def test_None覆盖被跳过():
    cfg = load_config(DEFAULT_JSON, {"movetime_ms": None, "dry_run": None})
    assert cfg.movetime_ms == DEFAULTS.movetime_ms


def test_默认配置文件本身可解析():
    cfg = load_config(DEFAULT_JSON, {})
    assert cfg.engine_path == Path("vendor/pikafish/pikafish.exe")
    assert cfg.nnue_path == Path("vendor/pikafish/pikafish.nnue")
    assert cfg.movetime_ms == 1200
    assert cfg.threads == 8
    assert cfg.hash_mb == 512
    assert cfg.conf_min == 0.60
    assert cfg.conf_good == 0.75
    assert cfg.tolerance_cells == 1
    assert cfg.gameover_idle_sec == 12.0
    assert cfg.human_delay is None  # 文件里 enabled=false
    assert cfg.record is True


def test_配置文件里启用human_delay(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        '{"session": {"human_delay": {"enabled": true, "min_s": 1.0, "max_s": 3.5}}}',
        encoding="utf-8",
    )
    assert load_config(p, {}).human_delay == (1.0, 3.5)


def test_未知覆盖键被拒绝():
    with pytest.raises(ValueError):
        load_config(DEFAULT_JSON, {"no_such_option": 1})
