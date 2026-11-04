"""下载并安装皮卡鱼引擎（幂等）。

用法：`uv run python tools/fetch_pikafish.py`

下载源优先走 `gh-proxy.com` 镜像（实测 21.6 MB/s，直连只有 ~25 KB/s），失败再
回退 GitHub 直连。下载完必须过 **大小 + sha256** 校验——镜像不可信没关系，哈希
对得上就说明与官方发布字节一致。

解压靠 Windows 自带的 `tar.exe`（bsdtar 3.6.2，原生支持 7z），不引入额外依赖。
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jjchess.engine.uci import UciEngine, engine_present
from jjchess.rules.board import START_FEN

VENDOR = ROOT / "vendor"
EXE_PATH = VENDOR / "pikafish" / "pikafish.exe"
NNUE_PATH = VENDOR / "pikafish" / "pikafish.nnue"

ARCHIVE_NAME = "Pikafish.2026-09-06.7z"
EXPECTED_SIZE = 53603424
EXPECTED_SHA256 = "41952bbfe2520faceb5902c69e6ab4845cc999841d2b49a95cc1be7867a25e5b"

_RELEASE = (
    "https://github.com/official-pikafish/Pikafish/releases/download/"
    "Pikafish-2026-09-06/" + ARCHIVE_NAME
)
URLS = [
    "https://gh-proxy.com/" + _RELEASE,
    _RELEASE,
]

#: 7z 包里的文件名。
_MEMBER_EXE = "Pikafish-Windows-x86-64-universal.exe"
_MEMBER_NNUE = "pikafish.nnue"

_CURL_CANDIDATES = [Path(r"C:\Windows\System32\curl.exe"), Path("curl.exe")]
_TAR_CANDIDATES = [Path(r"C:\Windows\System32\tar.exe"), Path("tar.exe")]


def _tool(candidates: list[Path], flag: str, label: str) -> str:
    for cand in candidates:
        if cand.is_absolute():
            if cand.is_file():
                return str(cand)
        else:
            found = shutil.which(str(cand))
            if found:
                return found
    raise SystemExit(f"找不到 {label}，请确认系统自带该命令（本脚本需要 {flag}）")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> bool:
    curl = _tool(_CURL_CANDIDATES, "curl.exe", "curl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载中：{url}")
    proc = subprocess.run(
        [
            curl,
            "-L",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "-o",
            str(dest),
            url,
        ],
        check=False,
    )
    if proc.returncode != 0:
        print(f"curl 退出码 {proc.returncode}")
        return False
    if not dest.is_file():
        print("下载后文件不存在")
        return False
    if dest.stat().st_size != EXPECTED_SIZE:
        print(f"文件大小不符：{dest.stat().st_size} != {EXPECTED_SIZE}")
        return False
    return True


def _verify(path: Path) -> None:
    actual = _sha256(path)
    if actual != EXPECTED_SHA256:
        path.unlink(missing_ok=True)
        raise SystemExit(f"下载文件校验失败，请重试（sha256={actual}）")
    print("sha256 校验通过")


def _install(archive: Path) -> None:
    tar = _tool(_TAR_CANDIDATES, "tar.exe", "tar")
    extract_dir = VENDOR / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("解压中……")
    proc = subprocess.run([tar, "-xf", str(archive), "-C", str(extract_dir)], check=False)
    if proc.returncode != 0:
        raise SystemExit(f"解压失败（tar 退出码 {proc.returncode}）")

    src_exe = extract_dir / _MEMBER_EXE
    src_nnue = extract_dir / _MEMBER_NNUE
    for src in (src_exe, src_nnue):
        if not src.is_file():
            raise SystemExit(f"压缩包里找不到 {src.name}")

    EXE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_exe, EXE_PATH)
    # 权重是独立文件，必须和 exe 放在同一目录。
    shutil.copy2(src_nnue, NNUE_PATH)
    print(f"已安装：{EXE_PATH}")
    print(f"已安装：{NNUE_PATH}")


def _self_check() -> int:
    if not engine_present(EXE_PATH):
        print("自检失败：引擎可执行文件不存在")
        return 1
    with UciEngine(EXE_PATH, threads=4, hash_mb=128, nnue=NNUE_PATH) as engine:
        analysis = engine.analyse(START_FEN, movetime_ms=300)
    if not analysis.ok:
        print("自检失败：引擎没有给出着法")
        return 1
    print(f"自检通过：bestmove={analysis.bestmove} depth={analysis.depth}")
    return 0


def main() -> int:
    if engine_present(EXE_PATH) and NNUE_PATH.is_file():
        print(f"引擎已就绪：{EXE_PATH}")
        return _self_check()

    archive = VENDOR / ARCHIVE_NAME
    ok = False
    for url in URLS:
        if _download(url, archive):
            ok = True
            break
        print(f"该源失败：{url}")
    if not ok:
        raise SystemExit("所有下载源都失败了，请检查网络或代理")

    _verify(archive)
    _install(archive)
    return _self_check()


if __name__ == "__main__":
    raise SystemExit(main())
