"""
Pre-commit health check for animatediff-pipeline.

Run this before pushing to GitHub:
    python scripts/check_repo.py

Checks:
  1. .gitkeep files exist in outputs/ and references/
  2. No API keys, tokens, or hardcoded personal paths in source files
  3. No generated MP4s, model weights, or large files tracked by git
  4. config.yaml has all required keys
  5. All Python files are syntax-valid
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PASS = "✅"
FAIL = "❌"

issues: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    symbol = PASS if ok else FAIL
    print(f"  {symbol} {label}" + (f": {detail}" if detail else ""))
    if not ok:
        issues.append(f"{label} — {detail}" if detail else label)


# ── 1. .gitkeep files ────────────────────────────────────────────────
print("\n[1] .gitkeep files")
for d in ["outputs", "references"]:
    target = ROOT / d
    target.mkdir(exist_ok=True)
    gitkeep = target / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
        check(f"{d}/.gitkeep", True, "created")
    else:
        check(f"{d}/.gitkeep", True, "exists")

# ── 2. No secrets or hardcoded paths ────────────────────────────────
print("\n[2] Secret and hardcoded-path scan")

SECRET_PATTERNS = [
    (r"hf_[A-Za-z0-9]{20,}", "HuggingFace token"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI API key"),
    (r"ghp_[A-Za-z0-9]{20,}", "GitHub PAT"),
    (r"(?<!['\"])(?<!\w)/home/[a-z_][\w-]+/", "hardcoded Linux home path"),
    (r"(?<!['\"])/Users/[A-Za-z][\w-]+/", "hardcoded macOS path"),
    (r"password\s*=\s*['\"][^'\"]{3,}", "hardcoded password"),
    (r"secret\s*=\s*['\"][^'\"]{3,}", "hardcoded secret"),
]

# Known safe strings that look like secrets but aren't
SAFE = {
    "runwayml", "guoyww", "h94", "animatediff", "stable-diffusion",
    "ip-adapter", "ip_adapter", "huggingface-hub", "YOUR_USERNAME",
}

scan_files = (
    list(ROOT.rglob("*.py"))
    + list(ROOT.rglob("*.yaml"))
    + list(ROOT.rglob("*.toml"))
    + list(ROOT.rglob("*.env"))
)
# Exclude venv, .cache, __pycache__
scan_files = [
    f for f in scan_files
    if not any(part in f.parts for part in ("venv", ".venv", ".cache", "__pycache__"))
]

found_secrets = False
for fpath in scan_files:
    src = fpath.read_text(errors="replace")
    for pattern, label in SECRET_PATTERNS:
        for m in re.finditer(pattern, src):
            val = m.group()
            if any(s in val for s in SAFE):
                continue
            rel = fpath.relative_to(ROOT)
            line = src[: m.start()].count("\n") + 1
            check(f"{rel}:{line}", False, f"{label}: {val[:40]}")
            found_secrets = True

if not found_secrets:
    check("No secrets or hardcoded paths found", True)

# ── 3. No large / binary files in tracked source dirs ───────────────
print("\n[3] Large / binary file check")

BANNED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv",
                     ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp"}
BANNED_SIZE_MB = 5

found_large = False
source_dirs = [ROOT / d for d in ["utils", "tests", "data"] if (ROOT / d).exists()]
source_dirs.append(ROOT)

for fpath in ROOT.rglob("*"):
    if not fpath.is_file():
        continue
    if any(part in fpath.parts for part in ("venv", ".venv", ".cache", "__pycache__", "outputs", "references")):
        continue
    size_mb = fpath.stat().st_size / (1024 * 1024)
    if fpath.suffix.lower() in BANNED_EXTENSIONS:
        rel = fpath.relative_to(ROOT)
        check(str(rel), False, f"binary/media file should not be committed ({fpath.suffix})")
        found_large = True
    elif size_mb > BANNED_SIZE_MB:
        rel = fpath.relative_to(ROOT)
        check(str(rel), False, f"file is {size_mb:.1f} MB — consider git-ignoring")
        found_large = True

if not found_large:
    check("No large or binary files in source tree", True)

# ── 4. config.yaml key coverage ─────────────────────────────────────
print("\n[4] config.yaml key coverage")
try:
    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = [
        ("diffusion", "base_model"), ("diffusion", "dtype"), ("diffusion", "seed"),
        ("diffusion", "height"), ("diffusion", "width"),
        ("animation", "motion_module"), ("animation", "num_frames"), ("animation", "fps"),
        ("animation", "guidance_scale"), ("animation", "num_inference_steps"),
        ("sd", "num_inference_steps"), ("sd", "guidance_scale"),
        ("character", "use_ip_adapter"), ("character", "ip_adapter_model"),
        ("character", "ip_adapter_weight"), ("character", "reference_image"),
        ("character", "scale"),
        ("style", "base_prompt"), ("style", "negative_prompt"),
        ("generation", "max_shots"), ("output", "directory"), ("output", "final_fps"),
    ]
    missing = [(s, k) for s, k in required if k not in cfg.get(s, {})]
    if missing:
        for s, k in missing:
            check(f"{s}.{k}", False, "missing from config.yaml")
    else:
        check("All required keys present", True)
except ImportError:
    check("pyyaml not installed", False, "run: pip install pyyaml")
except FileNotFoundError:
    check("config.yaml", False, "file not found")

# ── 5. Python syntax check ───────────────────────────────────────────
print("\n[5] Python syntax")
py_files = [
    f for f in ROOT.rglob("*.py")
    if not any(p in f.parts for p in ("venv", ".venv", "__pycache__"))
]
all_ok = True
for fpath in py_files:
    try:
        ast.parse(fpath.read_text())
    except SyntaxError as e:
        check(str(fpath.relative_to(ROOT)), False, str(e))
        all_ok = False
if all_ok:
    check(f"All {len(py_files)} Python files parse cleanly", True)

# ── Summary ──────────────────────────────────────────────────────────
print()
if issues:
    print(f"{'FAIL':─^50}")
    print(f"{len(issues)} issue(s) found — fix before pushing:\n")
    for i in issues:
        print(f"  • {i}")
    sys.exit(1)
else:
    print(f"{'PASS':─^50}")
    print("Repo is clean and ready to push.")