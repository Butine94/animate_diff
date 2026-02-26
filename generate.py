"""
AnimateDiff — text-to-video generation CLI.

Usage
-----
    python generate.py
    python generate.py --prompt "a wolf walks through fog at dawn"
    python generate.py --prompts data/prompts.yaml
    python generate.py --config config.yaml --output ./outputs --num-shots 5
    python generate.py --reference ./references/character.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from train import VideoPipeline
from utils.prompt_utils import load_prompts, prompts_from_config
from utils.video_utils import compile_sequence, save_metadata, setup_directories

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("diffusers", "transformers", "accelerate", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate video clips from text prompts via AnimateDiff.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", metavar="PATH",
                   help="YAML configuration file.")
    p.add_argument("--prompts", default=None, metavar="PATH",
                   help="YAML or plain-text prompt file (overrides config).")
    p.add_argument("--prompt", default=None, metavar="TEXT",
                   help="Single prompt — generates one shot.")
    p.add_argument("--num-shots", type=int, default=None, metavar="N",
                   help="Cap on number of shots.")
    p.add_argument("--fps", type=int, default=None, metavar="N",
                   help="Output frame rate.")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip final sequence compilation.")
    p.add_argument("--verbose", action="store_true",
                   help="Debug-level logging.")
    return p


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse config: %s", exc)
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)

    if args.output is not None:
        config["output"]["directory"] = args.output
    if args.fps is not None:
        config["animation"]["fps"] = args.fps
        config["character"]["use_ip_adapter"] = True

    output_dir = config["output"]["directory"]
    num_shots = args.num_shots or config["generation"]["max_shots"]
    setup_directories([output_dir])

    if args.prompt:
        shots = [{"id": 1, "prompt": args.prompt, "method": "animatediff"}]
    elif args.prompts:
        shots = load_prompts(args.prompts, max_shots=num_shots)
    else:
        prompts_path = config.get("generation", {}).get("prompts_path")
        shots = (
            load_prompts(prompts_path, max_shots=num_shots)
            if prompts_path
            else prompts_from_config(config, max_shots=num_shots)
        )

    if not shots:
        logger.error("No prompts resolved — aborting.")
        sys.exit(1)

    logger.info(
        "%d shot(s) | %dx%d | %d frames @ %d fps | character: %s",
        len(shots),
        config["diffusion"]["width"],
        config["diffusion"]["height"],
        config["animation"]["num_frames"],
        config["animation"]["fps"],
        "enabled" if config["character"].get("use_ip_adapter") else "disabled",
    )
    for s in shots:
        logger.info("  [%s] %s", s.get("method", "animatediff"), s["prompt"][:80])

    pipeline = VideoPipeline(config)
    try:
        completed = pipeline.generate(shots, output_dir)
    finally:
        pipeline.cleanup()

    save_metadata(
        {"config": config, "shots_completed": len(completed), "shots": completed},
        str(Path(output_dir) / "metadata.json"),
    )

    if args.no_compile or not completed:
        return

    clip_paths = [s["video_path"] for s in completed if "video_path" in s]
    final_path = str(Path(output_dir) / "sequence.mp4")

    logger.info("Compiling sequence → %s", final_path)
    if compile_sequence(clip_paths, final_path, config["output"]["final_fps"]):
        logger.info("Done. Final video: %s", final_path)
    else:
        logger.error("Compilation failed.")


if __name__ == "__main__":
    main()
