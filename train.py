"""
AnimateDiff video generation pipeline.

Generates 16-frame motion clips via temporal attention.
Compel handles long prompts (>77 tokens).
"""

from __future__ import annotations

import logging
import os

import torch
from PIL import Image
from compel.compel import Compel
from diffusers import (
    AnimateDiffPipeline,
    DDIMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_video
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    AnimateDiff pipeline for text-to-video generation.

    Parameters
    ----------
    config:
        Parsed configuration dictionary (see ``config.yaml``).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.float16
            if self.device == "cuda" and config["diffusion"].get("dtype") == "fp16"
            else torch.float32
        )

        self.anim_pipe: AnimateDiffPipeline | None = None
        self.anim_compel: Compel | None = None

        self._load_animatediff_pipeline()

        logger.info("AnimateDiff pipeline ready on %s (%s).", self.device, self.dtype)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_animatediff_pipeline(self) -> None:
        base = self.config["diffusion"]["base_model"]
        module = self.config["animation"]["motion_module"]
        logger.info("Loading AnimateDiff adapter: %s", module)

        adapter = MotionAdapter.from_pretrained(module, torch_dtype=self.dtype)
        self.anim_pipe = AnimateDiffPipeline.from_pretrained(
            base, motion_adapter=adapter, torch_dtype=self.dtype, safety_checker=None
        )
        self.anim_pipe.scheduler = DDIMScheduler.from_config(
            self.anim_pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )

        if self.device == "cuda":
            self.anim_pipe.to("cuda")
            self.anim_pipe.enable_vae_slicing()
            self.anim_pipe.enable_attention_slicing(slice_size="auto")
        else:
            self.anim_pipe = self.anim_pipe.to(self.device)

        self.anim_compel = Compel(
            tokenizer=self.anim_pipe.tokenizer,
            text_encoder=self.anim_pipe.text_encoder,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, compel: Compel, prompt: str, negative: str):
        """
        Build length-matched positive/negative conditioning tensors.

        Compel pads tensors to equal length, which diffusers requires when
        the two prompts tokenise to different lengths.
        """
        pos = compel([prompt])
        neg = compel([negative]) if negative else compel([""])
        pos, neg = compel.pad_conditioning_tensors_to_same_length([pos, neg])
        return pos, neg

    # ------------------------------------------------------------------
    # Per-shot runner
    # ------------------------------------------------------------------

    def _run_animatediff(self, prompt: str, negative: str, seed: int) -> list[Image.Image]:
        pos, neg = self._embed(self.anim_compel, prompt, negative)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            output = self.anim_pipe(
                prompt_embeds=pos,
                negative_prompt_embeds=neg,
                num_frames=self.config["animation"]["num_frames"],
                height=self.config["diffusion"]["height"],
                width=self.config["diffusion"]["width"],
                num_inference_steps=self.config["animation"]["num_inference_steps"],
                guidance_scale=self.config["animation"]["guidance_scale"],
                generator=generator,
            )
        return output.frames[0]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, shots: list[dict], output_dir: str) -> list[dict]:
        """
        Generate a video clip for each shot descriptor.

        Parameters
        ----------
        shots:
            List of dicts with ``prompt`` key.
        output_dir:
            Directory to write ``.mp4`` files into.

        Returns
        -------
        list[dict]
            Input shots extended with a ``video_path`` key on success.
        """
        if not self.anim_pipe:
            raise RuntimeError("Pipeline not initialised.")

        base_prompt = self.config.get("style", {}).get("base_prompt", "")
        negative = self.config.get("style", {}).get("negative_prompt", "")
        base_seed = self.config["diffusion"]["seed"]
        fps = self.config["animation"]["fps"]

        completed: list[dict] = []
        bar = tqdm(shots, desc="Generating", unit="shot", ncols=88)

        for i, shot in enumerate(bar):
            bar.set_postfix(shot=i + 1)

            prompt = f"{base_prompt}, {shot['prompt']}" if base_prompt else shot["prompt"]
            seed = base_seed + i
            video_path = os.path.join(output_dir, f"shot_{i + 1:03d}.mp4")

            if self.device == "cuda":
                torch.cuda.empty_cache()

            try:
                frames = self._run_animatediff(prompt, negative, seed)
                export_to_video(frames, video_path, fps=fps)
                completed.append({**shot, "video_path": video_path})
                logger.debug("Shot %03d → %s", i + 1, video_path)

            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "CUDA OOM on shot %d — skipping. "
                    "Reduce num_frames or resolution in config.yaml.",
                    i + 1,
                )
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception:
                logger.exception("Unexpected error on shot %d — skipping.", i + 1)

        logger.info("%d/%d shot(s) rendered.", len(completed), len(shots))
        return completed

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all model weights and free GPU memory."""
        self.anim_pipe = None
        self.anim_compel = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("GPU memory released.")
