"""
AnimateDiff video generation pipeline.

Supports two generation methods per shot:
  - ``animatediff``  — 16-frame motion clips via temporal attention.
  - ``sd_still``     — single high-fidelity keyframe, held to num_frames.

Optional IP-Adapter character consistency applies to sd_still shots.
Compel handles long prompts (>77 tokens) for both methods.
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
    StableDiffusionPipeline,
)
from diffusers.utils import export_to_video
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Hybrid SD1.5 + AnimateDiff pipeline.

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

        self.sd_pipe: StableDiffusionPipeline | None = None
        self.anim_pipe: AnimateDiffPipeline | None = None
        self.sd_compel: Compel | None = None
        self.anim_compel: Compel | None = None
        self.character_image: Image.Image | None = None

        self._load_sd_pipeline()
        self._load_animatediff_pipeline()

        if config["character"].get("use_ip_adapter"):
            self._load_character()

        logger.info("All pipelines ready on %s (%s).", self.device, self.dtype)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_sd_pipeline(self) -> None:
        base = self.config["diffusion"]["base_model"]
        logger.info("Loading SD1.5: %s", base)

        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=self.dtype, safety_checker=None
        ).to(self.device)

        if self.device == "cuda":
            self.sd_pipe.enable_vae_slicing()
            self.sd_pipe.enable_attention_slicing(slice_size="auto")

        self.sd_compel = Compel(
            tokenizer=self.sd_pipe.tokenizer,
            text_encoder=self.sd_pipe.text_encoder,
        )

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
            # Move pipeline to GPU for inference.
            self.anim_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            self.anim_pipe.enable_vae_slicing()
            self.anim_pipe.enable_attention_slicing(slice_size="auto")
        else:
            self.anim_pipe = self.anim_pipe.to(self.device)

        self.anim_compel = Compel(
            tokenizer=self.anim_pipe.tokenizer,
            text_encoder=self.anim_pipe.text_encoder,
        )

    def _load_character(self) -> None:
        ref = self.config["character"]["reference_image"]
        if not os.path.exists(ref):
            logger.warning("Character reference not found at '%s' — IP-Adapter disabled.", ref)
            return
        try:
            self.sd_pipe.load_ip_adapter(
                self.config["character"]["ip_adapter_model"],
                subfolder="models",
                weight_name=self.config["character"]["ip_adapter_weight"],
            )
            self.sd_pipe.set_ip_adapter_scale(self.config["character"]["scale"])
            self.character_image = Image.open(ref).convert("RGB").resize((512, 512))
            logger.info("IP-Adapter loaded. Reference: %s", ref)
        except Exception:
            logger.exception("IP-Adapter load failed — character consistency disabled.")

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
    # Per-shot runners
    # ------------------------------------------------------------------

    def _run_sd_still(self, prompt: str, negative: str, seed: int) -> list[Image.Image]:
        pos, neg = self._embed(self.sd_compel, prompt, negative)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        kwargs: dict = dict(
            prompt_embeds=pos,
            negative_prompt_embeds=neg,
            height=self.config["diffusion"]["height"],
            width=self.config["diffusion"]["width"],
            num_inference_steps=self.config["sd"]["num_inference_steps"],
            guidance_scale=self.config["sd"]["guidance_scale"],
            generator=generator,
        )
        if self.character_image is not None:
            kwargs["ip_adapter_image"] = self.character_image
        with torch.inference_mode():
            image = self.sd_pipe(**kwargs).images[0]
        # Hold the still for the full target frame count.
        return [image] * self.config["animation"]["num_frames"]

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
            List of dicts with ``prompt`` and optional ``method`` keys.
            ``method`` is ``"animatediff"`` (default) or ``"sd_still"``.
        output_dir:
            Directory to write ``.mp4`` files into.

        Returns
        -------
        list[dict]
            Input shots extended with a ``video_path`` key on success.
        """
        if not self.sd_pipe or not self.anim_pipe:
            raise RuntimeError("Pipelines not initialised.")

        base_prompt = self.config.get("style", {}).get("base_prompt", "")
        negative = self.config.get("style", {}).get("negative_prompt", "")
        base_seed = self.config["diffusion"]["seed"]
        fps = self.config["animation"]["fps"]

        completed: list[dict] = []
        bar = tqdm(shots, desc="Generating", unit="shot", ncols=88)

        for i, shot in enumerate(bar):
            method = shot.get("method", "animatediff")
            bar.set_postfix(method=method, shot=i + 1)

            prompt = f"{base_prompt}, {shot['prompt']}" if base_prompt else shot["prompt"]
            seed = base_seed + i
            video_path = os.path.join(output_dir, f"shot_{i + 1:03d}.mp4")

            if self.device == "cuda":
                torch.cuda.empty_cache()

            try:
                if method == "sd_still":
                    frames = self._run_sd_still(prompt, negative, seed)
                else:
                    frames = self._run_animatediff(prompt, negative, seed)

                export_to_video(frames, video_path, fps=fps)
                # Spread to avoid mutating the caller's dict.
                completed.append({**shot, "video_path": video_path})
                logger.debug("Shot %03d [%s] → %s", i + 1, method, video_path)

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
        for attr in ("sd_pipe", "anim_pipe", "sd_compel", "anim_compel", "character_image"):
            setattr(self, attr, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("GPU memory released.")