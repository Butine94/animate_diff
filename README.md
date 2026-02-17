# animate_diff

Text-to-video generation using Stable Diffusion 1.5 and AnimateDiff.

## Installation
```bash
git clone https://github.com/Butine94/animate_diff.git
cd animate_diff
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Settings are in `config.yaml`. Key parameters:

- `diffusion.base_model` — SD1.5 checkpoint
- `animation.motion_module` — AnimateDiff adapter path
- `animation.num_frames` — frames per clip (8-24)
- `style.base_prompt` — prepended to all prompts

## Usage
```bash
python generate.py --prompts data/prompts.yaml
```

See `data/prompts.yaml` for prompt file format.
