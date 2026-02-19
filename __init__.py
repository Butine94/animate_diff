"""
animatediff-pipeline — text-to-video generation via AnimateDiff.
"""

try:
    from .train import VideoPipeline  # requires torch + diffusers
    __all__ = ["VideoPipeline"]
except ImportError:
    pass

__version__ = "0.1.0"