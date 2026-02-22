"""Video compilation and utilities"""
import os
import json
from pathlib import Path
from typing import List, Dict

def setup_directories(dir_list: List[str]) -> None:
    """Create required directories"""
    for directory in dir_list:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Directories ready: {', '.join(dir_list)}")

def read_text_file(filepath: str) -> str:
    """Read text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def save_metadata(data: Dict, filepath: str) -> bool:
    """Save metadata to JSON"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Metadata saved: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def compile_sequence(clip_paths: List[str], output_path: str, fps: int = 24) -> bool:
    """Compile video clips into final sequence"""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        print("MoviePy not installed. Install with: pip install moviepy")
        return False
    
    valid_clips = [p for p in clip_paths if Path(p).exists()]
    
    if not valid_clips:
        print("Error: No valid clips found")
        return False
    
    print(f"Compiling {len(valid_clips)} clips...")
    
    try:
        clips = [VideoFileClip(path) for path in valid_clips]
        final = concatenate_videoclips(clips, method="compose")
        
        final.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None,
            bitrate="5000k"
        )
        
        final.close()
        for clip in clips:
            clip.close()
        
        print(f"Sequence created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error compiling sequence: {e}")
        return False