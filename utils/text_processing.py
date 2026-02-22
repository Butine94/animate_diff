"""Text processing with motion detection"""
import re
from typing import List, Dict

SHOT_TYPES = ['wide establishing', 'medium tracking', 'close-up character', 'detail insert', 'dramatic reveal']

def parse_script(text: str, num_scenes: int = 5) -> List[Dict]:
    """Parse script into animated scenes"""
    sentences = split_into_segments(text)
    
    if not sentences or len(sentences) < num_scenes:
        print(f"Warning: Found {len(sentences)} sentences, need {num_scenes}. Using fallbacks.")
        return generate_fallback_scenes(num_scenes)
    
    return [
        {
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': sentences[i]
        }
        for i in range(num_scenes)
    ]

def split_into_segments(text: str) -> List[str]:
    """Split text into sentences"""
    text = re.sub(r'\s+', ' ', text).strip()
    segments = re.split(r'[.!?]+', text)
    return [s.strip() for s in segments if len(s.strip()) > 15]

def generate_fallback_scenes(num_scenes: int) -> List[Dict]:
    """Generate default noir scenes"""
    fallbacks = [
        "detective in fedora faces camera, front view, rain-soaked alley",
        "detective front-facing, looking at camera, rain falling steadily",
        "detective walks toward camera, frontal view, slow approach",
        "detective stops facing camera, lamp above, steam rising",
        "detective stares at camera, facing forward into darkness"
    ]
    
    return [
        {
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': fallbacks[i % len(fallbacks)]
        }
        for i in range(num_scenes)
    ]