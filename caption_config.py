"""
Caption Rendering Configuration
Controls which rendering engine to use for captions
"""

import os

# Caption rendering configuration
CAPTION_CONFIG = {
    # Primary rendering engine: 'ffmpeg' or 'moviepy'
    'primary_engine': os.environ.get('CAPTION_ENGINE', 'ffmpeg'),
    
    # Enable fallback to MoviePy if FFmpeg fails
    'enable_fallback': os.environ.get('CAPTION_FALLBACK', 'true').lower() == 'true',
    
    # FFmpeg quality settings
    'ffmpeg_quality': {
        'free': 'fast',
        'basic': 'medium',
        'pro': 'high'
    },
    
    # Enable advanced FFmpeg features
    'enable_word_highlighting': os.environ.get('CAPTION_WORD_HIGHLIGHT', 'true').lower() == 'true',
    'enable_text_effects': os.environ.get('CAPTION_TEXT_EFFECTS', 'true').lower() == 'true',
    'enable_multi_threading': os.environ.get('CAPTION_MULTI_THREAD', 'true').lower() == 'true',
    
    # Hardware acceleration settings
    'enable_hardware_acceleration': os.environ.get('ENABLE_HARDWARE_ACCEL', 'true').lower() == 'true',
    'prefer_nvenc': os.environ.get('PREFER_NVENC', 'true').lower() == 'true',
    'nvenc_preset': os.environ.get('NVENC_PRESET', 'p4'),  # p1-p7, p4 is balanced
    'enable_nvdec': os.environ.get('ENABLE_NVDEC', 'true').lower() == 'true',
    
    # Font configuration
    'font_fallback_enabled': True,
    'font_cache_enabled': True,
    
    # Performance settings
    'max_concurrent_renders': int(os.environ.get('MAX_CONCURRENT_RENDERS', '2')),
    'temp_cleanup_enabled': True,
    
    # Debug settings
    'debug_mode': os.environ.get('CAPTION_DEBUG', 'false').lower() == 'true',
    'save_filter_commands': os.environ.get('SAVE_FILTER_COMMANDS', 'false').lower() == 'true'
}

def get_caption_config():
    """Get the current caption configuration"""
    return CAPTION_CONFIG.copy()

def set_caption_engine(engine):
    """Set the caption rendering engine"""
    if engine not in ['ffmpeg', 'moviepy']:
        raise ValueError("Engine must be 'ffmpeg' or 'moviepy'")
    CAPTION_CONFIG['primary_engine'] = engine

def is_ffmpeg_enabled():
    """Check if FFmpeg rendering is enabled"""
    return CAPTION_CONFIG['primary_engine'] == 'ffmpeg'

def is_fallback_enabled():
    """Check if fallback to MoviePy is enabled"""
    return CAPTION_CONFIG['enable_fallback']

def get_quality_for_plan(user_plan):
    """Get video quality setting for user plan"""
    return CAPTION_CONFIG['ffmpeg_quality'].get(user_plan.lower(), 'medium')

def print_config():
    """Print current configuration"""
    print("Caption Rendering Configuration:")
    print("-" * 40)
    for key, value in CAPTION_CONFIG.items():
        print(f"{key}: {value}")
    print("-" * 40)

def is_hardware_acceleration_enabled():
    """Check if hardware acceleration is enabled"""
    return CAPTION_CONFIG['enable_hardware_acceleration']

def get_nvenc_preset():
    """Get the configured NVENC preset"""
    return CAPTION_CONFIG.get('nvenc_preset', 'p4')

def is_nvdec_enabled():
    """Check if NVDEC is enabled"""
    return CAPTION_CONFIG.get('enable_nvdec', True)

# Environment variable documentation
ENV_VARS_DOC = """
Caption Rendering Environment Variables:

CAPTION_ENGINE: 'ffmpeg' or 'moviepy' (default: ffmpeg)
    - Primary rendering engine to use

CAPTION_FALLBACK: 'true' or 'false' (default: true)  
    - Enable fallback to MoviePy if FFmpeg fails

CAPTION_WORD_HIGHLIGHT: 'true' or 'false' (default: true)
    - Enable word-by-word highlighting animations

CAPTION_TEXT_EFFECTS: 'true' or 'false' (default: true)
    - Enable advanced text effects (shadows, outlines, etc.)

CAPTION_MULTI_THREAD: 'true' or 'false' (default: true)
    - Enable multi-threaded FFmpeg processing

MAX_CONCURRENT_RENDERS: integer (default: 2)
    - Maximum number of concurrent video renders

CAPTION_DEBUG: 'true' or 'false' (default: false)
    - Enable debug output for caption rendering

SAVE_FILTER_COMMANDS: 'true' or 'false' (default: false)
    - Save FFmpeg filter commands to files for debugging

ENABLE_HARDWARE_ACCEL: 'true' or 'false' (default: true)
    - Enable NVIDIA hardware acceleration (NVENC/NVDEC)

PREFER_NVENC: 'true' or 'false' (default: true)
    - Prefer NVENC over software encoding when available

NVENC_PRESET: 'p1' to 'p7' (default: p4)
    - NVENC encoding preset (p1=fastest, p7=best quality)

ENABLE_NVDEC: 'true' or 'false' (default: true)
    - Enable NVDEC hardware video decoding

Examples:
    export CAPTION_ENGINE=ffmpeg
    export ENABLE_HARDWARE_ACCEL=true
    export PREFER_NVENC=true
    export NVENC_PRESET=p4
    export CAPTION_DEBUG=true
"""