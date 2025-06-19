"""
GPU Configuration Module

This module provides configuration options for GPU-accelerated video processing.
"""

import os
from typing import Dict, Any

# GPU Processing Configuration
GPU_CONFIG = {
    # Enable/disable GPU acceleration
    'ENABLE_GPU': os.environ.get('ENABLE_GPU', 'true').lower() == 'true',
    
    # Quality presets for different use cases
    'QUALITY_PRESETS': {
        'fast': {
            'description': 'Fast processing, lower quality',
            'cq': 28,  # Constant quality (higher = lower quality)
            'bitrate': '3M',
            'maxrate': '4M',
            'bufsize': '8M',
            'preset': 'fast'
        },
        'balanced': {
            'description': 'Balanced quality and speed (recommended)',
            'cq': 23,
            'bitrate': '4M',
            'maxrate': '6M',
            'bufsize': '12M',
            'preset': 'medium'
        },
        'high': {
            'description': 'High quality, slower processing',
            'cq': 20,
            'bitrate': '6M',
            'maxrate': '8M',
            'bufsize': '16M',
            'preset': 'slow'
        }
    },
    
    # Default quality preset
    'DEFAULT_QUALITY': os.environ.get('GPU_QUALITY_PRESET', 'balanced'),
    
    # FFmpeg paths (auto-detected if not specified)
    'FFMPEG_PATH': os.environ.get('FFMPEG_PATH', None),
    'FFPROBE_PATH': os.environ.get('FFPROBE_PATH', None),
    
    # GPU-specific settings
    'NVIDIA_SETTINGS': {
        'decoder': 'cuda',
        'encoder': 'h264_nvenc',
        'hwaccel_output_format': 'cuda'
    },
    
    # Fallback settings for CPU processing
    'CPU_FALLBACK': {
        'encoder': 'libx264',
        'preset': 'medium',
        'crf': 23
    },
    
    # Performance monitoring
    'ENABLE_PERFORMANCE_LOGGING': os.environ.get('GPU_PERF_LOG', 'false').lower() == 'true',
    
    # Timeout settings (in seconds)
    'CONVERSION_TIMEOUT': int(os.environ.get('GPU_TIMEOUT', '300')),  # 5 minutes
    'PROBE_TIMEOUT': int(os.environ.get('PROBE_TIMEOUT', '30')),      # 30 seconds
}

def get_gpu_config() -> Dict[str, Any]:
    """Get the current GPU configuration"""
    return GPU_CONFIG.copy()

def get_quality_preset(preset_name: str) -> Dict[str, Any]:
    """Get quality preset configuration"""
    presets = GPU_CONFIG['QUALITY_PRESETS']
    if preset_name not in presets:
        print(f"Warning: Unknown quality preset '{preset_name}', using 'balanced'")
        preset_name = 'balanced'
    
    return presets[preset_name].copy()

def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is enabled"""
    return GPU_CONFIG['ENABLE_GPU']

def get_default_quality() -> str:
    """Get the default quality preset name"""
    return GPU_CONFIG['DEFAULT_QUALITY']

def update_config(**kwargs):
    """Update GPU configuration at runtime"""
    for key, value in kwargs.items():
        if key in GPU_CONFIG:
            GPU_CONFIG[key] = value
            print(f"Updated GPU config: {key} = {value}")
        else:
            print(f"Warning: Unknown config key '{key}'")

def print_config():
    """Print current GPU configuration"""
    print("GPU Configuration:")
    print("=" * 30)
    print(f"GPU Enabled: {GPU_CONFIG['ENABLE_GPU']}")
    print(f"Default Quality: {GPU_CONFIG['DEFAULT_QUALITY']}")
    print(f"Performance Logging: {GPU_CONFIG['ENABLE_PERFORMANCE_LOGGING']}")
    print(f"Conversion Timeout: {GPU_CONFIG['CONVERSION_TIMEOUT']}s")
    
    print("\nQuality Presets:")
    for name, preset in GPU_CONFIG['QUALITY_PRESETS'].items():
        print(f"  {name}: {preset['description']}")
        print(f"    - Quality: {preset['cq']} (lower = better)")
        print(f"    - Bitrate: {preset['bitrate']}")
        print(f"    - Preset: {preset['preset']}")

if __name__ == "__main__":
    print_config()