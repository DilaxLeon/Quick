"""
GPU-Accelerated Video Processing Module

This module provides GPU-accelerated video processing functions using FFmpeg
with NVIDIA NVENC/NVDEC support for better performance compared to CPU-bound MoviePy.
"""

import os
import subprocess
import json
import tempfile
import shutil
import time
from typing import Tuple, Optional, Dict, Any
from gpu_config import get_gpu_config, get_quality_preset, is_gpu_enabled


class GPUVideoProcessor:
    """GPU-accelerated video processor using FFmpeg with NVIDIA acceleration"""
    
    def __init__(self):
        self.config = get_gpu_config()
        self.gpu_enabled = is_gpu_enabled()
        self.gpu_available = self._check_gpu_support() if self.gpu_enabled else False
        self.ffmpeg_path = self._find_ffmpeg()
        
        print(f"GPU Video Processor initialized:")
        print(f"  - GPU Enabled: {self.gpu_enabled}")
        print(f"  - GPU Available: {self.gpu_available}")
        print(f"  - FFmpeg Path: {self.ffmpeg_path}")
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable path"""
        # Check configuration first
        if self.config.get('FFMPEG_PATH'):
            ffmpeg_path = self.config['FFMPEG_PATH']
            try:
                result = subprocess.run([ffmpeg_path, '-version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"Using configured FFmpeg at: {ffmpeg_path}")
                    return ffmpeg_path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                print(f"Configured FFmpeg path not working: {ffmpeg_path}")
        
        # Check common locations
        possible_paths = [
            'ffmpeg',  # System PATH
            '/usr/bin/ffmpeg',  # Linux
            '/usr/local/bin/ffmpeg',  # Linux/macOS
            'C:\\ffmpeg\\bin\\ffmpeg.exe',  # Windows
            os.environ.get('FFMPEG_BINARY', 'ffmpeg')  # Environment variable
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"Found FFmpeg at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        raise RuntimeError("FFmpeg not found. Please install FFmpeg with NVIDIA support.")
    
    def _check_gpu_support(self) -> bool:
        """Check if NVIDIA GPU acceleration is available"""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("NVIDIA GPU detected")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        print("No NVIDIA GPU detected, falling back to CPU processing")
        return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"FFprobe failed: {result.stderr}")
            
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise RuntimeError("No video stream found")
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream.get('r_frame_rate', '24/1')),
                'duration': float(info['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'pixel_format': video_stream.get('pix_fmt', 'unknown')
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {str(e)}")
    
    def ensure_even_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Ensure dimensions are even numbers (required for H.264 with yuv420p)"""
        width = int(width)
        height = int(height)
        
        # Make even
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
            
        return width, height
    
    def convert_to_9_16_ratio_gpu(self, input_path: str, output_path: str, 
                                  quality: str = "balanced") -> str:
        """
        Convert video to 9:16 aspect ratio using GPU-accelerated FFmpeg
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            quality: Quality preset - "fast", "balanced", or "high"
            
        Returns:
            Path to the converted video
        """
        try:
            # Get video information
            video_info = self.get_video_info(input_path)
            width = video_info['width']
            height = video_info['height']
            fps = video_info['fps']
            
            print(f"Input video: {width}x{height} @ {fps}fps")
            
            # Calculate aspect ratios
            current_ratio = width / height
            target_ratio = 9 / 16
            
            print(f"Current ratio: {current_ratio:.3f}, Target ratio: {target_ratio:.3f}")
            
            # If already 9:16 or taller, no need to crop
            if current_ratio <= target_ratio:
                print("Video already has 9:16 aspect ratio or taller, copying file...")
                shutil.copyfile(input_path, output_path)
                return output_path
            
            # Calculate new dimensions for center crop
            new_width = int(height * target_ratio)
            new_width, height = self.ensure_even_dimensions(new_width, height)
            
            # Calculate crop position (center crop)
            x_offset = (width - new_width) // 2
            
            print(f"Cropping to: {new_width}x{height} (offset: {x_offset}, 0)")
            
            # Build FFmpeg command
            cmd = [self.ffmpeg_path, '-y']  # -y to overwrite output
            
            # Input options
            if self.gpu_available:
                # Use hardware-accelerated decoding
                cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            
            cmd.extend(['-i', input_path])
            
            # Video filters
            if self.gpu_available:
                # GPU-accelerated cropping and scaling
                vf_filters = [
                    f'crop={new_width}:{height}:{x_offset}:0',
                    'hwdownload',  # Download from GPU memory
                    'format=nv12'  # Convert to CPU format
                ]
            else:
                # CPU-only cropping
                vf_filters = [f'crop={new_width}:{height}:{x_offset}:0']
            
            cmd.extend(['-vf', ','.join(vf_filters)])
            
            # Get quality preset from configuration
            preset_config = get_quality_preset(quality)
            
            # Encoding options based on quality preset and GPU availability
            if self.gpu_available:
                # GPU-accelerated encoding
                nvidia_settings = self.config['NVIDIA_SETTINGS']
                cmd.extend([
                    '-c:v', nvidia_settings['encoder'],
                    '-preset', preset_config['preset'],
                    '-cq', str(preset_config['cq']),
                    '-b:v', preset_config['bitrate'],
                    '-maxrate', preset_config['maxrate'],
                    '-bufsize', preset_config['bufsize']
                ])
                print(f"Using GPU encoding with {quality} quality preset")
            else:
                # CPU fallback encoding
                cpu_settings = self.config['CPU_FALLBACK']
                cmd.extend([
                    '-c:v', cpu_settings['encoder'],
                    '-preset', cpu_settings['preset'],
                    '-crf', str(preset_config['cq']),  # Use CRF instead of CQ for CPU
                    '-b:v', preset_config['bitrate']
                ])
                print(f"Using CPU encoding with {quality} quality preset")
            
            # Common encoding options
            cmd.extend([
                '-pix_fmt', 'yuv420p',  # Compatibility
                '-profile:v', 'high',   # H.264 profile
                '-level', '4.0',        # H.264 level
                '-movflags', '+faststart',  # Web optimization
                '-c:a', 'aac',          # Audio codec
                '-b:a', '128k',         # Audio bitrate
                '-ar', '44100',         # Audio sample rate
                '-ac', '2'              # Stereo audio
            ])
            
            # Output file
            cmd.append(output_path)
            
            print(f"FFmpeg command: {' '.join(cmd)}")
            
            # Performance monitoring
            start_time = time.time()
            input_size = os.path.getsize(input_path)
            
            # Execute FFmpeg with configured timeout
            timeout = self.config.get('CONVERSION_TIMEOUT', 300)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            conversion_time = time.time() - start_time
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Verify output file exists and has reasonable size
            if not os.path.exists(output_path):
                raise RuntimeError("Output file was not created")
            
            output_size = os.path.getsize(output_path)
            if output_size < 1024:  # Less than 1KB is suspicious
                raise RuntimeError(f"Output file is too small ({output_size} bytes)")
            
            print(f"Successfully converted video: {output_path} ({output_size} bytes)")
            return output_path
            
        except Exception as e:
            # Clean up partial output file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise RuntimeError(f"GPU video conversion failed: {str(e)}")


# Global instance
_gpu_processor = None

def get_gpu_processor() -> GPUVideoProcessor:
    """Get or create the global GPU processor instance"""
    global _gpu_processor
    if _gpu_processor is None:
        _gpu_processor = GPUVideoProcessor()
    return _gpu_processor


def convert_to_9_16_ratio_gpu(input_path: str, output_path: str, 
                              quality: str = "balanced") -> str:
    """
    GPU-accelerated conversion to 9:16 aspect ratio
    
    This function replaces the MoviePy-based convert_to_9_16_ratio function
    with a GPU-accelerated FFmpeg implementation.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        quality: Quality preset - "fast", "balanced", or "high"
        
    Returns:
        Path to the converted video file
        
    Raises:
        RuntimeError: If conversion fails
    """
    processor = get_gpu_processor()
    return processor.convert_to_9_16_ratio_gpu(input_path, output_path, quality)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video information using FFprobe"""
    processor = get_gpu_processor()
    return processor.get_video_info(video_path)


if __name__ == "__main__":
    # Test the GPU processor
    processor = GPUVideoProcessor()
    print(f"GPU available: {processor.gpu_available}")
    print(f"FFmpeg path: {processor.ffmpeg_path}")