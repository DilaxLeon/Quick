"""
GPU Configuration Module

This module provides centralized GPU detection and configuration for all ML models
and video processing components in the application.

Supports:
- PyTorch models (Whisper, PyAnnote.audio, YOLOv8)
- FFmpeg hardware acceleration
- OpenCV GPU operations
"""

import os
import sys
import torch
import subprocess
import logging
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUConfig:
    """Centralized GPU configuration and detection."""
    
    def __init__(self):
        self.cuda_available = False
        self.device = "cpu"
        self.gpu_count = 0
        self.gpu_memory = 0
        self.ffmpeg_gpu_support = False
        self.opencv_gpu_support = False
        self._detect_gpu_capabilities()
    
    def _detect_gpu_capabilities(self):
        """Detect available GPU capabilities."""
        logger.info("Detecting GPU capabilities...")
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.device = "cuda"
            self.gpu_count = torch.cuda.device_count()
            
            # Get GPU memory info
            if self.gpu_count > 0:
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
                gpu_name = torch.cuda.get_device_properties(0).name
                logger.info(f"CUDA available: {gpu_name} with {self.gpu_memory}GB memory")
            else:
                logger.warning("CUDA available but no GPU devices found")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Check FFmpeg GPU support
        self._check_ffmpeg_gpu_support()
        
        # Check OpenCV GPU support
        self._check_opencv_gpu_support()
        
        # Log summary
        self._log_gpu_summary()
    
    def _check_ffmpeg_gpu_support(self):
        """Check if FFmpeg supports GPU acceleration."""
        try:
            # Check for NVIDIA hardware acceleration support
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                # Check for NVIDIA encoders
                nvidia_encoders = ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc']
                self.ffmpeg_gpu_support = any(encoder in output for encoder in nvidia_encoders)
                
                if self.ffmpeg_gpu_support:
                    logger.info("FFmpeg GPU acceleration (NVENC) available")
                else:
                    logger.info("FFmpeg GPU acceleration not available")
            else:
                logger.warning("Could not check FFmpeg GPU support")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Error checking FFmpeg GPU support: {e}")
            self.ffmpeg_gpu_support = False
    
    def _check_opencv_gpu_support(self):
        """Check if OpenCV supports GPU operations."""
        try:
            import cv2
            # Check if OpenCV was compiled with CUDA support
            build_info = cv2.getBuildInformation()
            self.opencv_gpu_support = 'CUDA:' in build_info and 'YES' in build_info.split('CUDA:')[1].split('\n')[0]
            
            if self.opencv_gpu_support:
                logger.info("OpenCV GPU support available")
            else:
                logger.info("OpenCV GPU support not available")
                
        except Exception as e:
            logger.warning(f"Error checking OpenCV GPU support: {e}")
            self.opencv_gpu_support = False
    
    def _log_gpu_summary(self):
        """Log a summary of GPU capabilities."""
        logger.info("=== GPU Configuration Summary ===")
        logger.info(f"CUDA Available: {self.cuda_available}")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU Count: {self.gpu_count}")
        logger.info(f"GPU Memory: {self.gpu_memory}GB")
        logger.info(f"FFmpeg GPU: {self.ffmpeg_gpu_support}")
        logger.info(f"OpenCV GPU: {self.opencv_gpu_support}")
        logger.info("================================")
    
    def get_torch_device(self) -> str:
        """Get the appropriate PyTorch device string."""
        return self.device
    
    def get_whisper_device(self) -> str:
        """Get the appropriate device for Whisper model."""
        # Whisper uses string device names
        return "cuda" if self.cuda_available else "cpu"
    
    def get_yolo_device(self) -> str:
        """Get the appropriate device for YOLOv8 model."""
        # YOLOv8 uses string device names or integers
        return "cuda" if self.cuda_available else "cpu"
    
    def get_pyannote_device(self) -> torch.device:
        """Get the appropriate device for PyAnnote.audio models."""
        # PyAnnote uses torch.device objects
        return torch.device(self.device)
    
    def get_ffmpeg_gpu_params(self) -> list:
        """Get FFmpeg parameters for GPU acceleration."""
        if not self.ffmpeg_gpu_support:
            return []
        
        # NVIDIA GPU acceleration parameters
        gpu_params = [
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda'
        ]
        
        return gpu_params
    
    def get_ffmpeg_encoder_params(self, codec='h264') -> list:
        """Get FFmpeg encoder parameters for GPU acceleration."""
        if not self.ffmpeg_gpu_support:
            return []
        
        if codec.lower() == 'h264':
            return ['-c:v', 'h264_nvenc']
        elif codec.lower() == 'hevc':
            return ['-c:v', 'hevc_nvenc']
        else:
            return []
    
    def get_ffmpeg_decoder_params(self, codec='h264') -> list:
        """Get FFmpeg decoder parameters for GPU acceleration."""
        if not self.ffmpeg_gpu_support:
            return []
        
        if codec.lower() == 'h264':
            return ['-c:v', 'h264_cuvid']
        elif codec.lower() == 'hevc':
            return ['-c:v', 'hevc_cuvid']
        else:
            return []
    
    def optimize_model_for_gpu(self, model, model_type: str = "pytorch"):
        """
        Optimize a model for GPU usage.
        
        Args:
            model: The model to optimize
            model_type: Type of model ("pytorch", "yolo", "whisper")
        
        Returns:
            Optimized model
        """
        if not self.cuda_available:
            logger.info(f"GPU not available, keeping {model_type} model on CPU")
            return model
        
        try:
            if model_type.lower() == "pytorch":
                # Standard PyTorch model
                model = model.to(self.device)
                logger.info(f"Moved PyTorch model to {self.device}")
                
            elif model_type.lower() == "yolo":
                # YOLOv8 model
                model.to(self.device)
                logger.info(f"Moved YOLOv8 model to {self.device}")
                
            elif model_type.lower() == "whisper":
                # Whisper models are handled during loading
                logger.info(f"Whisper model will use device: {self.get_whisper_device()}")
                
            return model
            
        except Exception as e:
            logger.warning(f"Failed to move {model_type} model to GPU: {e}")
            logger.info("Falling back to CPU")
            return model
    
    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """
        Get optimal batch size based on available GPU memory.
        
        Args:
            base_batch_size: Base batch size for CPU
            
        Returns:
            Optimal batch size
        """
        if not self.cuda_available:
            return base_batch_size
        
        # Estimate batch size based on GPU memory
        if self.gpu_memory >= 8:
            return base_batch_size * 4
        elif self.gpu_memory >= 4:
            return base_batch_size * 2
        else:
            return base_batch_size
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_gpu_accelerated_ffmpeg_params(self, codec='libx264', preset='medium', quality_level='medium'):
        """
        Get GPU-accelerated FFmpeg parameters for video encoding.
        
        Args:
            codec: Video codec ('libx264', 'hevc', etc.)
            preset: Encoding preset
            quality_level: Quality level ('low', 'medium', 'high', 'ultra')
            
        Returns:
            dict: Parameters for write_videofile with GPU acceleration
        """
        params = {
            'codec': codec,
            'preset': preset,
            'ffmpeg_params': []
        }
        
        if self.ffmpeg_gpu_support:
            # Add GPU acceleration parameters
            if codec == 'libx264':
                params['codec'] = 'h264_nvenc'
                params['ffmpeg_params'].extend([
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda'
                ])
            elif codec == 'libx265' or codec == 'hevc':
                params['codec'] = 'hevc_nvenc'
                params['ffmpeg_params'].extend([
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda'
                ])
            
            # Add quality settings for NVENC
            if quality_level == 'low':
                params['ffmpeg_params'].extend(['-preset', 'fast', '-cq', '28'])
            elif quality_level == 'medium':
                params['ffmpeg_params'].extend(['-preset', 'medium', '-cq', '23'])
            elif quality_level == 'high':
                params['ffmpeg_params'].extend(['-preset', 'slow', '-cq', '18'])
            elif quality_level == 'ultra':
                params['ffmpeg_params'].extend(['-preset', 'slow', '-cq', '15'])
            
            logger.info(f"Using GPU-accelerated encoding: {params['codec']}")
        else:
            logger.info(f"Using CPU encoding: {codec}")
        
        return params

# Global GPU configuration instance
gpu_config = GPUConfig()

# Convenience functions for easy access
def get_device() -> str:
    """Get the current device string."""
    return gpu_config.get_torch_device()

def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return gpu_config.cuda_available

def get_whisper_device() -> str:
    """Get device for Whisper."""
    return gpu_config.get_whisper_device()

def get_yolo_device() -> str:
    """Get device for YOLOv8."""
    return gpu_config.get_yolo_device()

def get_pyannote_device() -> torch.device:
    """Get device for PyAnnote."""
    return gpu_config.get_pyannote_device()

def get_ffmpeg_gpu_params() -> list:
    """Get FFmpeg GPU parameters."""
    return gpu_config.get_ffmpeg_gpu_params()

def optimize_model(model, model_type: str = "pytorch"):
    """Optimize model for GPU."""
    return gpu_config.optimize_model_for_gpu(model, model_type)

def clear_gpu_cache():
    """Clear GPU cache."""
    gpu_config.clear_gpu_cache()

def get_gpu_video_params(codec='libx264', preset='medium', quality='medium'):
    """Get GPU-accelerated video encoding parameters."""
    return gpu_config.get_gpu_accelerated_ffmpeg_params(codec, preset, quality)

def write_video_with_gpu_acceleration(clip, output_path, **kwargs):
    """
    Write video with GPU acceleration if available.
    
    Args:
        clip: MoviePy video clip
        output_path: Output file path
        **kwargs: Additional parameters for write_videofile
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get GPU parameters
        quality = kwargs.pop('quality_level', 'medium')
        codec = kwargs.get('codec', 'libx264')
        preset = kwargs.get('preset', 'medium')
        
        gpu_params = get_gpu_video_params(codec, preset, quality)
        
        # Update kwargs with GPU parameters
        kwargs.update({
            'codec': gpu_params['codec'],
            'preset': gpu_params['preset']
        })
        
        # Add FFmpeg parameters
        existing_ffmpeg_params = kwargs.get('ffmpeg_params', [])
        kwargs['ffmpeg_params'] = existing_ffmpeg_params + gpu_params['ffmpeg_params']
        
        # Write video
        clip.write_videofile(output_path, **kwargs)
        
        # Clear GPU cache after processing
        if gpu_config.cuda_available:
            clear_gpu_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Error writing video with GPU acceleration: {e}")
        
        # Fallback to CPU encoding
        try:
            logger.info("Falling back to CPU encoding...")
            # Remove GPU-specific parameters
            fallback_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['ffmpeg_params'] or not any('cuda' in str(p) for p in v)}
            fallback_kwargs['codec'] = 'libx264'  # Ensure CPU codec
            
            clip.write_videofile(output_path, **fallback_kwargs)
            return True
            
        except Exception as e2:
            logger.error(f"Error writing video with CPU fallback: {e2}")
            return False

# Initialize GPU configuration when module is imported
if __name__ != "__main__":
    logger.info("GPU configuration module loaded")