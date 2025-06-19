#!/usr/bin/env python3
"""
GPU Configuration Test Script

This script tests the GPU configuration and reports on available acceleration.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(__file__))

def test_gpu_config():
    """Test GPU configuration and report results."""
    print("=" * 60)
    print("GPU CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        from gpu_config import gpu_config, is_gpu_available, get_whisper_device, get_yolo_device, get_pyannote_device
        
        print(f"✓ GPU config module imported successfully")
        print()
        
        # Test basic GPU availability
        print("BASIC GPU DETECTION:")
        print(f"  CUDA Available: {gpu_config.cuda_available}")
        print(f"  Device: {gpu_config.device}")
        print(f"  GPU Count: {gpu_config.gpu_count}")
        print(f"  GPU Memory: {gpu_config.gpu_memory}GB")
        print()
        
        # Test model-specific device selection
        print("MODEL DEVICE CONFIGURATION:")
        print(f"  Whisper device: {get_whisper_device()}")
        print(f"  YOLOv8 device: {get_yolo_device()}")
        print(f"  PyAnnote device: {get_pyannote_device()}")
        print()
        
        # Test FFmpeg GPU support
        print("FFMPEG GPU ACCELERATION:")
        print(f"  FFmpeg GPU Support: {gpu_config.ffmpeg_gpu_support}")
        if gpu_config.ffmpeg_gpu_support:
            gpu_params = gpu_config.get_ffmpeg_gpu_params()
            print(f"  GPU Parameters: {gpu_params}")
            
            encoder_params = gpu_config.get_ffmpeg_encoder_params('h264')
            print(f"  H264 Encoder: {encoder_params}")
        print()
        
        # Test OpenCV GPU support
        print("OPENCV GPU SUPPORT:")
        print(f"  OpenCV GPU Support: {gpu_config.opencv_gpu_support}")
        print()
        
        # Test video encoding parameters
        print("GPU VIDEO ENCODING PARAMETERS:")
        from gpu_config import get_gpu_video_params
        
        for quality in ['low', 'medium', 'high', 'ultra']:
            params = get_gpu_video_params('libx264', 'medium', quality)
            print(f"  {quality.upper()}: codec={params['codec']}, params={len(params['ffmpeg_params'])} items")
        print()
        
        print("=" * 60)
        if gpu_config.cuda_available:
            print("✓ GPU ACCELERATION AVAILABLE")
            print("  Your system supports GPU acceleration for:")
            print("  - PyTorch models (Whisper, PyAnnote)")
            print("  - YOLOv8 person detection")
            if gpu_config.ffmpeg_gpu_support:
                print("  - FFmpeg video encoding/decoding")
            if gpu_config.opencv_gpu_support:
                print("  - OpenCV operations")
        else:
            print("⚠ GPU ACCELERATION NOT AVAILABLE")
            print("  The system will use CPU for all operations.")
            print("  Consider installing CUDA and GPU-enabled libraries for better performance.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing GPU configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_gpu():
    """Test PyTorch GPU functionality."""
    print("\nTORCH GPU TEST:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
                
            # Test tensor operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                print(f"  ✓ GPU tensor operations working")
            except Exception as e:
                print(f"  ✗ GPU tensor operations failed: {e}")
        
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")

def test_whisper_gpu():
    """Test Whisper GPU functionality."""
    print("\nWHISPER GPU TEST:")
    try:
        import whisper
        from gpu_config import get_whisper_device
        
        device = get_whisper_device()
        print(f"  Whisper device: {device}")
        
        # Try to load a small model
        print(f"  Loading tiny model on {device}...")
        model = whisper.load_model("tiny", device=device)
        print(f"  ✓ Whisper model loaded successfully on {device}")
        
        # Clean up
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  ✗ Whisper GPU test failed: {e}")

if __name__ == "__main__":
    success = test_gpu_config()
    test_torch_gpu()
    test_whisper_gpu()
    
    if success:
        print("\n✓ GPU configuration test completed successfully!")
    else:
        print("\n✗ GPU configuration test failed!")
        sys.exit(1)