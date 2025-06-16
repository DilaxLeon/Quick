"""
MoviePy Configuration Utility

This module provides a centralized way to configure MoviePy settings
across all parts of the application.
"""

import os
from moviepy.config import change_settings

def configure_moviepy_temp_dir(temp_dir="/app/tmp"):
    """
    Configure MoviePy to use a specific temporary directory.
    
    Args:
        temp_dir (str): Path to the temporary directory. Defaults to "/app/tmp"
        
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        # Create the temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Configure MoviePy to use this directory
        change_settings({"TEMP_DIR": temp_dir})
        
        print(f"MoviePy configured to use temp directory: {temp_dir}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not configure MoviePy temp directory {temp_dir}: {e}")
        print("MoviePy will use default temp directory")
        return False

def configure_moviepy_ffmpeg(ffmpeg_path=None):
    """
    Configure MoviePy to use a specific ffmpeg binary.
    
    Args:
        ffmpeg_path (str): Path to the ffmpeg binary. If None, uses environment variable or default.
        
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        if ffmpeg_path is None:
            # Check if we're in production environment
            if os.environ.get('PRODUCTION') == 'true':
                ffmpeg_path = os.environ.get('FFMPEG_BINARY', '/usr/bin/ffmpeg')
            else:
                # Development mode - let MoviePy find ffmpeg automatically
                print("Development mode: Using system ffmpeg")
                return True
        
        change_settings({"FFMPEG_BINARY": ffmpeg_path})
        print(f"MoviePy configured to use ffmpeg binary at: {ffmpeg_path}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not configure MoviePy ffmpeg binary {ffmpeg_path}: {e}")
        print("MoviePy will use default ffmpeg")
        return False

def configure_moviepy_imagemagick(imagemagick_path=None):
    """
    Configure MoviePy to use a specific ImageMagick binary.
    
    Args:
        imagemagick_path (str): Path to the ImageMagick binary.
        
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        if imagemagick_path is None:
            # Default ImageMagick path for Windows development
            imagemagick_path = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
        
        if os.path.exists(imagemagick_path):
            change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
            print(f"MoviePy configured to use ImageMagick binary at: {imagemagick_path}")
            return True
        else:
            print(f"ImageMagick binary not found at: {imagemagick_path}")
            return False
        
    except Exception as e:
        print(f"Warning: Could not configure MoviePy ImageMagick binary {imagemagick_path}: {e}")
        print("MoviePy will use default ImageMagick")
        return False

def configure_moviepy_all(temp_dir="/app/tmp", ffmpeg_path=None, imagemagick_path=None):
    """
    Configure all MoviePy settings at once.
    
    Args:
        temp_dir (str): Path to the temporary directory
        ffmpeg_path (str): Path to the ffmpeg binary
        imagemagick_path (str): Path to the ImageMagick binary
        
    Returns:
        dict: Dictionary with configuration results
    """
    results = {
        'temp_dir': configure_moviepy_temp_dir(temp_dir),
        'ffmpeg': configure_moviepy_ffmpeg(ffmpeg_path),
        'imagemagick': configure_moviepy_imagemagick(imagemagick_path)
    }
    
    print(f"MoviePy configuration complete. Results: {results}")
    return results

# Auto-configure when module is imported
if __name__ != "__main__":
    # Only configure temp directory by default when imported
    configure_moviepy_temp_dir()