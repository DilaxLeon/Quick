import os
import urllib.parse

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(dotenv_path):
        print(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path)
        print(f"HF_TOKEN is {'set' if 'HF_TOKEN' in os.environ else 'not set'}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables won't be loaded from .env file.")
    print("Please install python-dotenv: pip install python-dotenv")
import uuid
import random
import json
import time
import re
import math
import sys
import subprocess
from datetime import datetime
import functools
import mimetypes
import logging # For logging utilities
import werkzeug # For type checking response.response
import boto3
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
# json and sys are already imported globally
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, Response, session, current_app, send_file
from flask_cors import CORS
import whisper
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip
from moviepy.config import change_settings
import moviepy.config as mpc
from moviepy_config import configure_moviepy_all
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
from werkzeug.utils import secure_filename
from profanity_list import contains_profanity, find_profane_word, censor_profanity, contains_profane_phrase
from beep_generator import get_beep_for_duration
from ffmpeg_caption_renderer import FFmpegCaptionRenderer
from caption_config import get_caption_config, is_ffmpeg_enabled, is_fallback_enabled, get_quality_for_plan, is_hardware_acceleration_enabled, get_nvenc_preset, is_nvdec_enabled
# Speech enhancement feature removed

# Configure MoviePy settings (temp directory, ffmpeg, etc.)
# This ensures MoviePy writes temp files like TEMP_MPY_wvf_snd.mp4 to /app/tmp
configure_moviepy_all()

# Global variable to store the Whisper model
# We'll load it on demand to avoid startup issues
model = None



app = Flask(__name__)
# Configure CORS for production and development
allowed_origins = [
    "https://quickcap.pro",
    "https://www.quickcap.pro",
    # Add development origins for local testing
    "http://localhost:3000",
    "http://localhost:5173"
]

# More comprehensive CORS configuration
CORS(app, 
     resources={
         r"/*": {
             "origins": allowed_origins,
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
             "expose_headers": ["Content-Range", "X-Content-Range"],
             "supports_credentials": True,
             "max_age": 600
         }
     })

# Set a secret key for session management
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_key_for_testing_only')

# Configure Flask for longer video processing
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Decorator for logging route details
def log_route_details(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Ensure logger is configured if not already. This is a basic setup.
        if not current_app.logger.handlers and current_app.debug:
            # Add a basic handler if none exist and in debug mode.

            handler = logging.StreamHandler(sys.stdout) # Log to stdout
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            current_app.logger.addHandler(handler)
            current_app.logger.setLevel(logging.INFO) 

        logger = current_app.logger

        # Log request details
        log_message_request_parts = [f"Request: {request.method} {request.url}"]
        
        safe_headers = {}
        if request.headers:
            for k, v in request.headers.items():
                if k.lower() in ['authorization', 'cookie', 'x-api-key', 'x-csrf-token', 'x-auth-token', 'proxy-authorization', 'secret', 'password']:
                    safe_headers[k] = '***REDACTED***'
                elif k.lower() in ['user-agent', 'content-type', 'accept', 'referer', 'content-length', 'x-forwarded-for', 'x-request-id', 'origin', 'host']:
                    safe_headers[k] = v
            if safe_headers:
                 log_message_request_parts.append(f"Headers: {json.dumps(safe_headers)}")

        payload_logged = False
        if request.is_json:
            try:
                json_payload = request.get_json(silent=True)
                if json_payload is not None:
                    json_str = json.dumps(json_payload)
                    log_message_request_parts.append(f"JSON Payload: {json_str[:1000]}{'...' if len(json_str) > 1000 else ''}")
                    payload_logged = True
                else: 
                    data_bytes_for_json_none = request.get_data()
                    if data_bytes_for_json_none:
                        try:
                            data_str = data_bytes_for_json_none.decode('utf-8', errors='replace')
                            log_message_request_parts.append(f"Request Data (is_json=True, but no JSON payload, raw data): {data_str[:500]}{'...' if len(data_str) > 500 else ''}")
                        except Exception: 
                            log_message_request_parts.append("Request Data (is_json=True, but no JSON payload): (binary or non-utf8 data)")
                        payload_logged = True
                    else: 
                        log_message_request_parts.append("Request Data: (empty, Content-Type: application/json)")
                        payload_logged = True
            except Exception as e:
                log_message_request_parts.append(f"Error getting/logging JSON payload: {str(e)}")
                payload_logged = True 

        if not payload_logged and request.form:
            try:
                form_data_str = str(request.form.to_dict())
                log_message_request_parts.append(f"Form Data: {form_data_str[:1000]}{'...' if len(form_data_str) > 1000 else ''}")
                payload_logged = True
            except Exception as e:
                log_message_request_parts.append(f"Error logging Form Data: {str(e)}")
                payload_logged = True
        
        if not payload_logged and request.data: 
             try:
                decoded_data = request.data.decode('utf-8', errors='replace')
                log_message_request_parts.append(f"Request Data (raw): {decoded_data[:500]}{'...' if len(decoded_data) > 500 else ''}")
             except Exception as e: 
                log_message_request_parts.append(f"Request Data (raw): (Error decoding: {str(e)} or binary)")
        
        logger.info("\n".join(log_message_request_parts))

        response_val = None
        try:
            response_val = f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in route {request.method} {request.url}: {str(e)}", exc_info=True)
            raise

        log_message_response_parts = [f"Response: {request.method} {request.url}"]

        if response_val is not None:
            actual_response_obj = None
            status_code_val = None
            # headers_val = None # Not directly used for logging content, but good for context if needed

            if isinstance(response_val, Response):
                actual_response_obj = response_val
                status_code_val = actual_response_obj.status_code
                # headers_val = actual_response_obj.headers
            elif isinstance(response_val, tuple):
                try:
                    temp_resp = current_app.make_response(response_val)
                    actual_response_obj = temp_resp
                    status_code_val = temp_resp.status_code
                    # headers_val = temp_resp.headers
                except Exception:
                    log_message_response_parts.append(f"Raw Tuple Return: {str(response_val)[:1000]}")
            else: 
                try:
                    temp_resp = current_app.make_response(jsonify(response_val) if isinstance(response_val, (dict, list)) else str(response_val))
                    actual_response_obj = temp_resp
                    status_code_val = temp_resp.status_code
                    # headers_val = temp_resp.headers
                except Exception:
                     log_message_response_parts.append(f"Raw Return Value (not Flask Response/Tuple): {str(response_val)[:1000]}")

            if status_code_val is not None:
                log_message_response_parts.append(f"Status: {status_code_val}")
            
            if actual_response_obj and hasattr(actual_response_obj, 'mimetype'):
                mimetype = actual_response_obj.mimetype
                is_json_response = 'application/json' in mimetype
                is_text_response = 'text/' in mimetype or is_json_response

                if hasattr(actual_response_obj, 'get_data'):
                    try:
                        response_data_bytes = actual_response_obj.get_data() 
                        if len(response_data_bytes) > 2 * 1024 * 1024: 
                             log_message_response_parts.append(f"Response Data: (Too large to log: {len(response_data_bytes)} bytes, Mimetype: {mimetype})")
                        elif is_text_response:
                            response_data_str = response_data_bytes.decode('utf-8', errors='replace')
                            log_message_response_parts.append(f"Response Data ({mimetype}): {response_data_str[:1000]}{'...' if len(response_data_str) > 1000 else ''}")
                        else: 
                            log_message_response_parts.append(f"Response Data: (Binary or non-text, Mimetype: {mimetype}, Size: {len(response_data_bytes)} bytes)")
                    except Exception as e: 
                        log_message_response_parts.append(f"Note: Could not log response data (Error: {str(e)}). Mimetype: {mimetype if mimetype else 'N/A'}")
                elif hasattr(actual_response_obj, 'response') and isinstance(actual_response_obj.response, (list, werkzeug.wsgi.ClosingIterator)):
                    log_message_response_parts.append(f"Note: Response is a stream (e.g., from send_file). Data not logged. Mimetype: {mimetype if mimetype else 'N/A'}")
                else:
                     log_message_response_parts.append(f"Note: Response object structure not recognized for data logging. Mimetype: {mimetype if mimetype else 'N/A'}")
            elif not actual_response_obj and not any(msg_part.startswith("Raw") for msg_part in log_message_response_parts):
                log_message_response_parts.append("Note: Could not determine response object details for logging.")
        else: 
            log_message_response_parts.append("Note: Route returned None.")

        logger.info("\n".join(log_message_response_parts))
        return response_val
    return decorated_function


@app.route('/api/status', methods=['GET', 'OPTIONS'])
@log_route_details
def api_status():
    if request.method == 'OPTIONS':
        # Pre-flight request. Reply successfully:
        return jsonify({'status': 'ok'})

    # Include path information for debugging
    try:
        processed_files = os.listdir(PROCESSED_FOLDER) if os.path.exists(PROCESSED_FOLDER) else []
        processed_files_count = len(processed_files)
    except:
        processed_files_count = 0

    return jsonify({
        'success': True,
        'status': 'online',
        'message': 'API is online and ready.',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'debug_info': {
            'processed_folder': PROCESSED_FOLDER,
            'processed_folder_exists': os.path.exists(PROCESSED_FOLDER),
            'processed_files_count': processed_files_count,
            'working_directory': os.getcwd(),
            'use_r2_storage': USE_R2_STORAGE
        }
    })

# --- Storage Configuration ---
USE_R2_STORAGE = os.environ.get('USE_R2_STORAGE', 'false').lower() == 'true'
R2_ENDPOINT = os.environ.get('R2_ENDPOINT')
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_VIDEO_BUCKET = os.environ.get('R2_VIDEO_BUCKET')
R2_PUBLIC_URL = os.environ.get('R2_PUBLIC_URL')

r2_client = None
if USE_R2_STORAGE:
    print("R2 Storage is ENABLED.")
    if not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_VIDEO_BUCKET, R2_PUBLIC_URL]):
        print("FATAL: R2 storage is enabled, but one or more R2 environment variables are missing.")
        sys.exit(1)
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4')
        )
        # Verify connection by listing buckets
        r2_client.list_buckets()
        print("Successfully connected to R2.")
    except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
        print(f"FATAL: Failed to connect to R2: {e}")
        sys.exit(1)
    # In R2 mode, these folders are just temporary local workspaces
    UPLOAD_FOLDER = 'temp_uploads'
    PROCESSED_FOLDER = 'temp_processed'
else:
    print("R2 Storage is DISABLED. Using local file system.")
    UPLOAD_FOLDER = 'local_uploads'
    PROCESSED_FOLDER = 'local_processed'

print(f"Using paths: UPLOAD_FOLDER={UPLOAD_FOLDER}, PROCESSED_FOLDER={PROCESSED_FOLDER}")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
# Maximum video duration limits based on plan (in seconds)
MAX_VIDEO_DURATION = {
    'free': 40,     # 40 seconds for free plan
    'basic': 120,   # 2 minutes for basic plan
    'pro': 300      # 5 minutes for pro plan
}
DEFAULT_MAX_DURATION = 300  # Default maximum duration (5 minutes)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('static/previews', exist_ok=True)

# Ensure absolute paths for the folders
UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)
PROCESSED_FOLDER = os.path.abspath(PROCESSED_FOLDER)

# Debug: Print the resolved paths
print(f"Resolved absolute paths:")
print(f"  UPLOAD_FOLDER: {UPLOAD_FOLDER}")
print(f"  PROCESSED_FOLDER: {PROCESSED_FOLDER}")
print(f"  Current working directory: {os.getcwd()}")

# Check if we're in a production environment (like /app/)
# and adjust paths accordingly if needed
# Production detection: if /app directory exists and we're not already using absolute /app paths
if os.path.exists('/app') and not PROCESSED_FOLDER.startswith('/app/'):
    production_processed = '/app/local_processed'
    production_uploads = '/app/local_uploads'
    
    # Always use production paths when /app directory exists (production environment)
    print("Production environment detected, switching to /app paths")
    print(f"  Switching from {PROCESSED_FOLDER} to {production_processed}")
    print(f"  Switching from {UPLOAD_FOLDER} to {production_uploads}")
    
    PROCESSED_FOLDER = production_processed
    UPLOAD_FOLDER = production_uploads
    
    # Ensure production directories exist
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"Final paths:")
print(f"  UPLOAD_FOLDER: {UPLOAD_FOLDER}")
print(f"  PROCESSED_FOLDER: {PROCESSED_FOLDER}")

def verify_production_paths():
    """Verify that we're using the correct paths in production"""
    print(f"=== PATH VERIFICATION ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PROCESSED_FOLDER variable: {PROCESSED_FOLDER}")
    print(f"app.config['PROCESSED_FOLDER']: {app.config.get('PROCESSED_FOLDER', 'NOT SET')}")
    print(f"/app directory exists: {os.path.exists('/app')}")
    print(f"/app/local_processed exists: {os.path.exists('/app/local_processed')}")
    print(f"PROCESSED_FOLDER path exists: {os.path.exists(PROCESSED_FOLDER)}")
    
    # List files in the processed directory
    try:
        if os.path.exists(PROCESSED_FOLDER):
            files = os.listdir(PROCESSED_FOLDER)
            print(f"Files in PROCESSED_FOLDER ({len(files)} total): {files[:10]}{'...' if len(files) > 10 else ''}")
        else:
            print(f"PROCESSED_FOLDER does not exist: {PROCESSED_FOLDER}")
    except Exception as e:
        print(f"Error listing PROCESSED_FOLDER: {e}")
    
    # Also check /app/local_processed if it's different
    if PROCESSED_FOLDER != '/app/local_processed' and os.path.exists('/app/local_processed'):
        try:
            files = os.listdir('/app/local_processed')
            print(f"Files in /app/local_processed ({len(files)} total): {files[:10]}{'...' if len(files) > 10 else ''}")
        except Exception as e:
            print(f"Error listing /app/local_processed: {e}")
    print(f"=== END PATH VERIFICATION ===")

def get_next_output_number():
    """
    Get the next sequential output number for Quickcap output files
    Returns a string like '01', '02', etc.
    """
    try:
        # List all files in the processed folder
        files = os.listdir(PROCESSED_FOLDER)
        
        # Filter files that match the pattern "Quickcap output XX"
        output_files = [f for f in files if re.match(r'Quickcap output \d+\.', f)]
        
        if not output_files:
            # No existing files, start at 01
            return "01"
        
        # Extract the numbers from the filenames
        numbers = []
        for filename in output_files:
            match = re.search(r'Quickcap output (\d+)\.', filename)
            if match:
                numbers.append(int(match.group(1)))
        
        # Get the highest number and add 1
        if numbers:
            next_num = max(numbers) + 1
        else:
            next_num = 1
            
        # Format as a 2-digit string (01, 02, etc.)
        return f"{next_num:02d}"
    except Exception as e:
        print(f"Error getting next output number: {str(e)}")
        # Fallback to a random number if there's an error
        return f"{random.randint(1, 99):02d}"

# Set Flask app configuration AFTER production environment detection to ensure correct paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Debug: Verify Flask config has the correct paths
print(f"Flask app config verification:")
print(f"  app.config['UPLOAD_FOLDER']: {app.config['UPLOAD_FOLDER']}")
print(f"  app.config['PROCESSED_FOLDER']: {app.config['PROCESSED_FOLDER']}")

# We'll load the model on-demand or use a preloaded one
# The model will be initialized when needed.
model = None

def test_ffmpeg_availability():
    """Test if FFMPEG is available and working"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFMPEG is available")
            
            # Check for NVENC support
            nvenc_result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
            if 'h264_nvenc' in nvenc_result.stdout:
                print("✓ NVENC (h264_nvenc) encoder is available")
            else:
                print("⚠ NVENC encoder not available (software encoding will be used)")
            
            # Check for NVDEC support  
            hwaccel_result = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], capture_output=True, text=True)
            if 'cuda' in hwaccel_result.stdout:
                print("✓ CUDA hardware acceleration is available")
            else:
                print("⚠ CUDA hardware acceleration not available")
            
            return True
        else:
            print("✗ FFMPEG is not available")
            return False
    except FileNotFoundError:
        print("✗ FFMPEG is not installed or not in PATH")
        return False
    except Exception as e:
        print(f"✗ Error checking FFMPEG: {str(e)}")
        return False

# Test FFMPEG availability at startup
print("\n=== FFMPEG Availability Check ===")
test_ffmpeg_availability()
print("=== End FFMPEG Check ===\n")

def allowed_file(filename):
    if '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    
    # Explicitly reject WebM files
    if extension == 'webm':
        return False
        
    return extension in ALLOWED_EXTENSIONS

def check_video_duration(video_path, user_plan='free'):
    """Check if video duration is within the allowed limit for the user's plan
    
    Args:
        video_path: Path to the video file
        user_plan: User's subscription plan ('free', 'basic', or 'pro')
        
    Returns:
        True if video duration is within the limit, False otherwise
    """
    print(f"Checking video duration for: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    print(f"User plan: {user_plan}")
    
    # Get the maximum duration for the user's plan
    max_duration = MAX_VIDEO_DURATION.get(user_plan.lower(), DEFAULT_MAX_DURATION)
    
    try:
        clip = VideoFileClip(video_path)
        clip.fps = clip.fps or 24
        duration = clip.duration
        print(f"Video duration: {duration} seconds (max allowed for {user_plan} plan: {max_duration} seconds)")
        clip.close()
        return duration <= max_duration
    except Exception as e:
        print(f"Error checking video duration: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_even_dimensions(width, height):
    """Ensure dimensions are even numbers (required for H.264 with yuv420p)"""
    width = int(width)
    height = int(height)
    
    # Make sure width and height are even
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
        
    return width, height

def convert_webm_to_mp4(input_path, output_path):
    """Convert WebM video to MP4 format - DEPRECATED
    
    This function is kept for backward compatibility but should not be used.
    WebM format is no longer supported.
    
    Args:
        input_path: Path to the input WebM file
        output_path: Path to the output MP4 file
        
    Returns:
        Path to the converted MP4 file
    
    Raises:
        ValueError: WebM format is no longer supported
    """
    raise ValueError("WebM format is no longer supported")

def convert_to_9_16_ratio(input_path, output_path):
    """Convert video to 9:16 aspect ratio using cropping if needed"""
    clip = VideoFileClip(input_path)
    clip.fps = clip.fps or 24
    
    # Calculate current aspect ratio
    width, height = clip.size
    current_ratio = width / height
    target_ratio = 9 / 16
    
    # If already 9:16 or taller, no need to crop
    if current_ratio <= target_ratio:
        # Just copy the file
        shutil.copyfile(input_path, output_path)
        return output_path
    
    # Need to crop to 9:16
    new_width = int(height * target_ratio)
    
    # Ensure dimensions are even (required for H.264 with yuv420p)
    new_width, height = ensure_even_dimensions(new_width, height)
    
    # Center crop
    x_center = width / 2
    x1 = int(x_center - new_width / 2)
    
    # Crop the video
    cropped_clip = clip.crop(x1=x1, y1=0, width=new_width, height=height)
    cropped_clip = cropped_clip.set_fps(clip.fps)
    cropped_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        audio_bitrate="128k",
        preset="veryfast",
        temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
        remove_temp=True,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-movflags", "+faststart"
        ]
    )
    
    clip.close()
    cropped_clip.close()
    
    return output_path

def convert_to_9_16_ratio_ffmpeg(input_path, output_path):
    """Convert video to 9:16 aspect ratio using FFMPEG with hardware acceleration"""
    print(f"🔧 FFMPEG FUNCTION CALLED: {input_path} -> {output_path}")
    try:
        # First, get video info to determine current aspect ratio
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'v:0', input_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to probe video: {result.stderr}")
        
        video_info = json.loads(result.stdout)
        video_stream = video_info['streams'][0]
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        current_ratio = width / height
        target_ratio = 9 / 16
        
        print(f"Input video dimensions: {width}x{height}, current ratio: {current_ratio:.3f}, target ratio: {target_ratio:.3f}")
        
        # If already 9:16 or taller (portrait), just copy the file
        if current_ratio <= target_ratio:
            print("Video is already 9:16 or taller, copying file...")
            shutil.copyfile(input_path, output_path)
            return output_path
        
        # Check if hardware acceleration is enabled
        use_hwaccel = is_hardware_acceleration_enabled()
        use_nvdec = is_nvdec_enabled()
        
        print(f"Converting {width}x{height} video to 9:16 aspect ratio (1080x1920)")
        
        # Build FFMPEG command with hardware acceleration if enabled
        if use_hwaccel and use_nvdec:
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-hwaccel', 'cuda',  # Use GPU (NVDEC) to decode the video
                '-i', input_path,
                '-vf', 'crop=(in_h*9/16):in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920',  # Crop to center 9:16, then scale to 1080x1920
                '-c:v', 'h264_nvenc',  # Use GPU (NVENC) to encode the video
                '-b:v', '5M',  # Set bitrate to 5 Mbps
                '-preset', 'fast',  # Set speed preset (preserves quality)
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for web streaming
                output_path
            ]
            print("Using hardware acceleration: NVDEC decode + NVENC encode")
        else:
            # Fallback to software encoding
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', 'crop=(in_h*9/16):in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920',  # Same filter chain but software
                '-c:v', 'libx264',  # Software H.264 encoding
                '-preset', 'fast',  # Fast encoding preset
                '-b:v', '5M',  # Set bitrate to 5 Mbps
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for web streaming
                output_path
            ]
            print("Using software encoding (hardware acceleration disabled)")
        
        print(f"Running FFMPEG command: {' '.join(ffmpeg_cmd)}")
        
        # Run FFMPEG
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFMPEG command failed, trying fallback...")
            print(f"FFMPEG stderr: {result.stderr}")
            
            # Fallback to software encoding without hardware acceleration
            ffmpeg_cmd_fallback = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', 'crop=(in_h*9/16):in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920',  # Same crop formula but software
                '-c:v', 'libx264',  # Software H.264 encoding
                '-preset', 'fast',  # Fast encoding preset
                '-b:v', '5M',  # Set bitrate to 5 Mbps
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for web streaming
                output_path
            ]
            
            print(f"Running fallback FFMPEG command: {' '.join(ffmpeg_cmd_fallback)}")
            result = subprocess.run(ffmpeg_cmd_fallback, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFMPEG conversion failed: {result.stderr}")
        
        print("FFMPEG conversion completed successfully")
        return output_path
        
    except Exception as e:
        print(f"Error in FFMPEG conversion: {str(e)}")
        # Fallback to the original MoviePy-based function if FFMPEG fails
        print("Falling back to MoviePy-based conversion...")
        return convert_to_9_16_ratio(input_path, output_path)

def censor_content(audio_path, segments, output_audio_path=None):
    """
    Censor profanity in both text and audio
    
    Args:
        audio_path: Path to the audio file
        segments: List of transcription segments
        output_audio_path: Path to save the censored audio
        
    Returns:
        Dictionary containing:
        - censored_segments: Segments with censored text
        - censored_audio_path: Path to the censored audio file
        - has_profanity: Boolean indicating if profanity was found
    """
    # Initialize variables
    has_profanity = False
    censored_segments = []
    
    # If no output path is specified, create one in the /app/tmp directory
    if output_audio_path is None:
        # Ensure we're using the writable tmp directory
        temp_dir = "/app/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        output_audio_path = os.path.join(temp_dir, f"censored_audio_{uuid.uuid4().hex}.wav")
    
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    
    # Create a list to store beep sound clips for profanity
    beep_clips = []
    
    # Process each segment
    for segment in segments:
        censored_segment = segment.copy()
        
        # Check if the segment has word-level timestamps
        if "words" in segment:
            censored_words = []
            
            # First, check for multi-word profane phrases
            words_to_check = segment["words"].copy()
            i = 0
            while i < len(words_to_check):
                # Try to find a profane phrase starting from this word
                phrase, start_idx, end_idx = contains_profane_phrase(words_to_check[i:i+5])  # Look ahead up to 5 words
                
                if phrase and start_idx >= 0:
                    # Found a profane phrase
                    has_profanity = True
                    print(f"Found profane phrase: {phrase}")
                    
                    # Process all words in the phrase
                    actual_start_idx = i + start_idx
                    actual_end_idx = i + end_idx
                    
                    # Get timing for the whole phrase
                    phrase_start = words_to_check[actual_start_idx]["start"]
                    phrase_end = words_to_check[actual_end_idx]["end"]
                    phrase_duration = phrase_end - phrase_start
                    
                    # Ensure timestamps are within audio duration
                    phrase_start = max(0, min(phrase_start, audio_duration))
                    phrase_end = max(0, min(phrase_end, audio_duration))
                    
                    # Create a beep for the whole phrase
                    if phrase_end > phrase_start:
                        beep_sound = get_beep_for_duration(phrase_duration, volume_factor=3.0)
                        beep_sound = beep_sound.set_start(phrase_start).set_end(phrase_end)
                        beep_clips.append(beep_sound)
                    
                    # Mark each word in the phrase
                    for j in range(actual_start_idx, actual_end_idx + 1):
                        word_info = words_to_check[j].copy()
                        word = word_info["word"]
                        
                        # Store original word
                        word_info["original_word"] = word
                        # Censor the word
                        word_info["word"] = censor_profanity(word)
                        # Mark as profane
                        word_info["is_profane"] = True
                        # Mark it as part of a phrase
                        word_info["phrase"] = phrase
                        
                        censored_words.append(word_info)
                    
                    # Skip past the end of this phrase
                    i = actual_end_idx + 1
                    
                else:
                    # No phrase found, process single word
                    word_info = words_to_check[i].copy()
                    word = word_info["word"]
                    
                    # Check if the word contains profanity
                    if contains_profanity(word):
                        # This word contains profanity
                        has_profanity = True
                        
                        # Find the exact profane word
                        profane_word = find_profane_word(word)
                        
                        # Store the original word before censoring
                        word_info["original_word"] = word
                        
                        # Censor the word in the text
                        word_info["word"] = censor_profanity(word)
                        
                        # Mark the word as profane
                        word_info["is_profane"] = True
                        
                        # Get word timing information
                        word_start = word_info["start"]
                        word_end = word_info["end"]
                        word_duration = word_end - word_start
                        
                        # Ensure the timestamps are within the audio duration
                        word_start = max(0, min(word_start, audio_duration))
                        word_end = max(0, min(word_end, audio_duration))
                        
                        # Only process if there's a valid duration
                        if word_end > word_start:
                            if profane_word and len(word) > 0:
                                # Get the position of the profane word in the current word
                                profane_pos = word.lower().find(profane_word.lower())
                                
                                if profane_pos >= 0:
                                    # Calculate the proportion of the word that is profane
                                    profane_proportion = len(profane_word) / len(word)
                                    # Adjust beep duration to match only the profane part
                                    beep_duration = word_duration * profane_proportion
                                    # Calculate beep start time based on position in word
                                    beep_start = word_start + (word_duration * (profane_pos / len(word)))
                                    beep_end = beep_start + beep_duration
                                    
                                    # Make beep sound louder for better audibility
                                    beep_sound = get_beep_for_duration(beep_duration, volume_factor=3.0)
                                    beep_sound = beep_sound.set_start(beep_start).set_end(beep_end)
                                    beep_clips.append(beep_sound)
                                else:
                                    # Fallback if we can't find the position - beep the whole word
                                    beep_sound = get_beep_for_duration(word_duration, volume_factor=3.0)
                                    beep_sound = beep_sound.set_start(word_start).set_end(word_end)
                                    beep_clips.append(beep_sound)
                            else:
                                # Fallback if we can't find the specific profane word - beep the whole word
                                beep_sound = get_beep_for_duration(word_duration, volume_factor=3.0)
                                beep_sound = beep_sound.set_start(word_start).set_end(word_end)
                                beep_clips.append(beep_sound)
                    else:
                        # Not profane, keep as is
                        word_info["is_profane"] = False
                    
                    censored_words.append(word_info)
                    i += 1
            
            censored_segment["words"] = censored_words
        
        censored_segments.append(censored_segment)
    
    # If profanity was found, censor the audio
    if has_profanity and beep_clips:
        print(f"Adding {len(beep_clips)} beep sounds for profanity")
        
        # Combine original audio with beep sounds
        # The beep sounds will automatically overlay the original audio at the specified times
        final_audio = CompositeAudioClip([audio_clip] + beep_clips)
        
        # Set the fps attribute from the original audio clip to fix the 'CompositeAudioClip has no attribute fps' error
        final_audio.fps = audio_clip.fps
        
        # Write the censored audio to file
        final_audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
        
        # Close all audio clips
        audio_clip.close()
        final_audio.close()
        for clip in beep_clips:
            clip.close()
    else:
        # No profanity found, just copy the original audio
        audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
        audio_clip.close()
    
    return {
        "censored_segments": censored_segments,
        "censored_audio_path": output_audio_path,
        "has_profanity": has_profanity
    }

def transcribe_video(video_path, filter_profanity=False, preloaded_model=None):
    """
    Transcribe video using Whisper and return segments with timestamps
    
    Args:
        video_path: Path to the video file
        filter_profanity: Whether to filter profanity in the transcription
        preloaded_model: Optional pre-loaded Whisper model to use
        
    Returns:
        Dictionary containing:
        - segments: List of transcription segments
        - video_path: Path to the video file (may be updated if profanity filtering is applied)
    """
    global model
    
    # Prefer the preloaded model if provided (from Modal)
    if preloaded_model is not None:
        print("Using preloaded Whisper model from container")
        transcription_model = preloaded_model
        
    # Otherwise use the global model if available
    elif model is not None:
        print("Using existing global Whisper model")
        transcription_model = model
        
    # If no model is available, load a new one
    else:
        print("Loading new Whisper model...")
        try:
            transcription_model = whisper.load_model("small")
            # Save to global for future use
            model = transcription_model
            print("Whisper model loaded successfully (using 'small' model for better accuracy)")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    # Transcribe the video with our selected model
    print(f"Transcribing video with {'preloaded' if preloaded_model is not None else 'on-demand'} model")
    result = transcription_model.transcribe(video_path, word_timestamps=True)
    segments = result["segments"]
    
    if filter_profanity:
        try:
            # Use /app/tmp directory for temporary files instead of the video directory
            # This ensures we're writing to a writable directory in containerized environments
            temp_dir = "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_audio_path = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4().hex}.wav")
            
            # Extract audio using MoviePy
            video_clip = VideoFileClip(video_path)
            video_clip.fps = video_clip.fps or 24
            # Store original audio before extracting
            original_audio = video_clip.audio
            original_audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
            
            # Use the censor_content function to process both text and audio
            print("Using precise word-level profanity censoring with beep sounds")
                
            # Use a unique filename with UUID to avoid conflicts
            censored_audio_filename = f"censored_audio_{uuid.uuid4().hex}.wav"
            result = censor_content(
                temp_audio_path, 
                segments, 
                output_audio_path=os.path.join(temp_dir, censored_audio_filename)
            )
            
            # Get the censored segments and audio path
            segments = result["censored_segments"]
            censored_audio_path = result["censored_audio_path"]
            
            if result["has_profanity"]:
                print("\n=== PROFANITY DETECTION RESULTS ===")
                print("Profanity detected and censored in the audio")
                
                # Print details of censored words
                print("\nCensored words:")
                censored_count = 0
                for segment in segments:
                    if "words" in segment:
                        for word_info in segment["words"]:
                            if word_info.get("is_profane", False):
                                censored_count += 1
                                original = word_info.get("original_word", "unknown")
                                censored = word_info["word"]
                                start_time = word_info["start"]
                                end_time = word_info["end"]
                                print(f"  {censored_count}. Original: '{original}' → Censored: '{censored}'")
                                print(f"     Timestamp: {start_time:.2f}s - {end_time:.2f}s")
                
                print(f"\nTotal profane words censored: {censored_count}")
                print("==================================\n")
                
                # Replace the audio in the video (using the preserved video_clip)
                audio_clip = AudioFileClip(censored_audio_path)
                video_clip = video_clip.set_audio(audio_clip)
                video_clip = video_clip.set_fps(video_clip.fps)
                
                # Save the video with censored audio
                censored_video_path = os.path.join(temp_dir, "censored_" + os.path.basename(video_path))
                video_clip.write_videofile(
                    censored_video_path,
                    codec="libx264",
                    audio_codec="aac",
                    audio_bitrate="256k",  # Even higher bitrate for better audio quality and clearer beep sounds
                    preset="veryfast",
                    temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                    remove_temp=True,
                    ffmpeg_params=[
                        "-pix_fmt", "yuv420p",
                        "-profile:v", "high",
                        "-level", "4.0"
                    ],
                    logger=None
                )
                
                print("Created censored video with precise beep sounds at exact profanity word positions")
                
                # Close all clips before file operations
                video_clip.close()
                audio_clip.close()
                
                # Instead of replacing the original file, we'll use the censored version directly
                if os.path.exists(censored_video_path):
                    print(f"Censored video created successfully at: {censored_video_path}")
                    # Update the video_path to point to the censored version
                    video_path = censored_video_path
                else:
                    print("Warning: Censored video file was not created successfully")
            else:
                # No profanity found, just close the video clip since we don't need to modify it
                print("No profanity detected in audio")
                video_clip.close()
            
            # Clean up temporary audio files
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                if os.path.exists(censored_audio_path) and censored_audio_path != temp_audio_path:
                    os.remove(censored_audio_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary audio files: {str(e)}")
                
        except Exception as e:
            print(f"Error censoring profanity: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return {
        "segments": segments,
        "video_path": video_path
    }

def split_into_phrases(segments, max_words=3):
    """Split transcription into phrases with max_words per phrase and include word-level timestamps"""
    phrases = []
    
    # Handle the new format where segments might be in a dictionary
    if isinstance(segments, dict):
        if "segments" in segments:
            segments = segments["segments"]
    
    for segment in segments:
        words = segment["words"]
        current_phrase = []
        current_start = None
        current_word_timestamps = []
        
        for word in words:
            if current_start is None:
                current_start = word["start"]
            
            # Store the word and its timestamps
            current_phrase.append(word["word"])
            
            # Check if this word has the is_profane flag
            is_profane = word.get("is_profane", False)
            
            # Store original word if available (for profane words)
            original_word = word.get("original_word", word["word"])
            
            current_word_timestamps.append({
                "word": word["word"],
                "original_word": original_word,
                "start": word["start"],
                "end": word["end"],
                "is_profane": is_profane
            })
            
            if len(current_phrase) >= max_words:
                phrases.append({
                    "text": " ".join(current_phrase),
                    "start": current_start,
                    "end": word["end"],
                    "word_timestamps": current_word_timestamps
                })
                current_phrase = []
                current_word_timestamps = []
                current_start = None
        
        # Add any remaining words
        if current_phrase:
            phrases.append({
                "text": " ".join(current_phrase),
                "start": current_start,
                "end": words[-1]["end"] if words else segment["end"],
                "word_timestamps": current_word_timestamps
            })
    
    return phrases

# No need for NLTK or predefined keywords anymore since we're using random selection

def format_time_srt(seconds):
    """Format time in SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def format_time_ass(seconds):
    """Format time in ASS format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}"

def generate_srt_from_segments(segments):
    """Generate SRT format captions from transcription segments"""
    srt_content = ""
    index = 1
    
    for segment in segments:
        start_time = format_time_srt(segment["start"])
        end_time = format_time_srt(segment["end"])
        text = segment["text"].strip()
        
        srt_content += f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
        index += 1
    
    return srt_content

def generate_ass_from_segments(segments, video_width=480, video_height=854):
    """Generate ASS format captions from transcription segments"""
    # ASS header
    current_time = datetime.now().strftime("%H:%M:%S")
    header = f"""[Script Info]
; Script generated by CaptionsApp
; Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Title: Auto-generated captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Events section
    events = ""
    for segment in segments:
        start_time = format_time_ass(segment["start"])
        end_time = format_time_ass(segment["end"])
        text = segment["text"].strip()
        
        # Escape any special characters in the text
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        
        events += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
    
    return header + events

# OPTIMIZATION 14: Cache for font objects to avoid reloading
_font_cache = {}

# Initialize FFmpeg caption renderer
ffmpeg_renderer = FFmpegCaptionRenderer()

def create_caption_clip(text, video_size, duration, font_size=40, template="minimal_white", current_word=None, current_word_color=None, is_profane=False, custom_text_color=None, custom_highlight_color=None, custom_font=None, caption_vertical_position=50):
    """Create a TextClip for a caption using PIL for font rendering with highlighting
    
    Args:
        text: The text to display
        video_size: The size of the video (width, height)
        duration: How long to display the caption
        font_size: Size of the font
        template: Which template to use (see below for all available templates)
        current_word: The word that is currently being spoken (for word-level highlighting)
        current_word_color: The color to use for highlighting the current word
        caption_vertical_position: Vertical position of captions (0-100, default: 50)
        
    Available Templates by Category:
    
    Classic/Minimal:
        - "minimal_white": Minimal White (Highlights Nouns)
        - "elegant_pink": Elegant Pink (Anton, White with Subtle Highlights)
        
    Entertainment/Vlogger:
        - "mrbeast": MrBeast Style (Highlights Keywords)
        - "vlogger_bold": Vlogger Bold (Helvetica Rounded, Blue Highlights with Black Outline)
        - "creator_clean": Creator Clean (Montserrat, White with Blue Highlights)
        - "reaction_pop": Reaction Pop (League Spartan, White with Red Accents)
        
    Social Media:
        - "tiktok": TikTok (Pink Highlight Bars)
        - "insta_story": Insta Story (Poetsen One, Light Pink Gradient)
        
    Educational/Informative:

        - "explainer_pro": Explainer Pro (Helvetica Rounded, Orange Highlight Bars)
        - "science_journal": Science Journal (Geist-Black with Highlighted Terms)
        
    Gaming:
        - "gaming_neon": Gaming Neon (Exo2, Lowercase Multi-color Neon with Glow Effect)
        
    Cinematic/Film:
        - "film_noir": Film Noir (Anton, Black with White Outline)
        - "cinematic_quote": Cinematic Quote (Proxima Nova Alt Condensed Black Italic, Large Elegant with Fade)
        - "cinematic_futura": Cinematic Futura (Futura PT Bold Oblique, Orange Text)
        
    Comedy/Memes:
        - "meme_orange": Meme Orange (Luckiest Guy, Orange with Black Outline)
        
    Trendy/Viral:
        - "green_bold": Green Bold (Green Keywords with Shadow)
        - "trendy_gradient": Trendy Gradient (Luckiest Guy, Pink to Blue Gradient)
    """
    # Define highlight colors for each template
    highlight_colors = {
        # Word by Word
        "yellow_impact": (255, 255, 0, 255),  # Bright yellow (#FFFF00)
        "bold_white": (255, 255, 255, 255),   # Pure white (#FFFFFF)
        "bold_green": (156, 255, 46, 255),    # Bright green (#9CFF2E)
        
        # Classic/Minimal
        "minimalist_sans": (240, 240, 240, 255),  # Light gray (#F0F0F0)
        "minimal_white": (255, 255, 255, 255),  # Pure white (no highlighting)
        "elegant_pink": (247, 55, 79, 255),  # Bright pink (#F7374F)
        
        # Entertainment/Vlogger
        "bold_sunshine": (255, 255, 0, 255),  # Yellow
        "creator_highlight": (255, 204, 0, 255),  # Yellow (#FFCC00)
        "vlog_caption": (255, 255, 255, 255),  # White
        # MrBeast will use a random color from yellow, red, and green
        # This is just the default color in the dictionary, the actual color will be chosen randomly
        "mrbeast": (255, 221, 0, 255),  # MrBeast Yellow (#ffdd00)
        "vlogger_bold": (82, 95, 225, 255),  # Blue (#525FE1)
        "bold_switch": (255, 255, 255, 255),  # White text (highlights handled separately)
        "creator_clean": (0, 191, 255, 255),  # Deep sky blue
        "reaction_pop": (255, 60, 60, 255),  # Red (#FF3C3C)
        
        # Social Media
        "tiktok_trend": (255, 255, 255, 255),  # White (#FFFFFF)
        "reels_ready": (255, 255, 255, 255),  # White (#FFFFFF)
        "tiktok": (255, 105, 180, 230),  # Hot pink with transparency (for highlight bar)
        "insta_story": (249, 206, 238, 255),  # Light pink (#F9CEEE)
        "blue_highlight": (61, 144, 215, 230),  # Blue (#3D90D7) with transparency (for highlight bar)
        
        # Educational/Informative
        "explainer_pro": (255, 140, 0, 230),  # Dark orange with transparency
        "science_journal": (137, 172, 70, 255),  # Green (#89AC46)
        
        # Gaming
        "gaming_neon": (57, 123, 255, 255),  # Neon blue (#007BFF)

        
        # Cinematic/Film
        "film_noir": (0, 0, 0, 255),  # Black (#000000)
        "cinematic_quote": (255, 255, 0, 255),  # Bright yellow (#FFFF00)
        "cinematic_futura": (247, 161, 101, 255),  # Orange (#F7A165)
        
        # Comedy/Memes
        "meme_orange": (255, 140, 0, 255),  # Orange

        
        # Motivational/Quotes

        
        # Trendy/Viral
        "green_bold": (0, 255, 0, 255),  # Bright Green
        "trendy_gradient": (255, 105, 180, 255),  # Hot pink (#FF69B4)
        "premium_orange": (235, 91, 0, 255),  # Orange (#EB5B00)
        "premium_yellow": (233, 208, 35, 255),  # Yellow (#E9D023)
        "neon_heartbeat": (255, 0, 255, 255),  # Neon Pink (#FF00FF)

        "meme_maker": (255, 255, 255, 255),  # White (#FFFFFF)
        "viral_pop": (255, 51, 153, 255),  # Bright Pink (#FF3399)
        "streamer_pro": (0, 255, 128, 255),  # Neon Green (#00FF80)
        "esports_caption": (255, 69, 0, 255)  # Red-Orange (#FF4500)
    }
    # Select font based on template category
    
    # Check if custom font is provided
    if custom_font:
        # Try to use the custom font with relative paths
        font_path = os.path.join('Fonts', f'{custom_font}.ttf')
        # If the font doesn't exist, try other extensions
        if not os.path.exists(font_path):
            font_path = os.path.join('Fonts', f'{custom_font}.otf')
        # If still doesn't exist, use default
        if not os.path.exists(font_path):
            font_path = os.path.join('Fonts', 'SpiegelSans.otf')
    # Word by Word
    elif template.lower() == "yellow_impact":
        font_path = os.path.join('Fonts', 'EuropaGroteskSH-Bol.otf')
    elif template.lower() == "bold_white":
        font_path = os.path.join('Fonts', 'Poppins-BlackItalic.ttf')
    elif template.lower() == "bold_green":
        font_path = os.path.join('Fonts', 'Poppins-ExtraBold.ttf')
    
    # Classic/Minimal
    elif template.lower() == "minimalist_sans":
        font_path = os.path.join('Fonts', 'HelveticaNeue-Light.ttf')
    elif template.lower() == "minimal_white":
        font_path = os.path.join('Fonts', 'SpiegelSans.otf')
    elif template.lower() == "elegant_pink":
        font_path = os.path.join('Fonts', 'Anton-Regular.ttf')
    
    # Entertainment/Vlogger
    elif template.lower() == "bold_sunshine":
        font_path = os.path.join('Fonts', 'Theboldfont.ttf')
    elif template.lower() == "creator_highlight":
        font_path = os.path.join('Fonts', 'Poppins-SemiBold.ttf')
    elif template.lower() == "vlog_caption":
        font_path = os.path.join('Fonts', 'Montserrat-Bold.ttf')
    elif template.lower() == "mrbeast":
        font_path = os.path.join('Fonts', 'Komikax.ttf')
    elif template.lower() == "vlogger_bold":
        font_path = os.path.join('Fonts', 'HelveticaRoundedLTStd-Bd.ttf')
    elif template.lower() == "bold_switch":
        font_path = os.path.join('Fonts', 'Theboldfont.ttf')
    elif template.lower() == "creator_clean":
        font_path = os.path.join('Fonts', 'Montserrat.ttf')
    elif template.lower() == "reaction_pop":
        font_path = os.path.join('Fonts', 'Proxima Nova Alt Condensed Black.otf')
    
    # Social Media
    elif template.lower() == "tiktok_trend":
        font_path = os.path.join('Fonts', 'Inter-Black.ttf')
    elif template.lower() == "reels_ready":
        font_path = os.path.join('Fonts', 'SFProDisplay-Bold.ttf')
    elif template.lower() == "tiktok":
        font_path = os.path.join('Fonts', 'Proxima Nova Alt Condensed Black Italic.otf')
    elif template.lower() == "insta_story":
        font_path = os.path.join('Fonts', 'PoetsenOne-Regular.ttf')
    elif template.lower() == "blue_highlight":
        font_path = os.path.join('Fonts', 'Poppins-ExtraBold.ttf')
    
    # Educational/Informative
    elif template.lower() == "educational":
        font_path = os.path.join('Fonts', 'ARIALBD.TTF')
    elif template.lower() == "tutorial_tech":
        font_path = os.path.join('Fonts', 'ARIALBD.TTF')
    elif template.lower() == "explainer_pro":
        font_path = os.path.join('Fonts', 'HelveticaRoundedLTStd-Bd.ttf')
    elif template.lower() == "science_journal":
        font_path = os.path.join('Fonts', 'Geist-Black.otf')
    
    # Gaming
    elif template.lower() == "gaming_neon":
        font_path = os.path.join('Fonts', 'Exo2-VariableFont_wght.ttf')
    
    # Cinematic/Film
    elif template.lower() == "film_noir":
        font_path = os.path.join('Fonts', 'Anton-Regular.ttf')
    elif template.lower() == "cinematic_quote":
        font_path = os.path.join('Fonts', 'Proxima Nova Alt Condensed Black Italic.otf')
    elif template.lower() == "cinematic_futura":
        font_path = os.path.join('Fonts', 'futura-pt-bold-oblique.otf')
    
    # Comedy/Memes
    elif template.lower() == "meme_orange":
        font_path = os.path.join('Fonts', 'LuckiestGuy.ttf')
    
    # Trendy/Viral
    elif template.lower() == "green_bold":
        font_path = os.path.join('Fonts', 'Uni Sans Heavy.otf')
    elif template.lower() == "trendy_gradient":
        font_path = os.path.join('Fonts', 'LuckiestGuy.ttf')
    elif template.lower() == "premium_orange":
        font_path = os.path.join('Fonts', 'Poppins-BoldItalic.ttf')
    elif template.lower() == "premium_yellow":
        font_path = os.path.join('Fonts', 'Poppins-BoldItalic.ttf')  # Same font as Premium Orange
    elif template.lower() == "neon_heartbeat":
        font_path = os.path.join('Fonts', 'Montserrat-SemiBoldItalic.ttf')
    elif template.lower() == "neon_pulse":
        font_path = os.path.join('Fonts', 'BebasNeue-Regular.ttf')

    elif template.lower() == "creator_highlight":
        font_path = os.path.join('Fonts', 'Poppins-SemiBold.ttf')
    elif template.lower() == "meme_maker":
        font_path = os.path.join('Fonts', 'Impact.ttf')
    elif template.lower() == "viral_pop":
        font_path = os.path.join('Fonts', 'Provicali.otf')
    elif template.lower() == "streamer_pro":
        font_path = os.path.join('Fonts', 'Rajdhani-Bold.ttf')
    elif template.lower() == "esports_caption":
        font_path = os.path.join('Fonts', 'Exo2-Black.ttf')
    
    # Default fallback
    else:
        font_path = os.path.join('Fonts', 'SpiegelSans.otf')
    
    # Handle text case based on template style
    
    # Templates that use lowercase
    if template.lower() in ["tiktok", "gaming_neon"]:
        text = text.lower()
    
    # Templates that use uppercase
    elif template.lower() in ["meme_orange", "viralpop_bold", "premium_orange", "premium_yellow", 
                             "neon_pulse", "viral_pop", "esports_caption", "bold_green",
                             "cinematic_futura", "bold_switch", "film_noir", "blue_highlight",
                             "meme_maker"]:
        text = text.upper()
    
    # Templates that use title case (capitalize each word)
    elif template.lower() in ["cinematic_quote", "reaction_pop"]:
        text = ' '.join(word.capitalize() for word in text.split())
    
    # All other templates use normal case (sentence case)
    else:
        pass
    
    # Print the font path to the terminal
    print(f"Using font: {font_path} with template: {template}")
    
    # Quality multiplier for high-resolution rendering
    quality_multiplier = 2.0
    
    # Helper function to measure text with quality scaling
    def measure_text_scaled(text_to_measure, font_to_use):
        bbox = draw.textbbox((0, 0), text_to_measure, font=font_to_use)
        width = int(bbox[2] / quality_multiplier)
        height = int(bbox[3] / quality_multiplier)
        return width, height
    
    # Create a temporary image with PIL - use full video dimensions for better quality
    img_width = video_size[0]
    img_height = video_size[1]  # Use full video height instead of 20%
    
    # Create a transparent background with higher quality settings
    text_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    
    # Enable high-quality text rendering
    try:
        # This ensures better text rendering quality
        draw.fontmode = "1"  # Enable anti-aliasing
    except:
        pass
    
    # Load the custom font
    try:
        # Optimize font path resolution to avoid redundant checks
        # If the path starts with '/root/Fonts/', it's likely a hardcoded path that doesn't exist
        if font_path.startswith('/root/Fonts/'):
            # Directly use the local Fonts directory
            font_name = os.path.basename(font_path)
            local_font_path = os.path.join('Fonts', font_name)
            if os.path.exists(local_font_path):
                font_path = local_font_path
            else:
                # Log only once if we can't find the font
                if font_name not in getattr(create_caption_clip, 'missing_fonts', set()):
                    if not hasattr(create_caption_clip, 'missing_fonts'):
                        create_caption_clip.missing_fonts = set()
                    create_caption_clip.missing_fonts.add(font_name)
                    print(f"Font not found: {font_name}, will use fallback")
        elif not os.path.exists(font_path):
            # For other paths, check if they exist
            local_font_path = os.path.join('Fonts', os.path.basename(font_path))
            if os.path.exists(local_font_path):
                font_path = local_font_path
            else:
                # Log only once if we can't find the font
                font_name = os.path.basename(font_path)
                if font_name not in getattr(create_caption_clip, 'missing_fonts', set()):
                    if not hasattr(create_caption_clip, 'missing_fonts'):
                        create_caption_clip.missing_fonts = set()
                    create_caption_clip.missing_fonts.add(font_name)
                    print(f"Font not found: {font_name}, will use fallback")
        
        # Try to load the font with better quality settings
        if os.path.exists(font_path):
            # Load font with higher resolution for better quality
            # Scale up font size for rendering, then scale down the image if needed
            quality_multiplier = 2.0  # Render at 2x size for better quality
            high_res_font_size = int(font_size * quality_multiplier)
            
            # OPTIMIZATION 15: Use font cache to avoid reloading fonts
            font_key = f"{font_path}_{high_res_font_size}"
            if font_key in _font_cache:
                font = _font_cache[font_key]
                # Don't log every time we use a cached font - too verbose
            else:
                font = ImageFont.truetype(font_path, high_res_font_size)
                _font_cache[font_key] = font
                print(f"Loaded and cached new font: {os.path.basename(font_path)} at size: {high_res_font_size}")
                
            # We don't need a separate highlighted font anymore since we'll use animation
            highlighted_font = font
        else:
            # Use a cached fallback font if possible
            if not hasattr(create_caption_clip, 'fallback_font_path'):
                # Find a fallback font only once
                fonts_dir = 'Fonts'  # Always use local Fonts directory
                if os.path.exists(fonts_dir) and os.listdir(fonts_dir):
                    # Prefer common fonts if available
                    preferred_fonts = ['Arial.ttf', 'Poppins-Regular.ttf', 'Montserrat.ttf', 'SpiegelSans.otf']
                    available_fonts = os.listdir(fonts_dir)
                    
                    # Try to find a preferred font first
                    fallback_font = None
                    for pref in preferred_fonts:
                        if pref in available_fonts:
                            fallback_font = pref
                            break
                    
                    # If no preferred font found, use the first available
                    if not fallback_font and available_fonts:
                        fallback_font = available_fonts[0]
                    
                    if fallback_font:
                        create_caption_clip.fallback_font_path = os.path.join(fonts_dir, fallback_font)
                        print(f"Using fallback font: {os.path.basename(create_caption_clip.fallback_font_path)}")
                    else:
                        create_caption_clip.fallback_font_path = None
                else:
                    create_caption_clip.fallback_font_path = None
            
            # Use the cached fallback font path
            if create_caption_clip.fallback_font_path:
                quality_multiplier = 2.0
                high_res_font_size = int(font_size * quality_multiplier)
                
                # Use font cache for fallback font
                font_key = f"{create_caption_clip.fallback_font_path}_{high_res_font_size}"
                if font_key in _font_cache:
                    font = _font_cache[font_key]
                else:
                    font = ImageFont.truetype(create_caption_clip.fallback_font_path, high_res_font_size)
                    _font_cache[font_key] = font
                    print(f"Loaded fallback font: {os.path.basename(create_caption_clip.fallback_font_path)}")
                
                highlighted_font = font
            else:
                # No fonts available, use default
                if not hasattr(create_caption_clip, 'default_font_logged'):
                    print("No fonts available, using default")
                    create_caption_clip.default_font_logged = True
                
                # Even for default font, try to scale it up
                quality_multiplier = 2.0
                high_res_font_size = int(font_size * quality_multiplier)
                try:
                    font = ImageFont.load_default().font_variant(size=high_res_font_size)
                except:
                    font = ImageFont.load_default()
                highlighted_font = font
    except Exception as e:
        print(f"Error loading font: {e}")
        # Fallback to default font with quality scaling
        quality_multiplier = 2.0
        high_res_font_size = int(font_size * quality_multiplier)
        try:
            font = ImageFont.load_default().font_variant(size=high_res_font_size)
        except:
            font = ImageFont.load_default()
        highlighted_font = font
    
    # Identify words to highlight based on sequential approach
    highlight_indices = {}
    
    # Split text into words
    words = text.split()
    
    # Initialize static variables to track highlighting state
    if not hasattr(create_caption_clip, 'word_position'):
        create_caption_clip.word_position = 0
    
    # Track the current phrase to know when we've moved to a new one
    if not hasattr(create_caption_clip, 'current_phrase'):
        create_caption_clip.current_phrase = None
    
    # Track if we've reached the end of the current phrase
    if not hasattr(create_caption_clip, 'phrase_completed'):
        create_caption_clip.phrase_completed = False
    
    # If we have words in the phrase, highlight one sequentially
    if words:
        # Get indices of words that are not just punctuation or spaces
        all_word_indices = []
        for idx, word in enumerate(words):
            # Only consider actual words with letters, not just punctuation, spaces, or empty strings
            if word.strip('.,!?:;()"\'-') and any(c.isalpha() for c in word):
                all_word_indices.append(idx)
        
        # If we have valid words to highlight
        if all_word_indices:
            # Check if this is a new phrase
            current_phrase_text = ' '.join(words)
            is_new_phrase = (create_caption_clip.current_phrase != current_phrase_text)
            
            # If this is a new phrase, reset the word position
            if is_new_phrase:
                # Always reset position for a new phrase to ensure we don't skip any phrases
                create_caption_clip.word_position = 0
                create_caption_clip.phrase_completed = False
                
                # Update the current phrase
                create_caption_clip.current_phrase = current_phrase_text
            
            # Special case for phrases with only one word - always highlight it
            if len(all_word_indices) == 1:
                highlight_indices[all_word_indices[0]] = True
                create_caption_clip.phrase_completed = True
            # If we've reached the end of possible positions, mark the phrase as completed
            # but don't reset the position yet - we'll wait for the next phrase
            elif create_caption_clip.word_position >= len(all_word_indices):
                create_caption_clip.phrase_completed = True
                # Don't highlight any word in this case
            else:
                # Highlight the current word in the sequence
                selected_index = all_word_indices[create_caption_clip.word_position]
                highlight_indices[selected_index] = True
                
                # Increment the position for the next call
                create_caption_clip.word_position += 1
    
    # Check if text is too wide for the screen and wrap if necessary
    # Determine if this is a vertical video (9:16 or similar aspect ratio)
    is_vertical = video_size[0] / video_size[1] <= 0.65  # Approximately 9:16 or narrower
    
    # Set max width percentages to 80% for all templates to avoid text going off-screen
    # This ensures captions stay within 80% of the vertical screen's width
    first_line_max_width_percentage = 0.8  # 80% of image width
    second_line_max_width_percentage = 0.8  # 80% of image width
    
    # Special handling for different templates
    if template.lower() in ["mrbeast", "meme_orange"]:
        # More restrictive for chunky fonts
        second_line_max_width_percentage = 0.65
    elif template.lower() in ["premium_orange", "premium_yellow"]:
        # Ensure premium styles don't exceed 80% of screen width for both lines
        # Using slightly more restrictive values to guarantee it stays within bounds
        first_line_max_width_percentage = 0.75  # 75% of image width for first line
        second_line_max_width_percentage = 0.75  # 75% of image width for second line
    
    # Calculate actual pixel widths
    first_line_max_width = int(img_width * first_line_max_width_percentage)
    second_line_max_width = int(img_width * second_line_max_width_percentage)
    
    # Measure the full text width using helper function
    text_width, text_height = measure_text_scaled(text, font)
    
    # If text is too wide, we need to wrap it, but limit to max 2 lines
    lines = []
    if text_width > first_line_max_width:
        words = text.split()
        total_words = len(words)
        
        # Special handling for Green Bold template
        if template.lower() == "green_bold":
            # For Green Bold, first line should have maximum 2 words but not exceed 80% width
            max_screen_width_80_percent = int(img_width * 0.8)
            
            if total_words <= 1:
                # If we have only 1 word, just use one line
                lines = [words[0]]
            elif total_words == 2:
                # If we have exactly 2 words, check if they fit within 80% width
                two_words = " ".join(words)
                two_words_width, _ = measure_text_scaled(two_words, font)
                
                if two_words_width > max_screen_width_80_percent:
                    # If 2 words exceed 80% width, split them into two lines
                    lines = [words[0], words[1]]
                else:
                    # Both words fit within 80% width
                    lines = [two_words]
            else:
                # More than 2 words
                # First check if 2 words fit within 80% width for first line
                first_two_words = " ".join(words[:2])
                first_two_words_width, _ = measure_text_scaled(first_two_words, font)
                
                if first_two_words_width > max_screen_width_80_percent:
                    # If 2 words exceed 80% width, use only 1 word for first line
                    first_line = words[0]
                    remaining_words = words[1:]
                else:
                    # 2 words fit within 80% width for first line
                    first_line = first_two_words
                    remaining_words = words[2:]
                
                # Now handle second line (if needed)
                if remaining_words:
                    if len(remaining_words) == 1:
                        # Only one word left, use it for second line
                        second_line = remaining_words[0]
                    else:
                        # Check if 2 words fit within 80% width for second line
                        second_two_words = " ".join(remaining_words[:2])
                        second_two_words_width, _ = measure_text_scaled(second_two_words, font)
                        
                        if second_two_words_width > max_screen_width_80_percent:
                            # If 2 words exceed 80% width, use only 1 word for second line
                            second_line = remaining_words[0]
                            # Any remaining words will be moved to the next phrase
                        else:
                            # 2 words fit within 80% width for second line
                            second_line = second_two_words
                            # Any remaining words will be moved to the next phrase
                    
                    lines = [first_line, second_line]
                else:
                    # No remaining words, just use first line
                    lines = [first_line]
        
        # Special handling for chunky font templates (MrBeast and Meme Orange)
        elif template.lower() in ["mrbeast", "meme_orange"] and total_words >= 2:
            # Check if two words would exceed 80% of screen width
            two_words = " ".join(words[:2])
            two_words_width, _ = measure_text_scaled(two_words, font)
            max_screen_width_80_percent = int(img_width * 0.8)
            
            if two_words_width > max_screen_width_80_percent and total_words >= 2:
                # If two words exceed 80% of width, limit to 1 word per line
                if total_words >= 3:
                    # For 3+ words: first line gets 1 word, second line gets 1 word, rest are dropped
                    lines = [words[0], words[1]]
                else:
                    # For 2 words: 1 word per line
                    lines = [words[0], words[1]]
            else:
                # Two words fit within 80% of width
                if total_words >= 3:
                    # For 3+ words: 2 words on first line, 1 word on second line
                    first_line = " ".join(words[:2])
                    second_line = words[2]
                    lines = [first_line, second_line]
                else:
                    # For 2 words: both on first line
                    lines = [two_words]
        elif template.lower() in ["mrbeast", "meme_orange"] and total_words == 1:
            # For 1 word: just use that word
            lines = [words[0]]
        else:
            # For other templates or when we have fewer words, use the standard wrapping logic
            # First, try to find an optimal split point that keeps both lines within their max widths
            best_split_index = -1
            
            # Try different split points to find one where both lines fit within their constraints
            for i in range(1, total_words):
                # Check if first part fits in first line max width
                first_part = " ".join(words[:i])
                first_part_width, _ = measure_text_scaled(first_part, font)
                
                # Check if second part fits in second line max width
                second_part = " ".join(words[i:])
                second_part_width, _ = measure_text_scaled(second_part, font)
                
                # If both parts fit within their respective max widths, this is a valid split point
                if first_part_width <= first_line_max_width and second_part_width <= second_line_max_width:
                    best_split_index = i
                    # Prefer splits closer to the middle for better balance
                    if i >= total_words // 3 and i <= 2 * total_words // 3:
                        break
            
            # If we found a valid split point
            if best_split_index > 0:
                first_line = " ".join(words[:best_split_index])
                second_line = " ".join(words[best_split_index:])
                lines = [first_line, second_line]
            else:
                # If no good split point was found, use a balanced approach
                # First try to fit as many words as possible on the first line
                current_line = []
                current_line_width = 0
                
                for i, word in enumerate(words):
                    # Calculate width of this word (with space)
                    word_with_space = word + " " if i < total_words - 1 else word
                    word_width, _ = measure_text_scaled(word_with_space, font)
                    
                    # If adding this word would exceed first line max width
                    if current_line_width + word_width > first_line_max_width and current_line:
                        # Check if remaining words fit on second line
                        remaining = " ".join(words[i:])
                        remaining_width, _ = measure_text_scaled(remaining, font)
                        
                        if remaining_width <= second_line_max_width:
                            # We found a good split
                            first_line = " ".join(current_line)
                            second_line = remaining
                            lines = [first_line, second_line]
                            break
                        else:
                            # Need to be more aggressive with splitting
                            # Try to balance the lines better
                            mid_point = total_words // 2
                            first_line = " ".join(words[:mid_point])
                            second_line = " ".join(words[mid_point:])
                            
                            # Check if second line is still too wide
                            second_line_width, _ = measure_text_scaled(second_line, font)
                            if second_line_width > second_line_max_width:
                                # Need to move more words to first line
                                for j in range(mid_point, total_words - 1):
                                    test_first = " ".join(words[:j+1])
                                    test_second = " ".join(words[j+1:])
                                    test_first_width, _ = measure_text_scaled(test_first, font)
                                    test_second_width, _ = measure_text_scaled(test_second, font)
                                    
                                    if test_first_width <= first_line_max_width and test_second_width <= second_line_max_width:
                                        first_line = test_first
                                        second_line = test_second
                                        break
                            
                            lines = [first_line, second_line]
                            break
                    
                    # Add word to current line
                    current_line.append(word)
                    current_line_width += word_width
                
                # If we haven't created lines yet, all words fit on one line
                if not lines and current_line:
                    lines = [" ".join(current_line)]
    else:
        # Text fits on one line
        lines = [text]
    
    # Limit to maximum 2 lines for Green Bold template
    if template.lower() == "green_bold" and len(lines) > 2:
        lines = lines[:2]
        
    # Calculate total height needed for all lines based on template style
    
    # Extra large spacing (1.8x)
    if template.lower() in ["meme_orange", "bold_sunshine"]:
        line_height = text_height * 1.8  # Extra spacing for cartoon/meme styles and bold templates
    
    # Reduced spacing for gaming neon (1.5x)
    elif template.lower() == "gaming_neon":
        line_height = text_height * 1.5  # Reduced spacing for gaming neon style
    
    # Large spacing (1.6x)
    elif template.lower() in ["viralpop_bold", "reaction_pop", "green_bold", "premium_orange", 
                             "premium_yellow", "neon_heartbeat", "viral_pop", 
                             "streamer_pro", "esports_caption", "movie_title", "bold_switch"]:
        line_height = text_height * 1.6  # Good spacing for bold, dynamic styles
    
    # Medium-large spacing (1.5x for cinematic templates)
    elif template.lower() in ["cinematic_quote", "cinematic_futura"]:
        line_height = text_height * 1.5  # More spacing for contrast and impact
    
    # Extra large spacing (2.0x) for trendy gradient
    elif template.lower() == "trendy_gradient":
        line_height = text_height * 2.0  # Much larger spacing for trendy gradient style
    
    # Medium spacing (1.4x)
    elif template.lower() in ["elegant_pink", "creator_highlight"]:
        line_height = text_height * 1.4  # Moderate spacing for balanced styles
    
    # Medium-small spacing (1.3x)
    elif template.lower() in ["film_noir", "science_journal"]:
        line_height = text_height * 1.3  # Slightly more spacing for elegant, readable styles
    
    # Standard spacing (1.2x)
    else:
        line_height = text_height * 1.2  # Standard spacing for other templates
    total_text_height = line_height * len(lines)
    
    # Calculate vertical starting position for captions
    # Use custom vertical position if provided, otherwise use default positioning
    is_vertical = video_size[0] / video_size[1] <= 0.65  # Approximately 9:16 or narrower
    
    # Get the vertical position from function parameter
    # Convert from percentage (0-100) to ratio (0-1)
    custom_position_ratio = caption_vertical_position / 100
    # Ensure the position is within valid range
    caption_position_ratio = max(0.1, min(0.9, custom_position_ratio))
    print(f"Using custom caption vertical position: {caption_vertical_position}% ({caption_position_ratio:.2f} ratio)")
        
    y_position = int(img_height * caption_position_ratio) - (total_text_height // 2)
    
    # Draw each line
    for line_idx, line in enumerate(lines):
        # For highlighting words, we need to process word by word
        words = line.split()
        
        # First, calculate the total width of this line with proper spacing
        total_line_width = 0
        word_widths = []
        
        # Find which words in this line correspond to which tokens in the original text
        line_start_idx = 0
        if line_idx > 0:
            # Calculate how many words came before this line
            for prev_line_idx in range(line_idx):
                line_start_idx += len(lines[prev_line_idx].split())
        
        for word_idx, word in enumerate(words):
            # Add space after each word except the last one
            word_with_space = word + " " if word_idx < len(words) - 1 else word
            
            # Determine if this word should be highlighted
            original_idx = line_start_idx + word_idx
            is_highlighted = highlight_indices.get(original_idx, False)
            
            # Use the same font for all words since we'll use animation instead of scaling
            word_width, _ = measure_text_scaled(word_with_space, font)
                
            word_widths.append(word_width)
            total_line_width += word_width
        
        # Calculate horizontal position to center this line
        x_position = (img_width - total_line_width) // 2
        current_x = x_position
        
        # Draw each word with appropriate color
        for word_idx, word in enumerate(words):
            original_idx = line_start_idx + word_idx
            
            # Determine if this word should be highlighted
            is_highlighted = highlight_indices.get(original_idx, False)
            
            # Check if this is the current word being spoken (for word-level highlighting)
            is_current_word = False
            if current_word is not None and word.lower().strip('.,!?:;()"\'-') == current_word.lower().strip('.,!?:;()"\'-'):
                is_current_word = True
                
            # Check if this word contains asterisks (censored profanity)
            contains_asterisks = '*' in word
            
            # Get the word with space for rendering
            word_with_space = word + " " if word_idx < len(words) - 1 else word
            
            # Get the word dimensions for highlighting
            word_width = word_widths[word_idx]
            word_height = text_height
            
            # For current word highlighting (spoken word)
            if is_current_word and current_word_color is not None:
                # Draw a highlight behind the current word being spoken
                highlight_padding = int(font_size * 0.1)  # Padding around the text
                
                # Use the provided highlight color but with transparency
                r, g, b, a = current_word_color
                highlight_color = (r, g, b, 100)  # Use the same color but with transparency
                
                # Draw the highlight rectangle
                draw.rectangle(
                    [
                        current_x - highlight_padding, 
                        y_position + line_idx * line_height - highlight_padding,
                        current_x + word_width + highlight_padding,
                        y_position + line_idx * line_height + word_height + highlight_padding
                    ],
                    fill=highlight_color
                )
            
            # For TikTok template, draw a highlight bar behind keywords
            elif template.lower() == "tiktok" and is_highlighted:
                # Draw a highlight bar behind the word
                highlight_padding = int(font_size * 0.1)  # Padding around the text
                
                # Get the highlight color for TikTok
                highlight_color = highlight_colors.get("tiktok", (255, 105, 180, 230))  # Pink for TikTok
                
                # Draw the highlight rectangle
                draw.rectangle(
                    [
                        current_x - highlight_padding, 
                        y_position + line_idx * line_height - highlight_padding,
                        current_x + word_width + highlight_padding,
                        y_position + line_idx * line_height + word_height + highlight_padding
                    ],
                    fill=highlight_color
                )
                
                # All text is white in these templates
                word_color = (255, 255, 255, 255)
                
            # For Blue Highlight template, draw a highlight bar behind keywords
            elif template.lower() == "blue_highlight" and is_highlighted:
                # Draw a highlight bar behind the word
                highlight_padding = int(font_size * 0.2)  # Increased padding around the text (20% of font size)
                
                # Get the highlight color for Blue Highlight
                highlight_color = highlight_colors.get("blue_highlight", (61, 144, 215, 230))  # Blue (#3D90D7) for Blue Highlight
                
                # Draw the highlight rectangle
                draw.rectangle(
                    [
                        current_x - highlight_padding, 
                        y_position + line_idx * line_height - highlight_padding,
                        current_x + word_width + highlight_padding,
                        y_position + line_idx * line_height + word_height + highlight_padding
                    ],
                    fill=highlight_color
                )
                
                # All text is white in these templates
                word_color = (255, 255, 255, 255)
                
            # If this word is censored profanity, color it red
            elif contains_asterisks or (is_current_word and is_profane):
                # Use a brighter red for censored words
                word_color = (255, 0, 0, 255)  # Bright red for censored words
                
                # Removed the beep indicator (small red circle) as requested
            # If this is the current word being spoken, use the specified highlight color
            elif is_current_word and current_word_color is not None:
                word_color = current_word_color  # Use the provided highlight color
            elif template.lower() == "educational":
                # For Educational template, use off-white for normal text and alternate between yellow and sky blue for keywords
                if is_highlighted:
                    # Alternate between yellow and blue
                    if random.random() < 0.5:
                        word_color = highlight_colors.get("educational_blue", (135, 206, 235, 255))  # Sky Blue
                    else:
                        word_color = highlight_colors.get("educational_yellow", (255, 255, 0, 255))  # Yellow
                else:
                    word_color = (245, 245, 245, 255)  # Off-white (#F5F5F5)
            elif template.lower() == "green_bold":
                # For Green Bold template, use white for normal text and bright green for keywords
                if is_highlighted:
                    word_color = highlight_colors.get("green_bold", (0, 255, 0, 255))
                else:
                    word_color = (255, 255, 255, 255)  # White
            # Bold Contrast template removed
            elif template.lower() == "meme_orange":
                # For Meme Orange template, all text is orange (with black outline)
                word_color = highlight_colors.get("meme_orange", (255, 140, 0, 255))
            elif template.lower() == "viralpop_bold":
                # For ViralPop Bold template, all text is white (with black outline)
                word_color = highlight_colors.get("viralpop_bold", (255, 255, 255, 255))
            elif template.lower() == "film_noir":
                # For Film Noir template, all text is black (with white outline)
                word_color = highlight_colors.get("film_noir", (0, 0, 0, 255))
            elif template.lower() == "cinematic_futura":
                # For Cinematic Futura template, all text is orange (#F7A165)
                word_color = highlight_colors.get("cinematic_futura", (247, 161, 101, 255))
            elif template.lower() == "premium_yellow":
                # For Premium Yellow template, all text is bright yellow (with black outline)
                word_color = highlight_colors.get("premium_yellow", (233, 208, 35, 255))
            elif template.lower() == "neon_heartbeat":
                # For Neon Heartbeat template, all text is neon pink (with glow effect)
                word_color = highlight_colors.get("neon_heartbeat", (255, 0, 255, 255))
            elif template.lower() == "elegant_shadow":
                # For Elegant Shadow template, all text is white (with drop shadow)
                # Use a soft blue-green/teal color for highlighted keywords
                if is_highlighted:
                    word_color = highlight_colors.get("elegant_shadow", (255, 11, 85, 255))
                else:
                    word_color = (255, 255, 255, 255)  # Pure white for regular text
                    



                
            elif template.lower() == "explainer_pro":
                # For Explainer Pro template, all text is white with highlight bar for highlighted words
                word_color = (255, 255, 255, 255)  # White text for all words
                
                # Only draw highlight bars for non-empty words that are highlighted
                if is_highlighted and word.strip():
                    # Draw a highlight bar behind the word (similar to TikTok style)
                    highlight_padding = int(font_size * 0.1)  # Padding around the text
                    
                    # Get the highlight color for Explainer Pro
                    highlight_color = highlight_colors.get("explainer_pro", (255, 140, 0, 230))  # Orange with transparency
                    
                    # Draw the highlight rectangle
                    draw.rectangle(
                        [
                            current_x - highlight_padding, 
                            y_position + line_idx * line_height - highlight_padding,
                            current_x + word_width + highlight_padding,
                            y_position + line_idx * line_height + word_height + highlight_padding
                        ],
                        fill=highlight_color
                    )
            # Special handling for gaming_neon template - apply neon colors to all words
            elif template.lower() == "gaming_neon":
                # Define the neon colors
                neon_colors = [
                    (0, 255, 255, 255),   # Cyan (#00FFFF)
                    (255, 0, 255, 255),   # Magenta (#FF00FF)
                    (0, 123, 255, 255),   # Blue (#007BFF)
                    (255, 105, 180, 255), # Hot Pink (#FF69B4)
                    (57, 255, 20, 255)    # Neon Green (#39FF14)
                ]
                
                # Use a consistent color for all words in a line
                # This ensures all words in the same line get the same color
                # But different lines might get different colors
                line_hash = hash(line) if line else random.randint(0, 1000)
                color_index = line_hash % len(neon_colors)
                word_color = neon_colors[color_index]
                
            # Special case for MrBeast template - switch between yellow, red, and green for highlighted words
            elif template.lower() == "mrbeast" and is_highlighted:
                # Randomly choose between yellow, red, and green for each highlighted word
                mrbeast_colors = [
                    (255, 255, 0, 255),  # Yellow
                    (255, 0, 0, 255),    # Red
                    (0, 255, 0, 255)     # Green
                ]
                word_color = random.choice(mrbeast_colors)
            
            # For all other templates, use the highlight color from the dictionary if highlighted, otherwise white
            else:
                template_lower = template.lower()
                
                # Use custom colors if provided
                if custom_highlight_color and is_highlighted:
                    # Convert hex color to RGBA
                    h = custom_highlight_color.lstrip('#')
                    word_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                elif custom_text_color and not is_highlighted:
                    # Convert hex color to RGBA
                    h = custom_text_color.lstrip('#')
                    word_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                # Otherwise use template colors
                elif is_highlighted and template_lower in highlight_colors:
                    word_color = highlight_colors[template_lower]
                else:
                    # Default fallback for any other templates or non-highlighted words
                    word_color = highlight_colors.get(template_lower, (255, 255, 255, 255)) if is_highlighted else (255, 255, 255, 255)
            
            # Add neon glow effect for gaming_neon, neon_heartbeat, and streamer_pro templates
            if template.lower() in ["gaming_neon", "neon_heartbeat", "streamer_pro"]:
                # Create a vibrant neon glow effect
                # Define glow colors based on the word color
                r, g, b, a = word_color
                
                # Create multiple layers of glow with decreasing opacity - reduced intensity
                # Special case for neon_heartbeat - reduced glow but keep light effect with subtle pulse
                if template.lower() == "neon_heartbeat":
                    # Create a subtle pulsing effect based on the current time
                    pulse_intensity = 0.7 + 0.3 * abs(math.sin(time.time() * 2.5))  # Subtle pulse between 70-100% intensity
                    
                    # Apply pulse intensity to the glow opacity
                    glow_layers = [
                        {"offset": 2, "color": (r, g, b, int(90 * pulse_intensity))},   # Inner glow (more transparent)
                        {"offset": 3, "color": (r, g, b, int(50 * pulse_intensity))},   # Middle glow (reduced)
                        {"offset": 4, "color": (r, g, b, int(25 * pulse_intensity))}    # Outer glow (very transparent)
                    ]
                    # Enhanced inner light effect for neon_heartbeat
                    inner_glow = {"offset": 1, "color": (255, 255, 255, int(180 * pulse_intensity))}
                else:
                    glow_layers = [
                        {"offset": 2, "color": (r, g, b, 120)},  # Inner glow (less opaque)
                        {"offset": 3, "color": (r, g, b, 70)},   # Middle glow (reduced)
                        {"offset": 5, "color": (r, g, b, 40)}    # Outer glow (more transparent)
                    ]
                    # Standard inner glow for other templates
                    inner_glow = {"offset": 1, "color": (255, 255, 255, 150)}
                
                # Draw the glow layers from outside in
                for layer in reversed(glow_layers):
                    offset = max(1, int(font_size * 0.03 * layer["offset"]))
                    glow_color = layer["color"]
                    
                    # Draw the glow in all directions
                    for dx, dy in [
                        (-offset, -offset), (0, -offset), (offset, -offset),
                        (-offset, 0),                     (offset, 0),
                        (-offset, offset),  (0, offset),  (offset, offset)
                    ]:
                        draw.text(
                            (current_x + dx, y_position + line_idx * line_height + dy),
                            word_with_space, font=font, fill=glow_color
                        )
                
                # Draw the inner white glow for extra brightness
                offset = max(1, int(font_size * 0.03 * inner_glow["offset"]))
                for dx, dy in [(-offset, 0), (0, -offset), (offset, 0), (0, offset)]:
                    draw.text(
                        (current_x + dx, y_position + line_idx * line_height + dy),
                        word_with_space, font=font, fill=inner_glow["color"]
                    )
            
            # Special handling for bold_switch template - white text with alternating highlight colors and black stroke
            elif template.lower() == "bold_switch":
                # Set text color to white for all words
                word_color = (255, 255, 255, 255)  # White text
                
                # Define the highlight colors to switch between
                cyan_color = (22, 217, 228, 255)  # #16D9E4
                yellow_color = (246, 255, 7, 255)  # #F6FF07
                
                # Determine which highlight color to use based on word position
                # Only apply color to highlighted words, keep others white
                if is_highlighted:
                    if word_idx % 2 == 0:
                        word_color = cyan_color
                    else:
                        word_color = yellow_color
                
                # Add black stroke (3px)
                outline_color = (0, 0, 0, 255)  # Black outline
                outline_thickness = 3  # 3px black outline
                
                # Draw the outline by drawing the text multiple times with small offsets
                for dx, dy in [(-outline_thickness, -outline_thickness), 
                              (outline_thickness, -outline_thickness), 
                              (-outline_thickness, outline_thickness), 
                              (outline_thickness, outline_thickness),
                              (0, -outline_thickness),
                              (0, outline_thickness),
                              (-outline_thickness, 0),
                              (outline_thickness, 0)]:
                    draw.text((current_x + dx, y_position + line_idx * line_height + dy), 
                             word_with_space, font=font, fill=outline_color)
                
            # Add outline for templates that need it
            elif template.lower() in ["yellow_impact", "bold_white", "bold_green", "mrbeast", "green_bold", "meme_orange", "film_noir", "bold_sunshine", 
                                     "premium_orange", "premium_yellow", "meme_maker", "viral_pop", "esports_caption", "trailer_text", "minimal_white", "blue_highlight"]:
                # Set outline color and thickness based on template
                if template.lower() == "yellow_impact":
                    outline_color = (0, 0, 0, 255)  # Black outline for Yellow Impact
                    outline_thickness = max(4, int(font_size * 0.16))  # Increased thickness for black outline to match larger font
                elif template.lower() == "bold_white":
                    outline_color = (0, 0, 0, 255)  # Black outline for Bold White
                    outline_thickness = 2  # Exactly 2px black outline as requested
                elif template.lower() == "bold_green":
                    outline_color = (0, 0, 0, 255)  # Black outline for Bold Green
                    outline_thickness = 4  # Exactly 4px black outline as requested
                elif template.lower() == "meme_orange":
                    outline_color = (0, 0, 0, 255)  # Black outline for Meme Orange
                    outline_thickness = max(2, int(font_size * 0.1))  # Moderate black outline for cartoon effect

                elif template.lower() == "film_noir":
                    outline_color = (255, 255, 255, 255)  # White outline for Film Noir
                    outline_thickness = max(2, int(font_size * 0.08))  # Medium white outline for contrast
                elif template.lower() == "bold_sunshine":
                    outline_color = (0, 0, 0, 255)  # Black outline for Bold Sunshine
                    outline_thickness = 2  # Updated to 2px black outline
                elif template.lower() == "premium_orange":
                    outline_color = (0, 0, 0, 255)  # Black outline for Premium Orange
                    outline_thickness = 3  # Exactly 3px black outline as requested
                elif template.lower() == "premium_yellow":
                    outline_color = (0, 0, 0, 255)  # Black outline for Premium Yellow
                    outline_thickness = 3  # Exactly 3px black outline, same as Premium Orange
                elif template.lower() == "meme_maker":
                    outline_color = (0, 0, 0, 255)  # Black outline for Meme Maker
                    outline_thickness = max(3, int(font_size * 0.12))  # Thick black outline for memes
                elif template.lower() == "viral_pop":
                    outline_color = (0, 0, 0, 255)  # Black outline for Viral Pop
                    outline_thickness = max(2, int(font_size * 0.1))  # Medium black outline
                elif template.lower() == "esports_caption":
                    outline_color = (0, 0, 0, 255)  # Black outline for Esports Caption
                    outline_thickness = max(2, int(font_size * 0.1))  # Medium black outline
                elif template.lower() == "minimal_white":
                    outline_color = (0, 0, 0, 255)  # Black outline for Minimal White
                    outline_thickness = 3  # Exactly 3px black outline as requested
                elif template.lower() == "blue_highlight":
                    outline_color = (0, 0, 0, 255)  # Black outline for Blue Highlight
                    outline_thickness = 1  # 1px black outline as requested

                else:
                    outline_color = (0, 0, 0, 255)  # Black outline for others
                    outline_thickness = max(2, int(font_size * 0.1))
                
                # Draw the outline by drawing the text multiple times with small offsets
                for dx, dy in [(-outline_thickness, -outline_thickness), 
                              (outline_thickness, -outline_thickness), 
                              (-outline_thickness, outline_thickness), 
                              (outline_thickness, outline_thickness),
                              (0, -outline_thickness),
                              (0, outline_thickness),
                              (-outline_thickness, 0),
                              (outline_thickness, 0)]:
                    draw.text((current_x + dx, y_position + line_idx * line_height + dy), 
                             word_with_space, font=font, fill=outline_color)
            
            # Shadow effect for Clean Anton template (removed for Green Bold)
            if False:  # Placeholder to maintain code structure
                pass
            elif template.lower() in ["elegant_pink", "minimalist_sans"]:
                # Create a subtle drop shadow effect
                shadow_offset = max(2, int(font_size * 0.06))  # Smaller offset for subtle effect
                shadow_color = (0, 0, 0, 160)  # Black with transparency
                
                # Draw multiple shadow layers for a softer effect
                for offset in range(1, shadow_offset + 1):
                    draw.text((current_x + offset, y_position + line_idx * line_height + offset), 
                            word_with_space, font=font, fill=shadow_color)
                
                # Add a subtle highlight glow for keywords
                if is_highlighted:
                    # Create a subtle glow effect for highlighted keywords
                    glow_color = (255, 255, 200, 40)  # Very subtle yellow glow
                    glow_offset = max(1, int(font_size * 0.03))  # Very small offset for subtle glow
                    
                    # Draw the glow around the text
                    for dx, dy in [(-glow_offset, -glow_offset), 
                                  (glow_offset, -glow_offset), 
                                  (-glow_offset, glow_offset), 
                                  (glow_offset, glow_offset)]:
                        draw.text((current_x + dx, y_position + line_idx * line_height + dy), 
                                word_with_space, font=font, fill=glow_color)
            
            # Special handling for trendy_gradient template - apply gradient to all text
            if template.lower() == "trendy_gradient":
                # Use pink color with white stroke
                pink_color = (255, 105, 180, 255)  # #FF69B4 (hot pink)
                white_color = (255, 255, 255, 255)  # #FFFFFF (white)
                
                # First draw the white stroke/outline
                outline_thickness = 3  # 3px white outline
                
                # Draw the outline by drawing the text multiple times with small offsets
                for dx, dy in [(-outline_thickness, -outline_thickness), 
                              (outline_thickness, -outline_thickness), 
                              (-outline_thickness, outline_thickness), 
                              (outline_thickness, outline_thickness),
                              (0, -outline_thickness),
                              (0, outline_thickness),
                              (-outline_thickness, 0),
                              (outline_thickness, 0)]:
                    draw.text((current_x + dx, y_position + line_idx * line_height + dy), 
                             word_with_space, font=font, fill=white_color)
                
                # Then draw the pink text on top
                draw.text((current_x, y_position + line_idx * line_height), 
                         word_with_space, font=font, fill=pink_color)
            else:
                # Check if the word contains asterisks (censored profanity) or is marked as profane
                if '*' in word or (is_current_word and is_profane):
                    # Draw censored text in red
                    profanity_color = (255, 0, 0, 255)  # Bright red
                    
                    # Removed the beep indicator (small red circle) as requested
                    
                    # Draw the text
                    draw.text((current_x, y_position + line_idx * line_height), word_with_space, font=font, fill=profanity_color)
                else:
                    # For creator_clean style, add a pop-out animation effect for highlighted words
                    if template.lower() == "creator_clean" and is_highlighted:
                        # Create a pop-out animation effect
                        # This is achieved by drawing the text multiple times with slight offsets and varying opacity
                        # to create a pulsing/popping effect
                        
                        # Calculate the animation phase based on the current time
                        # This creates a continuous animation effect
                        animation_phase = (time.time() * 3) % 1.0  # Cycles through 0 to 1 every 1/3 second
                        
                        # Create a pulsing effect that grows and shrinks
                        # Use a sine wave to create smooth animation
                        pulse_scale = 0.15 * math.sin(animation_phase * 2 * math.pi) + 1.0  # Varies between 0.85 and 1.15
                        
                        # Draw a subtle glow/shadow effect behind the text
                        glow_color = word_color[:3] + (100,)  # Semi-transparent version of the text color
                        glow_offsets = [
                            (1, 1), (-1, 1), (1, -1), (-1, -1),  # Diagonal offsets
                            (0, 1), (0, -1), (1, 0), (-1, 0)     # Cardinal offsets
                        ]
                        
                        for dx, dy in glow_offsets:
                            # Scale the offsets based on the pulse
                            scaled_dx = dx * pulse_scale
                            scaled_dy = dy * pulse_scale
                            
                            # Draw the glow
                            draw.text(
                                (current_x + scaled_dx, y_position + line_idx * line_height + scaled_dy), 
                                word_with_space, 
                                font=font, 
                                fill=glow_color
                            )
                        
                        # Draw the main text
                        draw.text(
                            (current_x, y_position + line_idx * line_height), 
                            word_with_space, 
                            font=font, 
                            fill=word_color
                        )
                    else:
                        # Draw the main text with normal color and font
                        draw.text((current_x, y_position + line_idx * line_height), word_with_space, font=font, fill=word_color)
            
            # Move to the next word position
            current_x += word_widths[word_idx]
    
    # Save to a temporary file in /app/tmp with high quality settings
    temp_dir = "/app/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = os.path.join(temp_dir, f"caption_{uuid.uuid4().hex}.png")
    
    # Save with high quality PNG settings to preserve colors and transparency
    text_img.save(temp_filename, "PNG", optimize=False, compress_level=1)
    
    # Keep track of the temp file for later deletion
    temp_file = type('obj', (object,), {'name': temp_filename})
    
    # Create clip from the image with better quality settings
    from moviepy.editor import ImageClip
    clip = ImageClip(temp_file.name, transparent=True).set_duration(duration)
    
    # Since we're now using full video dimensions, position the clip at (0,0)
    # The text positioning is already handled within the image
    clip = clip.set_position((0, 0))
    
    # Schedule the temporary file for deletion
    os.unlink(temp_file.name)
    
    return clip

def process_video(input_path, options=None, preloaded_model=None):
    """Process a video by transcribing it and adding captions
    
    Args:
        input_path: Path to the input video
        options: Dictionary of processing options
        preloaded_model: Optional preloaded Whisper model to use
        
    Returns:
        Dictionary with results including processed video path
    """
    # Verify paths are correct for production environment
    verify_production_paths()
    # Handle both legacy parameter style and new options dictionary
    if options is None:
        # Legacy parameter style
        template = "minimal_white"
        highlight_color = None
        filter_profanity = False
        user_plan = "free" 
        enable_speaker_tracking = False
    else:
        # New options dictionary style
        template = options.get("template", "minimal_white")
        highlight_color = options.get("highlight_color", None)
        filter_profanity = options.get("censor_profanity", False)
        user_plan = options.get("user_plan", "free")
        enable_speaker_tracking = options.get("include_speakers", False)
        # Caption customization options
        caption_vertical_position = options.get("caption_vertical_position", 50)
        caption_font_size = options.get("caption_font_size", 100)
    """Process a video by transcribing it and adding captions
    
    Args:
        input_path: Path to the input video
        template: Template to use for captions
        highlight_color: Custom color ID for highlighting words (overrides template default)
        filter_profanity: Whether to filter profanity in the video
        user_plan: The user's subscription plan ("free", "basic", or "pro")
        enable_speaker_tracking: Whether to enable speaker tracking and diarization
        caption_vertical_position: Vertical position of captions (0-100, default: 50)
        caption_font_size: Font size of captions as percentage (50-150, default: 100)
        
    Returns:
        The filename of the processed video
    """
    # Import required modules
    import sys
    # Create a debug log file to track progress
    debug_log_path = os.path.join(PROCESSED_FOLDER, f"debug_log_{uuid.uuid4().hex}.txt")
    
    def log_debug(message):
        """Write a debug message to the log file and print it"""
        try:
            with open(debug_log_path, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                log_message = f"[{timestamp}] {message}\n"
                f.write(log_message)
            print(message)
        except Exception as e:
            print(f"Error writing to debug log: {str(e)}")
    
    # Start debug logging
    log_debug(f"=== PROCESS_VIDEO FUNCTION STARTED ===")
    log_debug(f"Input parameters:")
    log_debug(f"  - input_path: {input_path}")
    log_debug(f"  - template: {template}")
    log_debug(f"  - highlight_color: {highlight_color}")
    log_debug(f"  - filter_profanity: {filter_profanity}")
    log_debug(f"  - user_plan: {user_plan}")
    log_debug(f"  - enable_speaker_tracking: {enable_speaker_tracking}")
    
    # Log system information
    log_debug(f"System information:")
    log_debug(f"  - Python version: {sys.version}")
    log_debug(f"  - Platform: {sys.platform}")
    log_debug(f"  - Upload folder: {UPLOAD_FOLDER}")
    log_debug(f"  - Processed folder: {PROCESSED_FOLDER}")
    log_debug(f"  - Input file exists: {os.path.exists(input_path)}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        error_msg = f"Input file does not exist: {input_path}"
        print(f"ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Generate a unique output filename
    filename = os.path.basename(input_path)
    print(f"Original filename: {filename}")
    
    # WebM format is no longer supported
    if filename.lower().endswith('.webm'):
        error_msg = "WebM format is no longer supported"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Get the next sequential output number
    output_number = get_next_output_number()
    
    # Use a sequential naming convention
    output_filename = f"Quickcap output {output_number}.mp4"
    original_video_filename = filename  # Store the original filename for reference
    print(f"Generated output filename: {output_filename}")
    
    # Use the configured PROCESSED_FOLDER path
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    print(f"Using output path: {output_path}")
    
    # Print absolute path for debugging
    abs_output_path = os.path.abspath(output_path)
    print(f"Absolute output path: {abs_output_path}")
        
    # Ensure the processed directory exists with proper permissions
    try:
        print(f"Ensuring processed directory exists: {PROCESSED_FOLDER}")
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        print(f"Directory created or already exists")
        
        # Make sure the directory is writable
        try:
            os.chmod(PROCESSED_FOLDER, 0o777)
            print(f"Set directory permissions to 0o777")
        except Exception as chmod_error:
            print(f"Warning: Could not set directory permissions: {str(chmod_error)}")
        
        print(f"Ensured processed directory exists: {PROCESSED_FOLDER}")
        
        # List the contents of the parent directory if it exists
        parent_dir = os.path.dirname(PROCESSED_FOLDER)
        if parent_dir and os.path.exists(parent_dir):
            print(f"Parent directory contents: {os.listdir(parent_dir)}")
        else:
            print(f"Using root directory as processed folder: {PROCESSED_FOLDER}")
            
        # Check if the processed directory is writable
        if os.access(PROCESSED_FOLDER, os.W_OK):
            print(f"Processed directory is writable: {PROCESSED_FOLDER}")
        else:
            print(f"WARNING: Processed directory is NOT writable: {PROCESSED_FOLDER}")
            
        # Get absolute path of processed folder
        abs_processed_folder = os.path.abspath(PROCESSED_FOLDER)
        print(f"Using absolute processed folder path: {abs_processed_folder}")
        
    except Exception as e:
        print(f"ERROR creating processed directory: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Error ensuring processed directory exists: {str(e)}")
        
    # No need to create parent directory separately since we're using absolute paths
    # and have already created the PROCESSED_FOLDER
    print(f"Using absolute processed folder path: {PROCESSED_FOLDER}")
    # Just double-check the processed folder exists
    if not os.path.exists(PROCESSED_FOLDER):
        try:
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)
            print(f"Created processed directory: {PROCESSED_FOLDER}")
        except Exception as e:
            print(f"Error creating processed directory: {str(e)}")
    
    # Step 1: Apply speaker tracking if enabled (for Basic and Pro plans)
    if enable_speaker_tracking and (user_plan.lower() == "pro" or user_plan.lower() == "basic"):
        try:
            print("Applying speaker tracking and diarization...")
            from backend.services.speaker_tracking import process_video_with_speaker_tracking
            
            # Create a temporary path for the speaker-tracked video
            speaker_tracked_path = os.path.join(UPLOAD_FOLDER, f"speaker_tracked_{filename}")
            print(f"Speaker tracked path: {speaker_tracked_path}")
            
            # Process the video with speaker tracking
            print(f"Processing video with speaker tracking: {input_path} -> {speaker_tracked_path}")
            process_video_with_speaker_tracking(input_path, speaker_tracked_path)
            
            # Use the speaker-tracked video for further processing
            input_path = speaker_tracked_path
            print(f"Speaker tracking completed successfully, new input path: {input_path}")
        except Exception as e:
            print(f"ERROR during speaker tracking: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Continuing with original video...")
    
    # Speech enhancement feature has been removed
    print("Speech enhancement feature has been removed, skipping...")
    
    # Step 2: Convert to 9:16 if needed
    print(f"Converting video to 9:16 ratio if needed")
    converted_path = os.path.join(UPLOAD_FOLDER, f"converted_{filename}")
    print(f"Converted path: {converted_path}")
    
    try:
        print(f"🎬 STARTING FFMPEG CONVERSION: {input_path} -> {converted_path}")
        print(f"🚨 DEBUG: About to call convert_to_9_16_ratio_ffmpeg function")
        result_path = convert_to_9_16_ratio_ffmpeg(input_path, converted_path)
        print(f"✅ FFMPEG CONVERSION COMPLETED: {result_path}")
    except Exception as e:
        print(f"❌ FFMPEG CONVERSION FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("🔄 FALLING BACK TO MOVIEPY CONVERSION...")
        # Fallback to original MoviePy function
        try:
            result_path = convert_to_9_16_ratio(input_path, converted_path)
            print(f"✅ MOVIEPY FALLBACK COMPLETED: {result_path}")
        except Exception as fallback_error:
            print(f"❌ MOVIEPY FALLBACK ALSO FAILED: {str(fallback_error)}")
            raise
    
    # Step 3: Transcribe the video with optional profanity filtering
    print(f"Transcribing video with profanity filter: {filter_profanity}")
    try:
        print(f"Transcribing video: {converted_path}")
        # Pass the preloaded model if available (from modal_app.py)
        preloaded = options.get("preloaded_model") if options is not None else preloaded_model
        if preloaded is not None:
            print("Using preloaded model from Modal container")
        transcription_result = transcribe_video(
            converted_path, 
            filter_profanity=filter_profanity,
            preloaded_model=preloaded
        )
        print(f"Transcription completed successfully")
        segments = transcription_result["segments"]
        print(f"Number of segments: {len(segments)}")
    except Exception as e:
        print(f"ERROR during transcription: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # Update the converted path if it was changed during profanity filtering
    if "video_path" in transcription_result and transcription_result["video_path"] != converted_path:
        print(f"Converted path updated during profanity filtering")
        print(f"Old path: {converted_path}")
        converted_path = transcription_result["video_path"]
        print(f"New path: {converted_path}")
    
    # Save the transcription data to a JSON file
    print(f"Saving transcription data to JSON file")
    transcription_filename = output_filename.rsplit('.', 1)[0] + '.json'
    transcription_path = os.path.join(PROCESSED_FOLDER, transcription_filename)
    print(f"Transcription path: {transcription_path}")
    
    try:
        print(f"Writing transcription data to: {transcription_path}")
        with open(transcription_path, 'w') as f:
            json.dump({
                "segments": segments,
                "filename": output_filename,
                "profanity_filtered": filter_profanity
            }, f, indent=2)
        print(f"Transcription data saved successfully")
    except Exception as e:
        print(f"ERROR saving transcription data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # Step 4: Split into phrases with max 3 words
    print(f"Splitting transcription into phrases with max 3 words")
    try:
        phrases = split_into_phrases(segments, max_words=3)
        print(f"Number of phrases: {len(phrases)}")
    except Exception as e:
        print(f"ERROR splitting into phrases: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save a copy of the converted video without captions for future reprocessing
    print(f"Saving a copy of the converted video without captions")
    original_video_filename = f"original_{uuid.uuid4().hex}.mp4"
    original_video_path = os.path.join(PROCESSED_FOLDER, original_video_filename)
    print(f"Original video path: {original_video_path}")
    
    # For basic and pro plans, we need to resize the original video to the appropriate resolution
    print(f"Checking if video needs to be resized for user plan: {user_plan}")
    if user_plan.lower() in ["basic", "pro"]:
        try:
            print(f"Resizing video for {user_plan} plan")
            # Load the video
            print(f"Loading video from: {converted_path}")
            original_clip = VideoFileClip(converted_path)
            original_clip.fps = original_clip.fps or 24
            original_width, original_height = original_clip.size
            print(f"Original video dimensions: {original_width}x{original_height}")
            
            # Set resolution based on user plan
            if user_plan.lower() == "basic":
                print(f"Applying basic plan resolution (1080p)")
                # 1080p resolution for basic plan
                aspect_ratio = original_width / original_height
                print(f"Aspect ratio: {aspect_ratio}")
                if aspect_ratio < 1:  # Vertical video
                    print(f"Vertical video detected")
                    new_height = 1920
                    new_width = int(new_height * aspect_ratio)
                    print(f"Vertical video dimensions: {new_width}x{new_height}")
                else:  # Horizontal video
                    print(f"Horizontal video detected")
                    new_width = 1920
                    new_height = int(new_width / aspect_ratio)
                    print(f"Horizontal video dimensions: {new_width}x{new_height}")
                
                # Ensure dimensions are even
                print(f"Ensuring dimensions are even")
                new_width, new_height = ensure_even_dimensions(new_width, new_height)
                print(f"Final dimensions after ensuring even: {new_width}x{new_height}")
                
                # Resize to 1080p
                print(f"Resizing original video from {original_width}x{original_height} to 1080p: {new_width}x{new_height}")
                try:
                    original_clip = original_clip.resize((new_width, new_height))
                    original_clip = original_clip.set_fps(original_clip.fps)
                    print(f"Video resized successfully")
                except Exception as e:
                    print(f"ERROR during video resize: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            elif user_plan.lower() == "pro":
                print(f"Applying pro plan resolution (4K)")
                # Check if the original resolution is already high
                if original_width > 3840 or original_height > 3840:
                    # If already high resolution, just use the original size
                    print(f"Video already has high resolution ({original_width}x{original_height}), keeping original dimensions")
                    new_width, new_height = original_width, original_height
                else:
                    # 4K resolution is more stable than 8K
                    # Using 4K as the maximum resolution to avoid memory issues
                    aspect_ratio = original_width / original_height
                    print(f"Aspect ratio: {aspect_ratio}")
                    if aspect_ratio < 1:  # Vertical video
                        print(f"Vertical video detected")
                        new_height = 3840
                        new_width = int(new_height * aspect_ratio)
                        print(f"Vertical video dimensions: {new_width}x{new_height}")
                    else:  # Horizontal video
                        print(f"Horizontal video detected")
                        new_width = 3840
                        new_height = int(new_width / aspect_ratio)
                        print(f"Horizontal video dimensions: {new_width}x{new_height}")
                    
                    # Ensure dimensions are even
                    print(f"Ensuring dimensions are even")
                    new_width, new_height = ensure_even_dimensions(new_width, new_height)
                    print(f"Final dimensions after ensuring even: {new_width}x{new_height}")
                    
                    # Resize to 4K (safer than 8K)
                    print(f"Resizing original video from {original_width}x{original_height} to 4K: {new_width}x{new_height}")
                    try:
                        original_clip = original_clip.resize((new_width, new_height))
                        original_clip = original_clip.set_fps(original_clip.fps)
                        print(f"Video resized successfully")
                    except Exception as e:
                        print(f"ERROR during video resize: {str(e)}")
                        print(f"Error type: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                        raise
        except Exception as e:
            print(f"ERROR during video resizing: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Falling back to original video dimensions")
            # If resizing fails, use the original dimensions
            new_width, new_height = original_width, original_height
        
        # Set video quality based on user plan
        print(f"Setting video quality based on user plan: {user_plan}")
        video_bitrate = "8000k" if user_plan.lower() == "basic" else "12000k"
        audio_bitrate = "192k" if user_plan.lower() == "basic" else "256k"
        preset = "medium" if user_plan.lower() == "basic" else "slow"
        print(f"Video bitrate: {video_bitrate}, Audio bitrate: {audio_bitrate}, Preset: {preset}")
        
        # Write the resized video with improved error handling
        try:
            print(f"Preparing to write original video to: {original_video_path}")
            # Adjust ffmpeg parameters based on resolution
            ffmpeg_params = [
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-movflags", "+faststart"
            ]
            print(f"Base ffmpeg parameters: {ffmpeg_params}")
            
            # For 4K or higher, use a more compatible level
            if new_width >= 3840 or new_height >= 3840:
                print(f"Using level 5.1 for 4K or higher resolution")
                ffmpeg_params.extend(["-level", "5.1"])
            else:
                print(f"Using level 4.0 for resolution below 4K")
                ffmpeg_params.extend(["-level", "4.0"])
            
            # For very high resolutions, use a faster preset to avoid memory issues
            if new_width >= 3840 or new_height >= 3840:
                actual_preset = "fast"  # Use faster preset for high resolutions
                print(f"Using faster preset 'fast' for high resolution")
            else:
                actual_preset = preset
                print(f"Using standard preset '{preset}' for normal resolution")
                
            print(f"Writing video with dimensions: {new_width}x{new_height}, preset: {actual_preset}")
            print(f"Final ffmpeg parameters: {ffmpeg_params}")
            
            print(f"Starting to write original video file...")
            original_clip.write_videofile(
                original_video_path,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate=audio_bitrate,
                bitrate=video_bitrate,
                preset=actual_preset,
                temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                remove_temp=True,
                ffmpeg_params=ffmpeg_params,
                threads=2  # Limit threads to reduce memory usage
            )
        except Exception as e:
            print(f"ERROR writing high-resolution video: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Attempting to write video at a lower resolution...")
            
            try:
                print(f"Implementing fallback strategy for video writing")
                # Fall back to a lower resolution if the high-res export fails
                if new_width > 1920 or new_height > 1920:
                    print(f"Current resolution {new_width}x{new_height} is too high, scaling down to 1080p")
                    # Scale down to 1080p
                    scale_factor = min(1920 / new_width, 1920 / new_height)
                    fallback_width = int(new_width * scale_factor)
                    fallback_height = int(new_height * scale_factor)
                    print(f"Scale factor: {scale_factor}, new dimensions: {fallback_width}x{fallback_height}")
                    
                    # Ensure dimensions are even
                    print(f"Ensuring dimensions are even")
                    fallback_width, fallback_height = ensure_even_dimensions(fallback_width, fallback_height)
                    print(f"Final fallback dimensions: {fallback_width}x{fallback_height}")
                    
                    print(f"Falling back to {fallback_width}x{fallback_height} resolution")
                    
                    # Resize to the lower resolution
                    print(f"Resizing clip to lower resolution")
                    try:
                        resized_clip = original_clip.resize((fallback_width, fallback_height))
                        resized_clip = resized_clip.set_fps(original_clip.fps)
                        print(f"Resize successful")
                    except Exception as resize_error:
                        print(f"ERROR during fallback resize: {str(resize_error)}")
                        print(f"Error type: {type(resize_error).__name__}")
                        import traceback
                        traceback.print_exc()
                        print(f"Using original clip without resizing")
                        resized_clip = original_clip
                    
                    # Write with safer settings
                    print(f"Writing video with safer settings")
                    fallback_params = [
                        "-pix_fmt", "yuv420p",
                        "-profile:v", "main",
                        "-level", "4.0",
                        "-movflags", "+faststart"
                    ]
                    print(f"Fallback ffmpeg parameters: {fallback_params}")
                    
                    print(f"Starting to write fallback video file...")
                    resized_clip.write_videofile(
                        original_video_path,
                        codec="libx264",
                        audio_codec="aac",
                        audio_bitrate="128k",
                        bitrate="4000k",
                        preset="fast",
                        temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                        remove_temp=True,
                        ffmpeg_params=fallback_params,
                        threads=2
                    )
                    resized_clip.close()
                else:
                    # If already at a reasonable resolution, just write with safer settings
                    print(f"Resolution {new_width}x{new_height} is already reasonable, using safer settings")
                    safe_params = [
                        "-pix_fmt", "yuv420p",
                        "-profile:v", "main",
                        "-level", "4.0",
                        "-movflags", "+faststart"
                    ]
                    print(f"Safe ffmpeg parameters: {safe_params}")
                    
                    print(f"Starting to write video with safe settings...")
                    original_clip.write_videofile(
                        original_video_path,
                        codec="libx264",
                        audio_codec="aac",
                        audio_bitrate="128k",
                        bitrate="4000k",
                        preset="fast",
                        temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                        remove_temp=True,
                        ffmpeg_params=safe_params,
                        threads=2
                    )
                    print(f"Video written successfully with safe settings")
            except Exception as fallback_error:
                print(f"ERROR: Fallback video writing also failed: {str(fallback_error)}")
                print(f"Error type: {type(fallback_error).__name__}")
                import traceback
                traceback.print_exc()
                # If all video processing fails, copy the original file as a last resort
                import shutil
                print("Using original video file as fallback - copying original file")
                try:
                    print(f"Copying {converted_path} to {original_video_path}")
                    shutil.copy2(converted_path, original_video_path)
                    print(f"File copied successfully")
                except Exception as copy_error:
                    print(f"ERROR during file copy: {str(copy_error)}")
                    print(f"Error type: {type(copy_error).__name__}")
                    import traceback
                    traceback.print_exc()
                    raise
        finally:
            # Always close the clip to free resources
            try:
                print(f"Closing original clip to free resources")
                original_clip.close()
                print(f"Clip closed successfully")
            except Exception as close_error:
                print(f"WARNING: Error closing clip: {str(close_error)}")
                pass
    else:
        # For free plan, resize to 720p
        print(f"Processing for free plan (720p)")
        print(f"Loading video from: {converted_path}")
        original_clip = VideoFileClip(converted_path)
        original_clip.fps = original_clip.fps or 24
        original_width, original_height = original_clip.size
        print(f"Original video dimensions: {original_width}x{original_height}")
        
        # 720p resolution for free plan
        aspect_ratio = original_width / original_height
        if aspect_ratio < 1:  # Vertical video
            new_height = 1280
            new_width = int(new_height * aspect_ratio)
        else:  # Horizontal video
            new_width = 1280
            new_height = int(new_width / aspect_ratio)
            
        # Ensure dimensions are even
        new_width, new_height = ensure_even_dimensions(new_width, new_height)
        
        try:
            # Resize to 720p
            print(f"Resizing original video from {original_width}x{original_height} to 720p: {new_width}x{new_height}")
            resized_clip = original_clip.resize((new_width, new_height))
            resized_clip = resized_clip.set_fps(original_clip.fps)
            
            # Write with basic settings
            resized_clip.write_videofile(
                original_video_path,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate="128k",
                bitrate="2000k",
                preset="fast",
                temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                remove_temp=True,
                ffmpeg_params=[
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "main",
                    "-level", "4.0",
                    "-movflags", "+faststart"
                ],
                threads=2
            )
            resized_clip.close()
            original_clip.close()
        except Exception as e:
            print(f"Error resizing free plan video: {str(e)}")
            # If resizing fails, fall back to copying the original
            import shutil
            print("Falling back to original video dimensions for free plan")
            shutil.copy2(converted_path, original_video_path)
            try:
                original_clip.close()
            except:
                pass
    
    # Step 4: Add captions to the video with the selected template and highlight color
    # Add watermark for free plan users
    add_watermark = user_plan.lower() == "free"
    print(f"Adding captions to video: {converted_path} -> {output_path}")
    print(f"Template: {template}, Highlight color: {highlight_color}, Add watermark: {add_watermark}")
    print(f"Caption customization: vertical position={caption_vertical_position}%, font size={caption_font_size}%")
    
    try:
        print(f"Calling add_captions_to_video function")
        add_captions_to_video(
            converted_path, 
            output_path, 
            phrases, 
            template=template, 
            highlight_color=highlight_color, 
            add_watermark=add_watermark, 
            user_plan=user_plan,
            caption_vertical_position=caption_vertical_position,
            caption_font_size=caption_font_size
        )
        print(f"add_captions_to_video function completed successfully")
    except Exception as e:
        import traceback
        import sys
        
        print(f"\n\n==== CRITICAL ERROR IN ADD_CAPTIONS_TO_VIDEO CALL ====")
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Get the full exception info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Print the traceback
        print(f"\nDetailed traceback:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        # Re-raise the exception to be caught by the caller
        raise
    
    # Verify the output file exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Successfully created output file: {output_path} (Size: {file_size} bytes)")
        
        # Verify the processed directory contents
        print(f"Files in processed directory: {os.listdir(PROCESSED_FOLDER)}")
        
        # Verify file permissions
        try:
            import stat
            file_stat = os.stat(output_path)
            file_perms = stat.filemode(file_stat.st_mode)
            print(f"File permissions: {file_perms}")
        except Exception as e:
            print(f"Error checking file permissions: {str(e)}")
    else:
        print(f"WARNING: Output file not created: {output_path}")
        
        # Check if the directory exists and is writable
        output_dir = os.path.dirname(output_path)
        if os.path.exists(output_dir):
            print(f"Output directory exists: {output_dir}")
            print(f"Directory contents: {os.listdir(output_dir)}")
            
            try:
                import stat
                dir_stat = os.stat(output_dir)
                dir_perms = stat.filemode(dir_stat.st_mode)
                print(f"Directory permissions: {dir_perms}, Owner: {dir_stat.st_uid}, Group: {dir_stat.st_gid}")
                
                # Try to create a test file
                test_file = os.path.join(output_dir, "test_write.txt")
                with open(test_file, "w") as f:
                    f.write("Test write access")
                print(f"Successfully created test file: {test_file}")
                os.remove(test_file)
            except Exception as e:
                print(f"Error testing directory write access: {str(e)}")
        else:
            print(f"Output directory does not exist: {output_dir}")
    
    # Clean up temporary files
    # Add a small delay to ensure all file handles are closed
    import time
    time.sleep(0.5)  # 500ms delay
    
    try:
        print(f"Cleaning up temporary files")
        if os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                print(f"Removed converted file: {converted_path}")
            except PermissionError:
                print(f"Warning: Could not remove file {converted_path} - it may be in use")
        
        # WebM format is no longer supported
        
        # Don't remove the original file here, let the API endpoint handle it
        
        print(f"Process video function completed successfully")
        print(f"Returning output_filename: {output_filename}")
        print(f"Returning original_video_filename: {original_video_filename}")
        
        return output_filename, original_video_filename
    except Exception as e:
        import traceback
        import sys
        
        print(f"\n\n==== CRITICAL ERROR IN PROCESS_VIDEO (FINAL RETURN) ====")
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Get the full exception info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Print the traceback
        print(f"\nDetailed traceback:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        # Re-raise the exception to be caught by the caller
        raise

def add_captions_to_video_ffmpeg(video_path, output_path, phrases, template="minimal_white", highlight_color=None, add_watermark=False, user_plan="free", caption_vertical_position=50, caption_font_size=100):
    """Add captions to video using FFmpeg drawtext filters (HIGH PERFORMANCE)
    
    Args:
        video_path: Path to the input video
        output_path: Path where the captioned video will be saved
        phrases: List of phrases with timestamps
        template: Caption template to use
        highlight_color: Custom color ID for highlighting words (overrides template default)
        add_watermark: Whether to add a watermark to the video (for free plan)
        user_plan: The user's subscription plan ("free", "basic", or "pro")
        caption_vertical_position: Vertical position of captions (0-100, default: 50)
        caption_font_size: Font size of captions as percentage (50-150, default: 100)
    """
    import time
    start_time = time.time()
    
    print(f"Starting FFmpeg-based caption rendering...")
    print(f"Input video: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Template: {template}")
    print(f"Phrases count: {len(phrases)}")
    print(f"Vertical position: {caption_vertical_position}%")
    print(f"Font size: {caption_font_size}%")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Calculate actual font size based on percentage
    base_font_size = 40
    actual_font_size = int(base_font_size * (caption_font_size / 100))
    
    # Prepare custom colors if highlight_color is specified
    custom_colors = None
    if highlight_color:
        # Map highlight color IDs to actual colors
        color_map = {
            'yellow': '#FFFF00',
            'blue': '#0099FF', 
            'green': '#00FF00',
            'red': '#FF0000',
            'pink': '#FF69B4',
            'orange': '#FF8800',
            'purple': '#9900FF'
        }
        if highlight_color in color_map:
            custom_colors = {
                'text': 'white',
                'highlight': color_map[highlight_color],
                'outline': 'black'
            }
    
    # Quality setting based on user plan
    # Get quality setting from config
    video_quality = get_quality_for_plan(user_plan)
    
    try:
        # Use FFmpeg renderer for high-performance caption rendering
        success = ffmpeg_renderer.render_video_with_captions(
            input_video_path=video_path,
            output_video_path=output_path,
            phrases=phrases,
            template=template,
            font_size=actual_font_size,
            vertical_position=caption_vertical_position,
            custom_colors=custom_colors,
            enable_word_highlighting=True,
            video_quality=video_quality
        )
        
        if not success:
            raise RuntimeError("FFmpeg caption rendering failed")
        
        # Add watermark if needed (for free plan)
        if add_watermark and user_plan.lower() == "free":
            watermarked_output = output_path.replace('.mp4', '_watermarked.mp4')
            watermark_success = add_watermark_ffmpeg(output_path, watermarked_output)
            
            if watermark_success:
                # Replace original with watermarked version
                import shutil
                shutil.move(watermarked_output, output_path)
                print("Watermark added successfully")
            else:
                print("Warning: Watermark addition failed, continuing without watermark")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"FFmpeg caption rendering completed successfully!")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Output file: {output_path}")
        print(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"Error in FFmpeg caption rendering: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original MoviePy method if FFmpeg fails
        print("Falling back to MoviePy caption rendering...")
        return add_captions_to_video_moviepy(video_path, output_path, phrases, template, 
                                           highlight_color, add_watermark, user_plan, 
                                           caption_vertical_position, caption_font_size)


def add_watermark_ffmpeg(input_path, output_path):
    """Add watermark using FFmpeg"""
    try:
        import subprocess
        
        # Check if logo file exists
        logo_path = os.path.join('static', 'logo.png')
        
        if os.path.exists(logo_path):
            # Use image watermark
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-i', logo_path,
                '-filter_complex', 
                '[1:v]scale=100:100[logo];[0:v][logo]overlay=20:H-h-20:alpha=0.7',
                '-c:a', 'copy',
                output_path
            ]
        else:
            # Use text watermark
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', 
                "drawtext=text='Quickcap':fontsize=24:fontcolor=white@0.7:x=20:y=h-th-20",
                '-c:a', 'copy',
                output_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error adding watermark with FFmpeg: {str(e)}")
        return False


def add_captions_to_video(video_path, output_path, phrases, template="minimal_white", highlight_color=None, add_watermark=False, user_plan="free", caption_vertical_position=50, caption_font_size=100):
    """Add captions to video - uses FFmpeg by default with MoviePy fallback"""
    return add_captions_to_video_ffmpeg(video_path, output_path, phrases, template, 
                                      highlight_color, add_watermark, user_plan, 
                                      caption_vertical_position, caption_font_size)


def add_captions_to_video_moviepy(video_path, output_path, phrases, template="minimal_white", highlight_color=None, add_watermark=False, user_plan="free", caption_vertical_position=50, caption_font_size=100):
    """Original MoviePy-based caption rendering (FALLBACK)
    Add captions to video based on phrases and timestamps
    
    Args:
        video_path: Path to the input video
        output_path: Path where the captioned video will be saved
        phrases: List of phrases with timestamps
        template: Caption template to use ("minimal_white", "mrbeast", "tiktok", or "educational")
        highlight_color: Custom color ID for highlighting words (overrides template default)
        add_watermark: Whether to add a watermark to the video (for free plan)
        user_plan: The user's subscription plan ("free", "basic", or "pro")
        caption_vertical_position: Vertical position of captions (0-100, default: 50)
        caption_font_size: Font size of captions as percentage (50-150, default: 100)
    """
    # Start timing the process
    import time
    start_time = time.time()
    
    # Try to use hardware acceleration if available
    try:
        # Check if we've already determined hardware acceleration availability
        if not hasattr(add_captions_to_video, 'hw_accel_checked'):
            add_captions_to_video.hw_accel_checked = True
            add_captions_to_video.hw_accel_available = False
            
            # Check for NVIDIA GPU
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print("NVIDIA GPU detected, will try to use hardware acceleration")
                    add_captions_to_video.hw_accel_available = True
            except:
                pass
        
        # Load video with hardware acceleration if available
        if getattr(add_captions_to_video, 'hw_accel_available', False):
            print("Using hardware acceleration for video processing")
            video = VideoFileClip(video_path, audio_buffersize=2000, audio_fps=44100, audio_nbytes=2)
        else:
            video = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error setting up hardware acceleration: {str(e)}, falling back to standard mode")
        video = VideoFileClip(video_path)
    
    # Set default FPS if not available
    video.fps = video.fps or 24
    video_size = video.size
    
    print(f"Video loaded: {video_path}, size: {video_size}, duration: {video.duration:.1f}s")
    
    caption_clips = []
    
    # Use consistent font size for all videos
    # Word by Word Templates
    if template.lower() == "yellow_impact":
        font_size_percentage = 0.06  # Significantly increased font size for Yellow Impact template
    elif template.lower() == "bold_white":
        font_size_percentage = 0.055  # Increased font size for Bold White template
    elif template.lower() == "bold_green":
        font_size_percentage = 0.04  # 20% smaller for word-by-word templates
    
    # Premium Templates
    elif template.lower() == "neon_heartbeat":
        font_size_percentage = 0.048  # Medium font size for neon effect
    elif template.lower() == "premium_yellow":
        font_size_percentage = 0.042  # 16% smaller, same as Premium Orange
    elif template.lower() == "premium_orange":
        font_size_percentage = 0.042  # 16% smaller for Premium Orange template
    
    # Entertainment/Vlogger Templates
    elif template.lower() == "bold_sunshine":
        font_size_percentage = 0.05  # Standard size for entertainment
    elif template.lower() == "creator_highlight":
        font_size_percentage = 0.048  # Medium size for highlight background
    elif template.lower() == "vlog_caption":
        font_size_percentage = 0.045  # Slightly smaller for clean vlog look
    elif template.lower() == "mrbeast":
        font_size_percentage = 0.055  # Larger for impact
    elif template.lower() == "vlogger_bold":
        font_size_percentage = 0.05  # Standard bold size
    elif template.lower() == "creator_clean":
        font_size_percentage = 0.048  # Clean medium size
    elif template.lower() == "reaction_pop":
        font_size_percentage = 0.052  # Slightly larger for reactions
    
    # Classic/Minimal Templates
    elif template.lower() == "minimalist_sans":
        font_size_percentage = 0.045  # Smaller for minimalist look
    elif template.lower() == "minimal_white":
        font_size_percentage = 0.05  # Standard minimal size
    elif template.lower() == "elegant_pink":
        font_size_percentage = 0.048  # Medium elegant size
    
    # Social Media Templates
    elif template.lower() == "tiktok_trend":
        font_size_percentage = 0.052  # Larger for trend visibility
    elif template.lower() == "reels_ready":
        font_size_percentage = 0.05  # Standard reels size
    elif template.lower() == "tiktok":
        font_size_percentage = 0.048  # Medium TikTok size
    elif template.lower() == "insta_story":
        font_size_percentage = 0.045  # Smaller for story format
    elif template.lower() == "blue_highlight":
        font_size_percentage = 0.05  # Standard highlight size
    
    # Educational/Informative Templates
    elif template.lower() == "explainer_pro":
        font_size_percentage = 0.048  # Medium for readability
    elif template.lower() == "science_journal":
        font_size_percentage = 0.045  # Smaller for academic feel
    
    # Gaming Templates
    elif template.lower() == "streamer_pro":
        font_size_percentage = 0.052  # Larger for gaming impact
    elif template.lower() == "esports_caption":
        font_size_percentage = 0.055  # Large for esports visibility
    elif template.lower() == "gaming_neon":
        font_size_percentage = 0.05  # Standard neon size
    
    # Cinematic/Film Templates
    elif template.lower() == "film_noir":
        font_size_percentage = 0.048  # Medium cinematic size
    elif template.lower() == "cinematic_quote":
        font_size_percentage = 0.055  # Larger for dramatic quotes
    
    # Comedy/Memes Templates
    elif template.lower() == "meme_orange":
        font_size_percentage = 0.052  # Larger for meme impact
    
    # Trendy/Viral Templates
    elif template.lower() == "meme_maker":
        font_size_percentage = 0.058  # Large for meme visibility
    elif template.lower() == "viral_pop":
        font_size_percentage = 0.038  # 24% smaller for Viral Pop template
    elif template.lower() == "green_bold":
        font_size_percentage = 0.05  # Standard bold size
    elif template.lower() == "trendy_gradient":
        font_size_percentage = 0.052  # Slightly larger for trend appeal
    
    # Default fallback
    else:
        font_size_percentage = 0.05
        
    # Calculate base font size
    base_font_size = int(video_size[1] * font_size_percentage)
    
    # Apply custom font size scaling (50-150%)
    # Ensure the font size is within reasonable bounds
    font_size_scale = max(0.5, min(1.5, caption_font_size / 100))
    font_size = int(base_font_size * font_size_scale)
    
    print(f"Font size calculation: base={base_font_size}, scale={font_size_scale:.2f}, final={font_size}")
    
    # OPTIMIZATION 8: Precompute common values and cache font objects
    is_vertical = video_size[0] / video_size[1] <= 0.65  # Approximately 9:16 or narrower
    
    print(f"Using caption vertical position: {caption_vertical_position}%")
    
    # OPTIMIZATION 9: Batch process phrases in chunks to reduce memory pressure
    # Process phrases in batches to avoid memory issues with very long videos
    # For shorter videos, use larger batches to reduce overhead
    video_duration = clip.duration
    
    # Adaptive batch size based on video duration
    if video_duration < 30:  # Short videos (under 30 seconds)
        batch_size = 100     # Process more phrases at once
    elif video_duration < 60:  # Medium videos (30-60 seconds)
        batch_size = 50      # Standard batch size
    else:                    # Long videos (over 60 seconds)
        batch_size = 30      # Smaller batches to manage memory
        
    print(f"Video duration: {video_duration:.1f}s, using batch size: {batch_size}")
    
    # OPTIMIZATION 10: Prepare a font cache to avoid reloading fonts
    # We're already using the global _font_cache, so this is redundant
    # font_cache = {}
    
    # Standard caption processing for all templates
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i+batch_size]
        print(f"Processing caption batch {i//batch_size + 1} ({len(batch_phrases)} phrases)")
        
        for phrase in batch_phrases:
            start_time = phrase["start"]
            end_time = phrase["end"]
            duration = end_time - start_time
            
            # Special handling for Word by Word templates
            if template.lower() in ["yellow_impact", "bold_white", "bold_green"] and "word_timestamps" in phrase and len(phrase["word_timestamps"]) > 0:
                # For Yellow Impact template, we'll create a clip for each word individually
                for word_data in phrase["word_timestamps"]:
                    word_start = word_data["start"]
                    word_end = word_data["end"]
                    word_duration = word_end - word_start
                    
                    # Create a caption clip for just this word with increased font size
                    word_clip = create_caption_clip(
                        word_data["word"],  # Just the current word
                        video_size,
                        word_duration,
                        font_size=int(font_size * 1.1),  # Increase font size by only 10%
                        template=template,
                        caption_vertical_position=caption_vertical_position,
                        is_profane=word_data.get("is_profane", False)
                    )
                    
                    # Position the caption at 60% of the vertical screen height
                    # Calculate the position (60% from the top)
                    position_y = int(video_size[1] * 0.6)
                    positioned_clip = word_clip.set_position(('center', position_y)).set_start(word_start).set_duration(word_duration)
                    
                    # Add the caption clip to the list
                    caption_clips.append(positioned_clip)
                
                # Skip the regular processing for this phrase since we've handled it word by word
                continue
                
            # Check if we have word-level timestamps for other templates
            if "word_timestamps" in phrase and len(phrase["word_timestamps"]) > 0:
                # Create a caption clip for each word with its own timing
                for word_data in phrase["word_timestamps"]:
                    word_start = word_data["start"]
                    word_end = word_data["end"]
                    word_duration = word_end - word_start
                    
                    # OPTIMIZATION 11: Use a shared color mapping for all words
                    # Define custom color mapping if not already defined
                    if not hasattr(add_captions_to_video, 'custom_colors'):
                        add_captions_to_video.custom_colors = {
                            'white': (255, 255, 255, 255),  # Pure white
                            'neon-green': (57, 255, 20, 255),  # Neon green (#39FF14)
                            'electric-blue': (8, 146, 208, 255),  # Electric blue (#0892D0)
                            'cyber-yellow': (255, 211, 0, 255),  # Cyber yellow (#FFD300)
                            'hot-pink': (255, 105, 180, 255),  # Hot pink (#FF69B4)
                            'digital-orange': (255, 127, 0, 255),  # Digital orange (#FF7F00)
                            'tech-purple': (123, 104, 238, 255),  # Tech purple (#7B68EE)
                            'cyber-teal': (0, 206, 209, 255),  # Cyber teal (#00CED1)
                            'neon-red': (255, 49, 49, 255),  # Neon red (#FF3131)
                            'matrix-green': (0, 255, 65, 255),  # Matrix green (#00FF41)
                            'laser-blue': (0, 128, 255, 255),  # Laser blue (#0080FF)
                            'cyber-magenta': (255, 0, 255, 255),  # Cyber magenta (#FF00FF)
                            'quantum-cyan': (0, 255, 255, 255),  # Quantum cyan (#00FFFF)
                            'virtual-lime': (204, 255, 0, 255),  # Virtual lime (#CCFF00)
                            'digital-lavender': (181, 126, 220, 255)  # Digital lavender (#B57EDC)
                        }
                    custom_colors = add_captions_to_video.custom_colors
                
                    # OPTIMIZATION 12: Cache template highlight colors
                    if not hasattr(add_captions_to_video, 'template_colors'):
                        # Initialize template color cache
                        add_captions_to_video.template_colors = {
                            "minimal_white": (255, 255, 255, 255),  # Pure white (no highlighting)
                            "vlogger_bold": (82, 95, 225, 255),     # Vibrant blue (#525FE1)
                            "creator_clean": (0, 191, 255, 255),    # Deep sky blue
                            "reaction_pop": (255, 0, 0, 255),       # Red
                            "tiktok": (255, 105, 180, 230),         # Hot pink
                            "blue_highlight": (61, 144, 215, 230),  # Blue (#3D90D7)
                            "insta_story": (249, 206, 238, 255),    # Light pink (#F9CEEE)
                        }
                        
                        # Special case for MrBeast template with random colors
                        mrbeast_colors = [
                            (255, 255, 0, 255),  # Yellow
                            (255, 0, 0, 255),    # Red
                            (0, 255, 0, 255)     # Green
                        ]
                        add_captions_to_video.template_colors["mrbeast"] = random.choice(mrbeast_colors)
                    
                    # Get the highlight color for this template
                    template_key = template.lower()
                    if template_key in add_captions_to_video.template_colors:
                        template_highlight_color = add_captions_to_video.template_colors[template_key]
                    else:
                        # Default highlight color if template not found
                        template_highlight_color = (255, 255, 255, 255)  # White
                        
                        # Add more template colors as needed
                        if template_key == "educational":
                            # Alternate between yellow and blue for educational template
                            if random.random() < 0.5:
                                template_highlight_color = (135, 206, 235, 255)  # Sky Blue
                            else:
                                template_highlight_color = (255, 255, 0, 255)  # Yellow
                            # Cache this choice
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "tutorial_tech":
                            template_highlight_color = (61, 144, 215, 255)  # Blue (#3D90D7)
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "explainer_pro":
                            template_highlight_color = (255, 140, 0, 230)  # Orange with transparency
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "science_journal":
                            template_highlight_color = (75, 0, 130, 255)  # Indigo
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "gaming_neon":
                            # Randomly select one of the neon colors
                            neon_colors = [
                                (0, 255, 255, 255),   # Cyan (#00FFFF)
                                (255, 0, 255, 255),   # Magenta (#FF00FF)
                                (0, 123, 255, 255),   # Blue (#007BFF)
                                (255, 105, 180, 255), # Hot Pink (#FF69B4)
                                (57, 255, 20, 255)    # Neon Green (#39FF14)
                            ]
                            template_highlight_color = random.choice(neon_colors)
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                            
                            # For gaming_neon, we'll use the same color for all words in the phrase
                            # This will be handled in the word color selection logic
                        elif template_key == "film_noir":
                            template_highlight_color = (192, 192, 192, 255)  # Silver
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "cinematic_quote":
                            template_highlight_color = (255, 255, 0, 255)  # Bright yellow (#FFFF00)
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "meme_orange":
                            template_highlight_color = (255, 140, 0, 255)  # Orange
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "green_bold":
                            template_highlight_color = (0, 255, 0, 255)  # Bright Green
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "viralpop_bold":
                            template_highlight_color = (255, 255, 255, 255)  # White
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        elif template_key == "trendy_gradient":
                            template_highlight_color = (255, 26, 104, 255)  # Bright pink (#FF1A68)
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                        else:
                            template_highlight_color = (0, 255, 0, 255)  # Default: Bright lime green
                            add_captions_to_video.template_colors[template_key] = template_highlight_color
                
                    # Use custom color if specified, otherwise use template default
                if highlight_color and highlight_color in custom_colors:
                    word_highlight_color = custom_colors[highlight_color]
                else:
                    word_highlight_color = template_highlight_color
                
                # Check if this word is marked as profane
                is_profane = word_data.get("is_profane", False)
                
                # Create a caption with the full phrase text, but highlight the current word
                caption = create_caption_clip(
                    phrase["text"],
                    video_size,
                    word_duration,
                    font_size=font_size,
                    template=template,
                    caption_vertical_position=caption_vertical_position,
                    current_word=word_data["word"],
                    current_word_color=word_highlight_color,
                    is_profane=is_profane
                )
                
                caption = caption.set_start(word_start).set_duration(word_duration)
                caption_clips.append(caption)
        else:
            # Standard caption for all templates (fallback if no word timestamps)
            caption = create_caption_clip(
                phrase["text"],
                video_size,
                duration,
                font_size=font_size,
                template=template,
                caption_vertical_position=caption_vertical_position
            )
            
            caption = caption.set_start(start_time).set_duration(duration)
            caption_clips.append(caption)
    
    # Create watermark if needed
    watermark_clip = None
    if add_watermark:
        try:
            # Path to the Quickcap logo
            logo_path = os.path.join('frontend', 'public', 'Quickcap Logo.svg')
            
            # Check if the logo file exists
            if not os.path.exists(logo_path):
                print(f"Warning: Logo file not found at {logo_path}")
                # Try alternative paths
                alt_paths = [
                    os.path.join('frontend', 'public', 'Quickcap Logo.svg'),
                    os.path.join('static', 'Quickcap Logo.svg'),
                    os.path.join('static', 'images', 'Quickcap Logo.svg')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        logo_path = alt_path
                        print(f"Found logo at alternative path: {logo_path}")
                        break
            
            if os.path.exists(logo_path):
                # Convert SVG to PNG using PIL if it's an SVG file
                if logo_path.lower().endswith('.svg'):
                    try:
                        # For SVG conversion, we'll use a simple approach - create a text watermark instead
                        # since direct SVG rendering is complex
                        from PIL import Image, ImageDraw, ImageFont
                        
                        # Create a transparent image for the watermark - use a more moderate size
                        watermark_width = int(video_size[0] * 0.35)  # 35% of video width (reduced from 50%)
                        watermark_height = int(watermark_width * 0.3)  # Maintain aspect ratio (reduced from 0.4)
                        watermark_img = Image.new('RGBA', (watermark_width, watermark_height), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(watermark_img)
                        
                        # Use a font for the watermark text
                        try:
                            font_path = '/root/Fonts/Montserrat.ttf'
                            font_size = int(watermark_height * 0.6)
                            
                            # Use font cache for watermark font too
                            font_key = f"{font_path}_{font_size}"
                            if font_key in _font_cache:
                                font = _font_cache[font_key]
                            else:
                                font = ImageFont.truetype(font_path, font_size)
                                _font_cache[font_key] = font
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw the text "Quickcap"
                        text = "Quickcap"
                        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
                        position = ((watermark_width - text_width) // 2, (watermark_height - text_height) // 2)
                        
                        # Draw semi-transparent white text with 0.7 opacity (178 in RGBA)
                        draw.text(position, text, font=font, fill=(255, 255, 255, 178))  # 0.7 opacity (178 out of 255)
                        
                        # Save to a temporary file in /app/tmp
                        temp_dir = "/app/tmp"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_watermark_filename = os.path.join(temp_dir, f"watermark_{uuid.uuid4().hex}.png")
                        watermark_img.save(temp_watermark_filename)
                        
                        # Keep track of the temp file for later deletion
                        temp_watermark = type('obj', (object,), {'name': temp_watermark_filename})
                        
                        # Create the watermark clip
                        from moviepy.editor import ImageClip
                        watermark_clip = ImageClip(temp_watermark.name, transparent=True)
                        
                        # Set opacity to 0.7 (in addition to the RGBA opacity already set)
                        watermark_clip = watermark_clip.set_opacity(0.7)
                        
                        # Position in the bottom-left with safe margins to prevent cropping
                        margin_x = int(video_size[0] * 0.05)  # 5% margin from left
                        margin_y = int(video_size[1] * 0.05)  # 5% margin from bottom
                        watermark_clip = watermark_clip.set_position(lambda t: (margin_x, video_size[1] - watermark_clip.h - margin_y))
                        
                        # Set the duration to match the video
                        watermark_clip = watermark_clip.set_duration(video.duration)
                        
                        # Schedule the temporary file for deletion
                        try:
                            os.unlink(temp_watermark.name)
                        except Exception as e:
                            print(f"Warning: Could not delete temporary watermark file: {e}")
                        
                    except Exception as e:
                        print(f"Error creating watermark from SVG: {e}")
                else:
                    # If it's already a PNG or other image format
                    try:
                        from moviepy.editor import ImageClip
                        watermark_clip = ImageClip(logo_path, transparent=True)
                        
                        # Resize to a reasonable size (20% of video width)
                        watermark_width = int(video_size[0] * 0.20)
                        watermark_clip = watermark_clip.resize(width=watermark_width)
                        watermark_clip = watermark_clip.set_fps(video.fps)
                        
                        # Set opacity to 0.7
                        watermark_clip = watermark_clip.set_opacity(0.7)
                        
                        # Position in the bottom-left with safe margins to prevent cropping
                        margin_x = int(video_size[0] * 0.05)  # 5% margin from left
                        margin_y = int(video_size[1] * 0.05)  # 5% margin from bottom
                        watermark_clip = watermark_clip.set_position(lambda t: (margin_x, video_size[1] - watermark_clip.h - margin_y))
                        
                        # Set the duration to match the video
                        watermark_clip = watermark_clip.set_duration(video.duration)
                    except Exception as e:
                        print(f"Error creating watermark from image: {e}")
            else:
                # If logo file doesn't exist, create a text watermark
                from moviepy.editor import TextClip
                
                # Use a more moderate font size that won't get cut off
                watermark_clip = TextClip("Quickcap", fontsize=int(video_size[1] * 0.08),
                                         color='white', font='Arial-Bold', bg_color='transparent')
                
                # Set transparency to 0.7 as requested
                watermark_clip = watermark_clip.set_opacity(0.7)
                
                # Position in the bottom-left with safe margins to prevent cropping
                margin_x = int(video_size[0] * 0.05)  # 5% margin from left
                margin_y = int(video_size[1] * 0.05)  # 5% margin from bottom
                watermark_clip = watermark_clip.set_position(lambda t: (margin_x, video_size[1] - watermark_clip.h - margin_y))
                
                # Set the duration to match the video
                watermark_clip = watermark_clip.set_duration(video.duration)
                
        except Exception as e:
            print(f"Error adding watermark: {e}")
            import traceback
            traceback.print_exc()
            watermark_clip = None
    
    try:
        # Combine video with captions and watermark if available
        clips_to_combine = [video] + caption_clips
        if watermark_clip is not None:
            clips_to_combine.append(watermark_clip)
            
        final_video = CompositeVideoClip(clips_to_combine)
        final_video = final_video.set_fps(video.fps)
        
        # Get original dimensions
        original_width, original_height = final_video.size
        
        # Set resolution based on user plan
        if user_plan.lower() == "basic":
            # 1080p resolution for basic plan (1920 x 1080 for horizontal video)
            # Maintain aspect ratio
            aspect_ratio = original_width / original_height
            if aspect_ratio < 1:  # Vertical video
                new_height = 1920
                new_width = int(new_height * aspect_ratio)
            else:  # Horizontal video
                new_width = 1920
                new_height = int(new_width / aspect_ratio)
            
            # Ensure dimensions are even
            new_width, new_height = ensure_even_dimensions(new_width, new_height)
            
            # Resize to 1080p
            print(f"Resizing video from {original_width}x{original_height} to 1080p: {new_width}x{new_height}")
            final_video = final_video.resize((new_width, new_height))
            final_video = final_video.set_fps(video.fps)
            
        elif user_plan.lower() == "pro":
            try:
                # Check if the original resolution is already high
                if original_width > 3840 or original_height > 3840:
                    # If already high resolution, just use the original size
                    print(f"Video already has high resolution ({original_width}x{original_height}), keeping original dimensions")
                    new_width, new_height = original_width, original_height
                else:
                    # 4K resolution is more stable than 8K
                    # Using 4K as the maximum resolution to avoid memory issues
                    aspect_ratio = original_width / original_height
                    if aspect_ratio < 1:  # Vertical video
                        new_height = 3840
                        new_width = int(new_height * aspect_ratio)
                    else:  # Horizontal video
                        new_width = 3840
                        new_height = int(new_width / aspect_ratio)
                    
                    # Ensure dimensions are even
                    new_width, new_height = ensure_even_dimensions(new_width, new_height)
                    
                    # Resize to 4K (safer than 8K)
                    print(f"Resizing video from {original_width}x{original_height} to 4K: {new_width}x{new_height}")
                    final_video = final_video.resize((new_width, new_height))
                    final_video = final_video.set_fps(video.fps)
            except Exception as e:
                print(f"Error during video resizing: {str(e)}")
                print("Falling back to original video dimensions")
                # If resizing fails, use the original dimensions
                new_width, new_height = original_width, original_height
            
        else:
            # For free plan, resize to 720p
            aspect_ratio = original_width / original_height
            if aspect_ratio < 1:  # Vertical video
                new_height = 1280
                new_width = int(new_height * aspect_ratio)
            else:  # Horizontal video
                new_width = 1280
                new_height = int(new_width / aspect_ratio)
                
            # Ensure dimensions are even
            new_width, new_height = ensure_even_dimensions(new_width, new_height)
            
            # Resize to 720p
            print(f"Resizing video from {original_width}x{original_height} to 720p: {new_width}x{new_height}")
            try:
                final_video = final_video.resize((new_width, new_height))
                final_video = final_video.set_fps(video.fps)
            except Exception as e:
                print(f"Error resizing free plan video: {str(e)}")
                # If resizing fails, just ensure dimensions are even
                width, height = ensure_even_dimensions(original_width, original_height)
                if width != original_width or height != original_height:
                    try:
                        final_video = final_video.resize((width, height))
                        final_video = final_video.set_fps(video.fps)
                    except:
                        print("Falling back to original dimensions")
        
        # OPTIMIZATION 1: Use multiprocessing for rendering if available
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        
        # Get video duration for adaptive thread allocation
        video_duration = video.duration
        
        # For short videos, use more threads to speed up processing
        if video_duration < 30:
            render_threads = max(4, min(num_cores - 1, 8))  # Use up to 8 cores for short videos
        else:
            render_threads = max(2, min(num_cores - 1, 4))  # Use up to 4 cores for longer videos
            
        print(f"Video duration: {video_duration:.1f}s, using {render_threads} cores for rendering")
        
        # Set video quality based on user plan and video duration
        
        # For short videos, we can use faster presets without sacrificing much quality
        if video_duration < 30:  # Short videos (under 30 seconds)
            if user_plan.lower() == "free":
                video_bitrate = "2500k"  # Higher bitrate for short free videos
                audio_bitrate = "128k"   # Default audio bitrate
                preset = "ultrafast"     # Fastest preset for free users
            else:
                # For paid plans with short videos, use higher quality
                video_bitrate = "10000k" # Higher bitrate for short paid videos
                audio_bitrate = "192k"   # Better audio quality
                preset = "veryfast"      # Fast preset for short paid videos
            
            print(f"Short video detected ({video_duration:.1f}s), using optimized encoding settings")
        else:
            # Standard quality settings for longer videos
            if user_plan.lower() == "free":
                video_bitrate = "1500k"  # Default for free plan
                audio_bitrate = "128k"   # Default audio bitrate
                preset = "ultrafast"     # OPTIMIZATION 2: Fastest preset for free plan
            elif user_plan.lower() == "basic":
                # 4K quality for basic plan
                video_bitrate = "8000k"  # Higher bitrate for 4K
                audio_bitrate = "192k"   # Better audio quality
                preset = "veryfast"      # OPTIMIZATION 3: Faster preset for basic plan
            elif user_plan.lower() == "pro":
                # Highest quality for pro plan
                video_bitrate = "12000k" # Highest bitrate
                audio_bitrate = "256k"   # Best audio quality
                preset = "medium"        # OPTIMIZATION 4: Faster preset for pro plan that still maintains good quality
        
        # OPTIMIZATION 5: Determine optimal thread count based on video resolution
        if new_width * new_height > 1920 * 1080:
            # For high-resolution videos, use more threads
            encoding_threads = render_threads
        else:
            # For lower resolution, fewer threads are more efficient
            encoding_threads = max(2, render_threads - 1)
            
        print(f"Using preset '{preset}' with {encoding_threads} encoding threads")
            
        # OPTIMIZATION 13: Check for GPU availability and use hardware acceleration if available
        # Use the hardware acceleration check we already did when loading the video
        use_gpu = getattr(add_captions_to_video, 'hw_accel_available', False)
        gpu_codec = None
        
        # For short videos, always try to use hardware acceleration if available
        if video_duration < 30:
            # Start timing the hardware acceleration check
            hw_check_start = time.time()
            
            # Check for NVIDIA GPU (NVENC)
            try:
                import subprocess
                nvidia_check = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
                if nvidia_check.returncode == 0:
                    print("NVIDIA GPU detected, using NVENC hardware acceleration")
                    use_gpu = True
                    gpu_codec = "h264_nvenc"
                    # Override preset for NVENC
                    nvenc_presets = {
                        "ultrafast": "p1", # Fastest
                        "veryfast": "p2",
                        "medium": "p4",
                        "slow": "p7"  # Highest quality
                    }
                    nvenc_preset = nvenc_presets.get(preset, "p4")  # Default to medium quality
            except:
                pass
                
            # If NVIDIA not available, check for Intel QuickSync
            if not gpu_codec:
                try:
                    intel_check = subprocess.run("ffmpeg -hide_banner -encoders | findstr qsv", shell=True, capture_output=True, text=True)
                    if "h264_qsv" in intel_check.stdout:
                        print("Intel QuickSync detected, using QSV hardware acceleration")
                        use_gpu = True
                        gpu_codec = "h264_qsv"
                except:
                    pass
                    
            # If neither NVIDIA nor Intel, check for AMD (AMF)
            if not gpu_codec:
                try:
                    amd_check = subprocess.run("ffmpeg -hide_banner -encoders | findstr amf", shell=True, capture_output=True, text=True)
                    if "h264_amf" in amd_check.stdout:
                        print("AMD GPU detected, using AMF hardware acceleration")
                        use_gpu = True
                        gpu_codec = "h264_amf"
                except:
                    pass
                    
            print(f"Hardware acceleration check completed in {time.time() - hw_check_start:.2f}s")
        else:
            # For longer videos, use the cached hardware acceleration check
            if use_gpu:
                print("Using previously detected hardware acceleration")
                
                # Check for NVIDIA GPU (NVENC)
                try:
                    import subprocess
                    nvidia_check = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
                    if nvidia_check.returncode == 0:
                        gpu_codec = "h264_nvenc"
                        # Override preset for NVENC
                        nvenc_presets = {
                            "ultrafast": "p1", # Fastest
                            "veryfast": "p2",
                            "medium": "p4",
                            "slow": "p7"  # Highest quality
                        }
                        nvenc_preset = nvenc_presets.get(preset, "p4")  # Default to medium quality
                except:
                    pass
        
        # Prepare FFmpeg parameters
        if use_gpu:
            print(f"Using hardware acceleration with codec: {gpu_codec}")
            ffmpeg_params = [
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-threads", str(encoding_threads)  # Explicitly set thread count for FFmpeg
            ]
            
            # Add codec-specific parameters
            if gpu_codec == "h264_nvenc":
                ffmpeg_params.extend([
                    "-rc", "vbr",  # Variable bitrate mode
                    "-preset", nvenc_preset,
                    "-profile:v", "high"
                ])
            elif gpu_codec == "h264_qsv":
                ffmpeg_params.extend([
                    "-global_quality", "23",  # Quality setting for QSV
                    "-look_ahead", "1"  # Enable lookahead for better quality
                ])
            elif gpu_codec == "h264_amf":
                ffmpeg_params.extend([
                    "-quality", "speed" if user_plan.lower() == "free" else "balanced",
                    "-profile:v", "high"
                ])
                
            # For short videos, we can use higher bitrates for better quality
            if video_duration < 30:
                video_bitrate = str(int(int(video_bitrate.replace('k', '')) * 1.5)) + 'k'
                print(f"Short video detected, increasing bitrate to {video_bitrate}")
                
            # Time the video writing process
            write_start_time = time.time()
            print(f"Starting video write with hardware acceleration at {time.time() - start_time:.1f}s")
            
            try:
                # Use a temporary directory that's guaranteed to exist
                temp_dir = os.path.dirname(output_path) if os.path.exists(os.path.dirname(output_path)) else "."
                temp_audio_path = os.path.join(temp_dir, f"temp-audio-{uuid.uuid4().hex}.m4a")
                
                # For short videos, use the ultrafast preset with GPU encoding
                if video_duration < 30 and gpu_codec == "h264_nvenc":
                    # For NVENC, use p1 (fastest) preset for short videos
                    for i, param in enumerate(ffmpeg_params):
                        if param == "-preset" and i+1 < len(ffmpeg_params):
                            ffmpeg_params[i+1] = "p1"  # Fastest preset
                
                final_video.write_videofile(
                    output_path, 
                    codec=gpu_codec,
                    audio_codec="aac",
                    audio_bitrate=audio_bitrate,
                    bitrate=video_bitrate,
                    temp_audiofile=temp_audio_path,
                    remove_temp=True,
                    ffmpeg_params=ffmpeg_params,
                    threads=encoding_threads,  # Use the thread count we calculated
                    verbose=False,
                    logger=None,
                    fps=video.fps  # Ensure we maintain the original FPS
                )
                print(f"Video write completed in {time.time() - write_start_time:.1f}s (total: {time.time() - start_time:.1f}s)")
            except Exception as e:
                print(f"Error using hardware acceleration: {str(e)}, falling back to CPU encoding")
                use_gpu = False
        else:
            # Fall back to CPU encoding if no GPU is available
            print("No GPU acceleration available, using CPU encoding")
            ffmpeg_params = [
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.0",
                "-movflags", "+faststart",
                "-tune", "fastdecode",
                "-flags", "+cgop",  # Closed GOP for faster seeking
                "-threads", str(encoding_threads)  # Explicitly set thread count for FFmpeg
            ]
            
            # For short videos, we can use higher bitrates for better quality
            if video_duration < 30:
                video_bitrate = str(int(int(video_bitrate.replace('k', '')) * 1.5)) + 'k'
                print(f"Short video detected, increasing bitrate to {video_bitrate}")
                
            # Time the video writing process
            write_start_time = time.time()
            print(f"Starting video write with CPU encoding at {time.time() - start_time:.1f}s")
            
            # Use a temporary directory that's guaranteed to exist
            temp_dir = os.path.dirname(output_path) if os.path.exists(os.path.dirname(output_path)) else "."
            temp_audio_path = os.path.join(temp_dir, f"temp-audio-{uuid.uuid4().hex}.m4a")
            
            # For short videos, always use ultrafast preset with CPU encoding
            if video_duration < 30:
                preset = "ultrafast"  # Force ultrafast preset for short videos
                print(f"Short video detected, forcing 'ultrafast' preset for CPU encoding")
            
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                audio_bitrate=audio_bitrate,
                bitrate=video_bitrate,
                preset=preset,
                temp_audiofile=temp_audio_path,
                remove_temp=True,
                ffmpeg_params=ffmpeg_params,
                threads=encoding_threads,  # Use optimal thread count
                verbose=False,
                logger=None,
                fps=video.fps  # Ensure we maintain the original FPS
            )
            print(f"Video write completed in {time.time() - write_start_time:.1f}s (total: {time.time() - start_time:.1f}s)")
    finally:
        # Make sure clips are closed even if an error occurs
        try:
            video.close()
        except:
            pass
            
        try:
            final_video.close()
        except:
            pass
            
        # Close all caption clips
        for clip in caption_clips:
            try:
                clip.close()
            except:
                pass
    
    return output_path

@app.route('/')
@log_route_details
def index():
    return 'API is running'

@app.route('/api/templates', methods=['GET'])
@log_route_details
def api_templates():
    """API endpoint to get all available templates"""
    from flask import jsonify
    return jsonify(get_templates())

@app.route('/api/debug', methods=['GET'])
@log_route_details
def api_debug():
    """Debug endpoint to test various components"""
    from flask import jsonify
    import sys
    
    try:
        print("Debug endpoint called")
        
        # Test basic file operations
        test_file = os.path.join(UPLOAD_FOLDER, "test_debug.txt")
        with open(test_file, 'w') as f:
            f.write("Debug test file")
        
        # Test video loading
        from moviepy.editor import VideoFileClip
        test_video_path = None
        
        # Find a video file in the uploads folder
        for file in os.listdir(UPLOAD_FOLDER):
            if file.endswith('.mp4'):
                test_video_path = os.path.join(UPLOAD_FOLDER, file)
                break
        
        video_info = {}
        if test_video_path and os.path.exists(test_video_path):
            try:
                clip = VideoFileClip(test_video_path)
                clip.fps = clip.fps or 24
                video_info = {
                    "duration": clip.duration,
                    "size": clip.size,
                    "fps": clip.fps,
                    "path": test_video_path
                }
                clip.close()
            except Exception as e:
                video_info = {"error": str(e)}
        
        # Test transcription module
        transcription_status = "Not tested"
        try:
            # Check if the transcribe_video function is available
            if 'transcribe_video' in globals():
                transcription_status = "Transcription function available in app.py"
            else:
                transcription_status = "Transcription function not found in app.py"
        except Exception as e:
            transcription_status = f"Error: {str(e)}"
        
        # Return debug info
        return jsonify({
            "status": "success",
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "upload_folder": UPLOAD_FOLDER,
                "processed_folder": PROCESSED_FOLDER,
                "upload_folder_exists": os.path.exists(UPLOAD_FOLDER),
                "processed_folder_exists": os.path.exists(PROCESSED_FOLDER),
                "upload_folder_writable": os.access(UPLOAD_FOLDER, os.W_OK),
                "processed_folder_writable": os.access(PROCESSED_FOLDER, os.W_OK)
            },
            "video_info": video_info,
            "transcription_status": transcription_status
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/debug/transcribe', methods=['GET'])
@log_route_details
def api_debug_transcribe():
    """Debug endpoint to test transcription directly"""
    from flask import jsonify
    
    try:
        print("Transcription debug endpoint called")
        
        # Create a debug log file
        debug_log_path = os.path.join(PROCESSED_FOLDER, f"transcribe_debug_log_{uuid.uuid4().hex}.txt")
        
        def log_debug(message):
            """Write a debug message to the log file and print it"""
            try:
                with open(debug_log_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    log_message = f"[{timestamp}] {message}\n"
                    f.write(log_message)
                print(message)
            except Exception as e:
                print(f"Error writing to debug log: {str(e)}")
        
        log_debug("=== TRANSCRIPTION DEBUG STARTED ===")
        
        # Find a video file in the uploads folder
        test_video_path = None
        for file in os.listdir(UPLOAD_FOLDER):
            if file.endswith('.mp4'):
                test_video_path = os.path.join(UPLOAD_FOLDER, file)
                log_debug(f"Found test video: {test_video_path}")
                break
        
        if not test_video_path or not os.path.exists(test_video_path):
            log_debug("No test video found in uploads folder")
            return jsonify({"status": "error", "error": "No test video found"}), 404
        
        # Test transcription directly
        log_debug(f"Starting transcription test on: {test_video_path}")
        
        try:
            # Use the transcribe_video function directly from app.py
            log_debug("Using the transcribe_video function defined in app.py")
            
            # Call the transcription function
            log_debug("Calling transcribe_video function")
            segments = transcribe_video(test_video_path, filter_profanity=False)
            
            # Log the result
            log_debug(f"Transcription completed successfully")
            log_debug(f"Number of segments: {len(segments)}")
            
            # Return the first few segments
            segments_preview = segments[:3] if len(segments) > 3 else segments
            
            return jsonify({
                "status": "success",
                "message": "Transcription completed successfully",
                "segments_count": len(segments),
                "segments_preview": segments_preview,
                "debug_log": debug_log_path
            })
            
        except Exception as e:
            log_debug(f"ERROR during transcription: {str(e)}")
            log_debug(f"Error type: {type(e).__name__}")
            
            # Get the full exception info
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Format the traceback as a string
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in tb_lines:
                log_debug(line.rstrip())
            
            return jsonify({
                "status": "error", 
                "error": str(e),
                "error_type": type(e).__name__,
                "debug_log": debug_log_path
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500
    
@app.route('/api/debug/captions', methods=['GET'])
@log_route_details
def api_debug_captions():
    """Debug endpoint to test adding captions directly"""
    from flask import jsonify
    
    try:
        print("Captions debug endpoint called")
        
        # Create a debug log file
        debug_log_path = os.path.join(PROCESSED_FOLDER, f"captions_debug_log_{uuid.uuid4().hex}.txt")
        
        def log_debug(message):
            """Write a debug message to the log file and print it"""
            try:
                with open(debug_log_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    log_message = f"[{timestamp}] {message}\n"
                    f.write(log_message)
                print(message)
            except Exception as e:
                print(f"Error writing to debug log: {str(e)}")
        
        log_debug("=== CAPTIONS DEBUG STARTED ===")
        
        # Find a video file in the uploads folder
        test_video_path = None
        for file in os.listdir(UPLOAD_FOLDER):
            if file.endswith('.mp4'):
                test_video_path = os.path.join(UPLOAD_FOLDER, file)
                log_debug(f"Found test video: {test_video_path}")
                break
        
        if not test_video_path or not os.path.exists(test_video_path):
            log_debug("No test video found in uploads folder")
            return jsonify({"status": "error", "error": "No test video found"}), 404
        
        # Create a test output path
        test_output_path = os.path.join(PROCESSED_FOLDER, f"debug_captions_{uuid.uuid4().hex}.mp4")
        log_debug(f"Test output path: {test_output_path}")
        
        # Create some test phrases
        test_phrases = [
            {
                "text": "This is a test phrase",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.5},
                    {"word": "is", "start": 0.5, "end": 0.8},
                    {"word": "a", "start": 0.8, "end": 1.0},
                    {"word": "test", "start": 1.0, "end": 1.5},
                    {"word": "phrase", "start": 1.5, "end": 2.0}
                ]
            },
            {
                "text": "Testing caption functionality",
                "start": 2.5,
                "end": 4.5,
                "words": [
                    {"word": "Testing", "start": 2.5, "end": 3.0},
                    {"word": "caption", "start": 3.0, "end": 3.8},
                    {"word": "functionality", "start": 3.8, "end": 4.5}
                ]
            }
        ]
        
        log_debug(f"Created test phrases: {len(test_phrases)} phrases")
        
        # Test add_captions_to_video directly
        log_debug(f"Starting add_captions_to_video test")
        
        try:
            # Call the add_captions_to_video function
            log_debug("Calling add_captions_to_video function")
            add_captions_to_video(
                test_video_path, 
                test_output_path, 
                test_phrases, 
                template="minimal_white",
                highlight_color=None,
                caption_vertical_position=50,
                caption_font_size=100,
                add_watermark=False,
                user_plan="basic"
            )
            
            # Check if the output file exists
            if os.path.exists(test_output_path):
                file_size = os.path.getsize(test_output_path)
                log_debug(f"Output file created successfully: {test_output_path} (Size: {file_size} bytes)")
                
                return jsonify({
                    "status": "success",
                    "message": "Captions added successfully",
                    "output_path": test_output_path,
                    "file_size": file_size,
                    "debug_log": debug_log_path
                })
            else:
                log_debug(f"Output file was not created: {test_output_path}")
                return jsonify({
                    "status": "error",
                    "error": "Output file was not created",
                    "debug_log": debug_log_path
                }), 500
            
        except Exception as e:
            log_debug(f"ERROR during add_captions_to_video: {str(e)}")
            log_debug(f"Error type: {type(e).__name__}")
            
            # Get the full exception info
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Format the traceback as a string
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in tb_lines:
                log_debug(line.rstrip())
            
            return jsonify({
                "status": "error", 
                "error": str(e),
                "error_type": type(e).__name__,
                "debug_log": debug_log_path
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/process', methods=['POST', 'OPTIONS'])
@log_route_details
def api_process():
    """API endpoint to process a video with captions"""
    from flask import jsonify, request
    import sys  # Import sys at the function level to ensure it's available
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response
    
    # Debug logging
    print("\n\n==== API PROCESS ENDPOINT CALLED ====")
    print(f"Request files: {list(request.files.keys())}")
    print(f"Request form: {list(request.form.keys())}")
    print("==== END OF API PROCESS ENDPOINT INFO ====\n\n")
    
    # Check if a file was uploaded - accept both 'video' and 'video_data' keys for backward compatibility
    if 'video' in request.files:
        file = request.files['video']
    elif 'video_data' in request.files:
        file = request.files['video_data']
    else:
        print("Error: Neither 'video' nor 'video_data' in request.files")
        return jsonify({'error': 'No video file provided'}), 400
    
    # Check if the file is valid
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    print(f"File name: {file.filename}")
    print(f"File type: {file.content_type if hasattr(file, 'content_type') else 'unknown'}")
    
    if not allowed_file(file.filename):
        print(f"Error: File type not allowed. Filename: {file.filename}")
        return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get the template ID, highlight color, profanity filter option, user plan, and feature options
    template_id = request.form.get('template', 'minimal_white')
    highlight_color = request.form.get('highlight_color', None)
    filter_profanity = request.form.get('filter_profanity', 'false').lower() == 'true'
    user_plan = request.form.get('user_plan', 'free').lower()  # Default to free plan if not specified
    enable_speaker_tracking = request.form.get('enable_speaker_tracking', 'false').lower() == 'true'
    
    # Speaker tracking is available for Basic and Pro plans
    if enable_speaker_tracking and user_plan not in ['pro', 'basic']:
        print("Speaker tracking is only available for Basic and Pro plan users")
        enable_speaker_tracking = False
    
    # Check if user has credits (backend validation)
    user_id = request.form.get('user_id', None)
    user_credits = request.form.get('user_credits', None)
    
    if user_id and user_credits is not None:
        try:
            # Convert credits to integer
            user_credits = int(user_credits)
            
            # Check if user has enough credits (10 required)
            if user_credits < 10:
                print(f"Error: User {user_id} has insufficient credits: {user_credits}")
                return jsonify({
                    'error': True,
                    'message': 'You need at least 10 credits to process a video. Please upgrade your plan.'
                }), 403  # 403 Forbidden
        except ValueError:
            print(f"Error: Invalid credit value: {user_credits}")
            # Continue processing even if credit check fails to avoid breaking existing functionality
            pass
    
    # Handle custom templates
    if template_id.startswith('custom_'):
        print(f"Processing custom template: {template_id}")
        
        # If highlight_color is not provided, try to extract it from the template ID
        if not highlight_color:
            # Format: custom_timestamp_randomId_colorHex_fontCode
            color_match = re.search(r'custom_\d+_[a-z0-9]+_([a-f0-9]{6})(?:_([a-z0-9]+))?', template_id, re.IGNORECASE)
            if color_match:
                color_hex = color_match.group(1)
                highlight_color = f"#{color_hex}"
                print(f"Extracted highlight color from template ID: {highlight_color}")
                
                # Extract font code if available
                if len(color_match.groups()) > 1 and color_match.group(2):
                    font_code = color_match.group(2)
                    print(f"Extracted font code from template ID: {font_code}")
                    
                    # Map font code to actual font file
                    # This is a simple mapping - in a real app, you'd have a more robust system
                    font_mapping = {
                        'montserrat': 'Montserrat.ttf',
                        'roboto': 'Roboto-Bold.ttf',
                        'opensans': 'OpenSans-Bold.ttf',
                        'lato': 'Lato-Bold.ttf',
                        'poppins': 'Poppins-Bold.ttf',
                        'oswald': 'Oswald-Bold.ttf',
                        'raleway': 'Raleway-Bold.ttf',
                        'sourcesans': 'SourceSansPro-Bold.ttf',
                    }
                    
                    # Try to find a matching font or use a default
                    for key, font_file in font_mapping.items():
                        if font_code.lower().startswith(key):
                            print(f"Mapped font code {font_code} to font file: {font_file}")
                            # Set the font for the template
                            # Note: You would need to modify your template system to use this
                            break
        
        # Use default color if still not set
        if not highlight_color:
            highlight_color = "#00BFFF"  # Default blue
            print(f"Using default highlight color: {highlight_color}")
            
        print(f"Final highlight color for custom template: {highlight_color}")
    
    # Save the file
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    base_filename = f"{unique_id}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_filename)
    file.save(file_path)
    
    # Check video duration based on user's plan
    if not check_video_duration(file_path, user_plan):
        # Get the maximum duration for the user's plan
        max_duration = MAX_VIDEO_DURATION.get(user_plan.lower(), DEFAULT_MAX_DURATION)
        
        # Format the duration message based on the duration
        if max_duration < 60:
            duration_msg = f"{max_duration} seconds"
        else:
            duration_msg = f"{max_duration // 60} minutes"
            
        os.remove(file_path)  # Clean up
        return jsonify({
            'error': True,
            'message': f'Video duration exceeds the maximum allowed limit of {duration_msg} for your {user_plan.capitalize()} plan.'
        }), 413  # 413 Payload Too Large
    
    try:
        print(f"Starting video processing with parameters:")
        print(f"  - File path: {file_path}")
        print(f"  - Template ID: {template_id}")
        print(f"  - Highlight color: {highlight_color}")
        print(f"  - Filter profanity: {filter_profanity}")
        print(f"  - User plan: {user_plan}")
        print(f"  - Enable speaker tracking: {enable_speaker_tracking}")
        
        # Create a debug log file for this specific API call
        debug_log_path = os.path.join(PROCESSED_FOLDER, f"api_debug_log_{uuid.uuid4().hex}.txt")
        
        def log_api_debug(message):
            """Write a debug message to the API log file and print it"""
            try:
                with open(debug_log_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    log_message = f"[{timestamp}] {message}\n"
                    f.write(log_message)
                print(message)
            except Exception as e:
                print(f"Error writing to API debug log: {str(e)}")
        
        log_api_debug(f"=== API PROCESS ENDPOINT DETAILED DEBUG ===")
        log_api_debug(f"Starting video processing with parameters:")
        log_api_debug(f"  - File path: {file_path}")
        log_api_debug(f"  - Template ID: {template_id}")
        log_api_debug(f"  - Highlight color: {highlight_color}")
        log_api_debug(f"  - Filter profanity: {filter_profanity}")
        log_api_debug(f"  - User plan: {user_plan}")
        log_api_debug(f"  - Enable speaker tracking: {enable_speaker_tracking}")
        
        # Process the video with detailed error handling
        try:
            log_api_debug(f"Calling process_video function")
            # Create options dictionary for process_video
            # Get caption customization options if available
            caption_vertical_position_val = request.form.get("caption_vertical_position", 50)
            caption_font_size_val = request.form.get("caption_font_size", 100)
            
            # Convert to integers
            try:
                caption_vertical_position_val = int(caption_vertical_position_val)
                caption_font_size_val = int(caption_font_size_val)
            except ValueError:
                caption_vertical_position_val = 50
                caption_font_size_val = 100
                
            log_api_debug(f"  - Caption vertical position: {caption_vertical_position_val}%")
            log_api_debug(f"  - Caption font size: {caption_font_size_val}%")
                
            options = {
                'template': template_id,
                'highlight_color': highlight_color,
                'censor_profanity': filter_profanity,
                'user_plan': user_plan,
                'include_speakers': enable_speaker_tracking,
                'caption_vertical_position': caption_vertical_position_val,
                'caption_font_size': caption_font_size_val
            }
            output_filename, original_video_filename = process_video(
                file_path, 
                options
            )
            log_api_debug(f"process_video function completed successfully")
            log_api_debug(f"Output filename: {output_filename}")
            log_api_debug(f"Original video filename: {original_video_filename}")
        except Exception as process_error:
            log_api_debug(f"\n\n==== CRITICAL ERROR IN PROCESS_VIDEO CALL ====")
            log_api_debug(f"ERROR: {str(process_error)}")
            log_api_debug(f"Error type: {type(process_error).__name__}")
            
            # Get the full exception info
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Format the traceback as a string
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in tb_lines:
                log_api_debug(line.rstrip())
            
            log_api_debug(f"==== END OF PROCESS_VIDEO ERROR REPORT ====\n\n")
            
            # Re-raise the exception to be caught by the outer try-except
            raise
        
        print(f"Video processing completed successfully")
        print(f"  - Output filename: {output_filename}")
        print(f"  - Original video filename: {original_video_filename}")
        
        # Get the transcription filename
        transcription_filename = output_filename.rsplit('.', 1)[0] + '.json'
        print(f"  - Transcription filename: {transcription_filename}")
        
        # Get the template name
        template_name = "Default"
        for t in get_templates()["templates"]:
            if t["id"] == template_id:
                template_name = t["name"]
                break
        
        print(f"  - Template name: {template_name}")
        
        # Generate response
        # --- R2 Uploading ---
        final_video_url = None
        final_transcription_url = None
        final_original_video_url = None

        if USE_R2_STORAGE and r2_client:
            print("Uploading final artifacts to R2...")
            try:
                # Upload processed video
                processed_video_path = os.path.join(PROCESSED_FOLDER, output_filename)
                if os.path.exists(processed_video_path):
                    r2_client.upload_file(processed_video_path, R2_VIDEO_BUCKET, output_filename)
                    final_video_url = f"{R2_PUBLIC_URL}/{output_filename}"
                    print(f"Successfully uploaded processed video to R2: {final_video_url}")
                    os.remove(processed_video_path) # Clean up local file

                # Upload transcription
                transcription_file_path = os.path.join(PROCESSED_FOLDER, transcription_filename)
                if os.path.exists(transcription_file_path):
                    r2_client.upload_file(transcription_file_path, R2_VIDEO_BUCKET, transcription_filename)
                    final_transcription_url = f"{R2_PUBLIC_URL}/{transcription_filename}"
                    print(f"Successfully uploaded transcription to R2: {final_transcription_url}")
                    os.remove(transcription_file_path)

                # Upload original (un-captioned) video
                original_video_path = os.path.join(PROCESSED_FOLDER, original_video_filename)
                if os.path.exists(original_video_path):
                    r2_client.upload_file(original_video_path, R2_VIDEO_BUCKET, original_video_filename)
                    final_original_video_url = f"{R2_PUBLIC_URL}/{original_video_filename}"
                    print(f"Successfully uploaded original video to R2: {final_original_video_url}")
                    os.remove(original_video_path)

            except ClientError as e:
                print(f"ERROR: Failed to upload to R2: {e}")
                # Fallback to local URLs if R2 upload fails
                final_video_url = url_for('get_processed_file', filename=output_filename, _external=True)
                final_transcription_url = url_for('get_transcription', filename=transcription_filename, _external=True)
                final_original_video_url = url_for('get_processed_file', filename=original_video_filename, _external=True)
        else:
            # Generate local URLs if not using R2
            # Use the local_processed route for better compatibility with production paths
            # URL encode filenames to handle spaces and special characters
            encoded_output_filename = urllib.parse.quote(output_filename, safe='')
            encoded_transcription_filename = urllib.parse.quote(transcription_filename, safe='')
            encoded_original_filename = urllib.parse.quote(original_video_filename, safe='')
            
            final_video_url = url_for('serve_processed_file', filename=encoded_output_filename, _external=True)
            final_transcription_url = url_for('get_transcription', filename=encoded_transcription_filename, _external=True)
            final_original_video_url = url_for('serve_processed_file', filename=encoded_original_filename, _external=True)
            
            # Debug: Log the generated URLs
            print(f"Generated URLs:")
            print(f"  final_video_url: {final_video_url}")
            print(f"  final_transcription_url: {final_transcription_url}")
            print(f"  final_original_video_url: {final_original_video_url}")
            
            # Verify the files exist at the expected paths
            processed_video_path = os.path.join(PROCESSED_FOLDER, output_filename)
            print(f"  Video file exists at {processed_video_path}: {os.path.exists(processed_video_path)}")
            if os.path.exists(processed_video_path):
                print(f"  Video file size: {os.path.getsize(processed_video_path)} bytes")


        response = {
            'success': True,
            'filename': output_filename,
            'transcription_filename': transcription_filename,
            'view_url': f'/view/{output_filename}', # This can remain local as it's a server-side render
            'download_url': final_video_url,
            'video_url': final_video_url,
            'transcription_url': final_transcription_url,
            'template_name': template_name,
            'template_id': template_id,
            'highlight_color': highlight_color,
            'speaker_tracking_enabled': enable_speaker_tracking,
            'original_video_url': final_original_video_url
        }
        
        # Clean up the original file
        # Add a small delay to ensure all file handles are closed
        import time
        time.sleep(0.5)  # 500ms delay
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up original file: {file_path}")
            except PermissionError:
                print(f"Warning: Could not remove file {file_path} - it may be in use")
        
        print(f"API process completed successfully, returning response")
        return jsonify(response)
    
    except Exception as e:
        # Enhanced error logging
        import traceback
        import sys
        
        # Create a dedicated error log file
        error_log_path = os.path.join(PROCESSED_FOLDER, f"error_log_{uuid.uuid4().hex}.txt")
        
        def log_error(message):
            """Write an error message to the error log file and print it"""
            try:
                with open(error_log_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    log_message = f"[{timestamp}] {message}\n"
                    f.write(log_message)
                print(message)
            except Exception as log_error:
                print(f"Error writing to error log: {str(log_error)}")
        
        log_error(f"\n\n==== CRITICAL ERROR IN API_PROCESS ====")
        log_error(f"ERROR in api_process: {str(e)}")
        log_error(f"Error type: {type(e).__name__}")
        log_error(f"Error details: {str(e)}")
        
        # Get the full exception info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Log the traceback
        log_error(f"\nDetailed traceback:")
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in tb_lines:
            log_error(line.rstrip())
        
        # Log the stack trace
        log_error(f"\nStack trace:")
        stack_lines = traceback.format_stack()
        for line in stack_lines:
            log_error(line.rstrip())
        
        # Skip local variables logging to prevent memory issues
        log_error(f"\nLocal variables logging skipped to prevent memory issues")
        
        # Log system information
        log_error(f"\nSystem information:")
        log_error(f"  Python version: {sys.version}")
        log_error(f"  Platform: {sys.platform}")
        log_error(f"  Upload folder: {UPLOAD_FOLDER}")
        log_error(f"  Processed folder: {PROCESSED_FOLDER}")
        log_error(f"  Upload folder exists: {os.path.exists(UPLOAD_FOLDER)}")
        log_error(f"  Processed folder exists: {os.path.exists(PROCESSED_FOLDER)}")
        
        log_error(f"==== END OF ERROR REPORT ====\n\n")
        log_error(f"Error log saved to: {error_log_path}")
        
        # Print the error log path for easy reference
        print(f"\n\nERROR LOG SAVED TO: {error_log_path}\n\n")
        
        # Clean up on error
        import time
        time.sleep(0.5)  # 500ms delay
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up original file after error: {file_path}")
            except PermissionError:
                print(f"Warning: Could not remove file {file_path} - it may be in use")
        
        # Return a more detailed error message
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__,
            'error_details': traceback.format_exc()
        }
        
        return jsonify(error_response), 500

@app.route('/upload', methods=['POST'])
@log_route_details
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Get the selected template (default to "minimal_white" if not specified)
        template = request.form.get('template', 'minimal_white')
        user_plan = request.form.get('user_plan', 'free').lower()  # Default to free plan if not specified
        
        # Generate unique filename
        original_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = str(uuid.uuid4()) + '.' + original_extension
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Check if video duration is within the allowed limit for the user's plan
        if not check_video_duration(file_path, user_plan):
            # Get the maximum duration for the user's plan
            max_duration = MAX_VIDEO_DURATION.get(user_plan.lower(), DEFAULT_MAX_DURATION)
            
            # Format the duration message based on the duration
            if max_duration < 60:
                duration_msg = f"{max_duration} seconds"
            else:
                duration_msg = f"{max_duration // 60} minutes"
                
            # Clean up the file
            os.remove(file_path)
            return jsonify({
                'error': True,
                'message': f"Video duration exceeds the maximum allowed limit of {duration_msg} for your {user_plan.capitalize()} plan."
            }), 413  # 413 Payload Too Large
        
        # Process the video
        processed_filename = f"processed_{str(uuid.uuid4())}.mp4"  # Always use MP4 for processed files
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        # WebM format is no longer supported
        if original_extension == 'webm':
            # Clean up the file
            os.remove(file_path)
            return jsonify({
                'error': True,
                'message': "WebM format is no longer supported. Please convert your video to MP4, AVI, MOV, MKV, WMV, or FLV format."
            }), 400  # 400 Bad Request
        
        # Step 1: Convert to 9:16 if needed
        # Use /app/tmp for temporary files
        temp_dir = "/app/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        converted_path = os.path.join(temp_dir, f"converted_{uuid.uuid4().hex}_{os.path.basename(file_path)}")
        try:
            print(f"🎬 STARTING FFMPEG CONVERSION: {file_path} -> {converted_path}")
            convert_to_9_16_ratio_ffmpeg(file_path, converted_path)
            print(f"✅ FFMPEG CONVERSION COMPLETED: {converted_path}")
        except Exception as e:
            print(f"❌ FFMPEG CONVERSION FAILED: {str(e)}")
            print("🔄 FALLING BACK TO MOVIEPY CONVERSION...")
            convert_to_9_16_ratio(file_path, converted_path)
            print(f"✅ MOVIEPY FALLBACK COMPLETED: {converted_path}")
        
        # Step 2: Transcribe the video
        transcription_result = transcribe_video(converted_path)
        segments = transcription_result["segments"]
        
        # Update the converted path if it was changed during profanity filtering
        if "video_path" in transcription_result and transcription_result["video_path"] != converted_path:
            converted_path = transcription_result["video_path"]
        
        # Step 3: Split into phrases with max 3 words
        phrases = split_into_phrases(segments, max_words=3)
        
        # Step 4: Add captions to the video with the selected template
        # Add watermark for free plan users
        add_watermark = user_plan.lower() == "free"
        # Use default caption position and font size if not specified
        caption_vertical_position_val = request.form.get("caption_vertical_position", 50)
        caption_font_size_val = request.form.get("caption_font_size", 100)
        
        # Convert to integers
        try:
            caption_vertical_position_val = int(caption_vertical_position_val)
            caption_font_size_val = int(caption_font_size_val)
        except ValueError:
            caption_vertical_position_val = 50
            caption_font_size_val = 100
        
        add_captions_to_video(
            converted_path, 
            processed_path, 
            phrases, 
            template=template, 
            add_watermark=add_watermark, 
            user_plan=user_plan,
            caption_vertical_position=caption_vertical_position_val,
            caption_font_size=caption_font_size_val
        )
        
        # Clean up temporary files
        os.remove(file_path)
        os.remove(converted_path)
        
        return redirect(url_for('download_file', filename=processed_filename))
    
    return redirect(request.url)

@app.route('/download/<path:filename>')
@log_route_details
def download_file(filename):
    try:
        # URL decode the filename to handle spaces and special characters
        decoded_filename = urllib.parse.unquote(filename)
        print(f"Received download request for: {decoded_filename}")
        
        # Check if the file exists in the PROCESSED_FOLDER
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)
        
        # If file doesn't exist, try with processed_ prefix
        if not os.path.exists(file_path) and not decoded_filename.startswith('processed_'):
            processed_filename = f"processed_{decoded_filename}"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            
            if os.path.exists(processed_path):
                decoded_filename = processed_filename
                print(f"Found file with processed_ prefix: {decoded_filename}")
                # Serve the file directly
                download = request.args.get('download', 'false').lower() == 'true'
                return send_from_directory(
                    app.config['PROCESSED_FOLDER'], 
                    decoded_filename, 
                    as_attachment=download
                )
        
        # If the direct download is requested via query param
        if request.args.get('download', 'false').lower() == 'true':
            return send_from_directory(
                app.config['PROCESSED_FOLDER'], 
                decoded_filename, 
                as_attachment=True
            )
        
        # Otherwise, render the download template
        return render_template('download.html', filename=decoded_filename)
    except Exception as e:
        print(f"Error in download_file: {str(e)}")
        return f"Error serving file: {str(e)}", 500
        
# This route is now deprecated in favor of /processed/<filename> but kept for compatibility
@app.route('/api/download', methods=['GET'])
@log_route_details
def api_download():
    filename = request.args.get('filename')
    if not filename:
        return "Missing filename parameter", 400
    # Redirect to the new, more robust endpoint
    return redirect(url_for('get_processed_file', filename=filename, download='true'))

@app.route('/processed/<path:filename>')
@log_route_details
def get_processed_file(filename):
    try:
        decoded_filename = urllib.parse.unquote(filename)
        print(f"Received processed file request for: {decoded_filename}")

        if USE_R2_STORAGE and r2_client:
            try:
                # Generate a presigned URL for the R2 object
                presigned_url = r2_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': R2_VIDEO_BUCKET, 'Key': decoded_filename},
                    ExpiresIn=3600  # URL expires in 1 hour
                )
                # Redirect the user to the presigned URL
                return redirect(presigned_url)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"File not found in R2 bucket: {decoded_filename}")
                    return "File not found", 404
                else:
                    print(f"Error generating presigned URL from R2: {e}")
                    return "Error accessing file", 500
        else:
            # Fallback to local storage
            download = request.args.get('download', 'false').lower() == 'true'
            if not os.path.exists(os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)):
                 return f"File not found: {decoded_filename}", 404
            return send_from_directory(app.config['PROCESSED_FOLDER'], decoded_filename, as_attachment=download)

    except Exception as e:
        print(f"Error serving processed file: {str(e)}")
        return f"Error serving file: {str(e)}", 500

@app.route('/local_processed/<path:filename>')
@log_route_details
def serve_processed_file(filename):
    """Serve files directly from the local_processed directory with proper MIME type"""
    try:
        # URL decode the filename to handle spaces and special characters
        decoded_filename = urllib.parse.unquote(filename)
        print(f"=== SERVE_PROCESSED_FILE REQUEST ===")
        print(f"Requested filename: {filename}")
        print(f"Decoded filename: {decoded_filename}")
        
        # Verify production paths are correct
        verify_production_paths()
        
        # Debug: Print current working directory and config
        print(f"Current working directory: {os.getcwd()}")
        print(f"PROCESSED_FOLDER config: {app.config['PROCESSED_FOLDER']}")
        
        # Use the absolute path from app config
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)
        print(f"Full file path: {file_path}")
        
        # Debug: List files in the processed directory
        try:
            files_in_dir = os.listdir(app.config['PROCESSED_FOLDER'])
            print(f"Files in processed directory: {files_in_dir}")
        except Exception as list_error:
            print(f"Error listing processed directory: {list_error}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            # Try alternative paths for debugging
            alt_path1 = os.path.join('/app/local_processed', decoded_filename)
            alt_path2 = os.path.join('local_processed', decoded_filename)
            print(f"Checking alternative path 1: {alt_path1} - exists: {os.path.exists(alt_path1)}")
            print(f"Checking alternative path 2: {alt_path2} - exists: {os.path.exists(alt_path2)}")
            return f"File not found: {decoded_filename}", 404
        
        # Get file size for debugging
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        # Auto-detect MIME type based on file extension
        mime_type, _ = mimetypes.guess_type(decoded_filename)
        if mime_type is None:
            # Default MIME type for unknown files
            mime_type = 'application/octet-stream'
        
        print(f"Serving file with MIME type: {mime_type}")
        
        # Use send_from_directory for better security and path handling
        return send_from_directory(app.config['PROCESSED_FOLDER'], decoded_filename, mimetype=mime_type)
    except Exception as e:
        print(f"Error serving local_processed file: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error serving file: {str(e)}", 500

@app.route('/view/<filename>')
@log_route_details
def view_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=False)

@app.route('/debug/video-access/<filename>')
@log_route_details
def debug_video_access(filename):
    """Debug endpoint to test video file access"""
    try:
        decoded_filename = urllib.parse.unquote(filename)
        print(f"=== DEBUG VIDEO ACCESS ===")
        print(f"Requested filename: {filename}")
        print(f"Decoded filename: {decoded_filename}")
        
        # Verify production paths
        verify_production_paths()
        
        # Check file existence
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)
        file_exists = os.path.exists(file_path)
        
        result = {
            'filename': decoded_filename,
            'file_path': file_path,
            'file_exists': file_exists,
            'processed_folder': app.config['PROCESSED_FOLDER'],
            'working_directory': os.getcwd()
        }
        
        if file_exists:
            result['file_size'] = os.path.getsize(file_path)
            result['file_permissions'] = oct(os.stat(file_path).st_mode)[-3:]
        
        # Also check alternative paths
        alt_paths = [
            os.path.join('/app/local_processed', decoded_filename),
            os.path.join('local_processed', decoded_filename),
            os.path.join('./local_processed', decoded_filename)
        ]
        
        result['alternative_paths'] = {}
        for alt_path in alt_paths:
            result['alternative_paths'][alt_path] = os.path.exists(alt_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/files')
@log_route_details
def debug_list_files():
    """Debug endpoint to list all files in the processed folder"""
    try:
        files = os.listdir(app.config['PROCESSED_FOLDER'])
        
        # Get file details
        file_details = []
        for filename in files:
            filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            stats = os.stat(filepath)
            file_details.append({
                'name': filename,
                'size': stats.st_size,
                'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'full_path': filepath,
                'url': url_for('serve_processed_file', filename=filename, _external=True)
            })
        
        # Sort by modification time (newest first)
        file_details.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'processed_folder': app.config['PROCESSED_FOLDER'],
            'file_count': len(files),
            'files': file_details,
            'working_directory': os.getcwd(),
            'alternative_paths': {
                '/app/local_processed': os.path.exists('/app/local_processed'),
                'local_processed': os.path.exists('local_processed'),
                './local_processed': os.path.exists('./local_processed')
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'processed_folder': app.config.get('PROCESSED_FOLDER', 'Not set'),
            'working_directory': os.getcwd()
        }), 500

@app.route('/debug/test-file/<filename>')
@log_route_details
def debug_test_file(filename):
    """Debug endpoint to test file access"""
    try:
        decoded_filename = urllib.parse.unquote(filename)
        
        # Test different possible paths
        paths_to_test = [
            os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename),
            os.path.join('/app/local_processed', decoded_filename),
            os.path.join('local_processed', decoded_filename),
            os.path.join('./local_processed', decoded_filename)
        ]
        
        results = {}
        for path in paths_to_test:
            results[path] = {
                'exists': os.path.exists(path),
                'is_file': os.path.isfile(path) if os.path.exists(path) else False,
                'size': os.path.getsize(path) if os.path.exists(path) else 0
            }
        
        return jsonify({
            'filename': decoded_filename,
            'paths_tested': results,
            'config_processed_folder': app.config['PROCESSED_FOLDER'],
            'working_directory': os.getcwd()
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/debug/cors-test', methods=['GET', 'POST', 'OPTIONS'])
@log_route_details
def debug_cors_test():
    """Debug endpoint to test CORS configuration"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'preflight_ok'})
    
    return jsonify({
        'method': request.method,
        'origin': request.headers.get('Origin'),
        'user_agent': request.headers.get('User-Agent'),
        'cors_status': 'working',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/captions/<path:filename>/srt')
@log_route_details
def get_srt_captions(filename):
    """Generate and download SRT captions for a processed video"""
    try:
        # URL decode the filename to handle spaces and special characters
        decoded_filename = urllib.parse.unquote(filename)
        print(f"Received SRT caption request for: {decoded_filename}")
        
        # List all files in the processed folder
        processed_files = os.listdir(app.config['PROCESSED_FOLDER'])
        
        # Try to find the exact file first
        video_path = os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)
        json_filename = decoded_filename.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
        
        # If JSON doesn't exist directly, try all our naming variants
        if not os.path.exists(json_path):
            # 1. Try with processed_ prefix
            if not decoded_filename.startswith('processed_'):
                processed_json = f"processed_{json_filename}"
                processed_json_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_json)
                
                if os.path.exists(processed_json_path):
                    json_filename = processed_json
                    json_path = processed_json_path
                    print(f"Found JSON with processed_ prefix: {json_filename}")
            
            # 2. Try with Quickcap output naming convention
            if not os.path.exists(json_path):
                # Look for the newest "Quickcap output XX.json" file
                output_files = [f for f in processed_files if f.startswith("Quickcap output ") and f.endswith(".json")]
                if output_files:
                    output_files.sort()  # Sort alphabetically to get the highest number
                    json_filename = output_files[-1]  # Get the last (highest numbered) file
                    json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
                    print(f"Found JSON with Quickcap output naming: {json_filename}")
        
        if not os.path.exists(json_path):
            print(f"Transcription data not found at: {json_path}")
            # Fall back to any json file in the folder if we can't find a specific match
            json_files = [f for f in processed_files if f.endswith('.json')]
            if json_files:
                json_filename = json_files[-1]  # Get the last file (likely the most recent)
                json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
                print(f"Falling back to most recent JSON file: {json_filename}")
            else:
                return "Transcription data not found", 404
    except Exception as e:
        print(f"Error processing SRT caption request: {str(e)}")
        return f"Error generating SRT captions: {str(e)}", 500
    
    # Load the transcription data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Generate SRT content
    srt_content = generate_srt_from_segments(data["segments"])
    
    # Save SRT file to local_processed directory
    base_filename = filename.rsplit(".", 1)[0]
    srt_filename = f"{base_filename}.srt"
    srt_path = os.path.join(app.config['PROCESSED_FOLDER'], srt_filename)
    
    try:
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(srt_content)
        print(f"Saved SRT file to: {srt_path}")
    except Exception as e:
        print(f"Warning: Could not save SRT file locally: {str(e)}")
    
    # Create a response with the SRT content
    response = Response(srt_content, mimetype='text/plain')
    response.headers['Content-Disposition'] = f'attachment; filename={base_filename}.srt'
    
    return response

@app.route('/captions/<path:filename>/ass')
@log_route_details
def get_ass_captions(filename):
    """Generate and download ASS captions for a processed video"""
    try:
        # URL decode the filename to handle spaces and special characters
        decoded_filename = urllib.parse.unquote(filename)
        print(f"Received ASS caption request for: {decoded_filename}")
        
        # List all files in the processed folder
        processed_files = os.listdir(app.config['PROCESSED_FOLDER'])
        
        # Try to find the exact file first
        video_path = os.path.join(app.config['PROCESSED_FOLDER'], decoded_filename)
        json_filename = decoded_filename.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
        
        # If JSON doesn't exist directly, try all our naming variants
        if not os.path.exists(json_path):
            # 1. Try with processed_ prefix
            if not decoded_filename.startswith('processed_'):
                processed_json = f"processed_{json_filename}"
                processed_json_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_json)
                
                if os.path.exists(processed_json_path):
                    json_filename = processed_json
                    json_path = processed_json_path
                    print(f"Found JSON with processed_ prefix: {json_filename}")
            
            # 2. Try with Quickcap output naming convention
            if not os.path.exists(json_path):
                # Look for the newest "Quickcap output XX.json" file
                output_files = [f for f in processed_files if f.startswith("Quickcap output ") and f.endswith(".json")]
                if output_files:
                    output_files.sort()  # Sort alphabetically to get the highest number
                    json_filename = output_files[-1]  # Get the last (highest numbered) file
                    json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
                    print(f"Found JSON with Quickcap output naming: {json_filename}")
        
        if not os.path.exists(json_path):
            print(f"Transcription data not found at: {json_path}")
            # Fall back to any json file in the folder if we can't find a specific match
            json_files = [f for f in processed_files if f.endswith('.json')]
            if json_files:
                json_filename = json_files[-1]  # Get the last file (likely the most recent)
                json_path = os.path.join(app.config['PROCESSED_FOLDER'], json_filename)
                print(f"Falling back to most recent JSON file: {json_filename}")
            else:
                return "Transcription data not found", 404
    except Exception as e:
        print(f"Error processing ASS caption request: {str(e)}")
        return f"Error generating ASS captions: {str(e)}", 500
    
    # Load the transcription data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get video dimensions if available
    video_width = 480  # Default width
    video_height = 854  # Default height (9:16 ratio)
    
    # Try to get actual video dimensions
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(video_path):
        try:
            with VideoFileClip(video_path) as clip:
                clip.fps = clip.fps or 24
                video_width, video_height = clip.size
        except Exception as e:
            print(f"Error getting video dimensions: {e}")
    
    # Generate ASS content
    ass_content = generate_ass_from_segments(data["segments"], video_width, video_height)
    
    # Save ASS file to local_processed directory
    base_filename = filename.rsplit(".", 1)[0]
    ass_filename = f"{base_filename}.ass"
    ass_path = os.path.join(app.config['PROCESSED_FOLDER'], ass_filename)
    
    try:
        with open(ass_path, 'w', encoding='utf-8') as ass_file:
            ass_file.write(ass_content)
        print(f"Saved ASS file to: {ass_path}")
    except Exception as e:
        print(f"Warning: Could not save ASS file locally: {str(e)}")
    
    # Create a response with the ASS content
    response = Response(ass_content, mimetype='text/plain')
    response.headers['Content-Disposition'] = f'attachment; filename={base_filename}.ass'
    
    return response

@app.route('/api/templates', methods=['GET'])
@log_route_details
def get_templates():
    """Return a list of all available templates with their details"""
    templates = [
        # Word by Word Templates
        {
            "id": "yellow_impact",
            "name": "Yellow Impact",
            "category": "Word by Word",
            "color": "#FFFF00",
            "description": "EuropaGroteskSH-Bol font with bright yellow text and thick black stroke, displaying one word at a time",
            "preview_url": "/static/previews/yellow_impact.jpg"
        },
        {
            "id": "bold_white",
            "name": "Bold White",
            "category": "Word by Word",
            "color": "#FFFFFF",
            "description": "Poppins-BlackItalic font with white text and 2px black stroke, displaying one word at a time",
            "preview_url": "/static/previews/bold_white.jpg"
        },

        # Neon Heartbeat (New Premium Template)
        {
            "id": "neon_heartbeat",
            "name": "Neon Heartbeat",
            "category": "Premium",
            "color": "#FF00FF",
            "description": "Montserrat Semi-Bold Italic text with subtle pulsing neon pink glow and enhanced light effect",
            "preview_url": "/static/previews/neon_heartbeat.jpg"
        },



        # Premium Yellow (New Premium Template)
        {
            "id": "premium_yellow",
            "name": "Premium Yellow",
            "category": "Premium",
            "color": "#E9D023",
            "description": "UPPERCASE Poppins Bold Italic text with bright yellow text and 3px black outline",
            "preview_url": "/static/previews/premium_yellow.jpg"
        },
        # Premium Orange (New Premium Template)
        {
            "id": "premium_orange",
            "name": "Premium Orange",
            "category": "Premium",
            "color": "#EB5B00",
            "description": "UPPERCASE Poppins Bold Italic text with white text, orange highlights and 3px black outline",
            "preview_url": "/static/previews/premium_orange.jpg"
        },
        # Bold Sunshine (New Template)
        {
            "id": "bold_sunshine",
            "name": "Bold Sunshine",
            "category": "Entertainment/Vlogger",
            "color": "#FFFF00",
            "description": "White text with yellow highlights and 2px black outline",
            "preview_url": "/static/previews/bold_sunshine.jpg"
        },
        # Creator Highlight (New Template)
        {
            "id": "creator_highlight",
            "name": "Creator Highlight",
            "category": "Entertainment/Vlogger",
            "color": "#FFCC00",
            "description": "Poppins SemiBold text with yellow highlight background and black text",
            "preview_url": "/static/previews/creator_highlight.jpg"
        },
        # Vlog Caption (New Template)
        {
            "id": "vlog_caption",
            "name": "Vlog Caption",
            "category": "Entertainment/Vlogger",
            "color": "#FFFFFF",
            "description": "Montserrat Bold text with white text and red accent bar",
            "preview_url": "/static/previews/vlog_caption.jpg"
        },
        # Classic/Minimal
        {
            "id": "minimalist_sans",
            "name": "Minimalist Sans",
            "category": "Classic/Minimal",
            "color": "#F0F0F0",
            "description": "Helvetica Neue Light text with clean white text and subtle blue accents",
            "preview_url": "/static/previews/minimalist_sans.jpg"
        },

        {
            "id": "minimal_white",
            "name": "Minimal White",
            "category": "Classic/Minimal",
            "color": "#ffffff",
            "description": "Clean white text with minimal styling",
            "preview_url": "/static/previews/minimal_white.jpg"
        },
        {
            "id": "elegant_pink",
            "name": "Elegant Pink",
            "category": "Classic/Minimal",
            "color": "#F7374F",
            "description": "White text with bright pink highlights and shadow",
            "preview_url": "/static/previews/elegant_pink.jpg"
        },
        
        # Entertainment/Vlogger
        {
            "id": "mrbeast",
            "name": "MrBeast",
            "category": "Entertainment/Vlogger",
            "color": "#ffdd00",
            "description": "Bold yellow text in MrBeast style",
            "preview_url": "/static/previews/mrbeast.jpg"
        },
        {
            "id": "vlogger_bold",
            "name": "Vlogger Bold",
            "category": "Entertainment/Vlogger",
            "color": "#525FE1",
            "description": "Bold text with blue highlights and black outline",
            "preview_url": "/static/previews/vlogger_bold.jpg"
        },
        {
            "id": "creator_clean",
            "name": "Creator Clean",
            "category": "Entertainment/Vlogger",
            "color": "#00BFFF",
            "description": "Clean text with blue highlights",
            "preview_url": "/static/previews/creator_clean.jpg"
        },
        {
            "id": "reaction_pop",
            "name": "Reaction Pop",
            "category": "Entertainment/Vlogger",
            "color": "#FF3C3C",
            "description": "Proxima Nova bold text with red accents",
            "preview_url": "/static/previews/reaction_pop.jpg"
        },
        
        # Social Media
        {
            "id": "tiktok_trend",
            "name": "TikTok Trend",
            "category": "Social Media",
            "color": "#FFFFFF",
            "description": "Inter Black bold white text with moving emoji background",
            "preview_url": "/static/previews/tiktok_trend.jpg"
        },
        {
            "id": "reels_ready",
            "name": "Reels Ready",
            "category": "Social Media",
            "color": "#FFFFFF",
            "description": "SF Pro Display Bold white text with gradient background bar",
            "preview_url": "/static/previews/reels_ready.jpg"
        },
        {
            "id": "tiktok",
            "name": "TikTok",
            "category": "Social Media",
            "color": "#ff69b4",
            "description": "Pink highlight bars in TikTok style",
            "preview_url": "/static/previews/tiktok.jpg"
        },

        {
            "id": "insta_story",
            "name": "Insta Story",
            "category": "Social Media",
            "color": "#F9CEEE",
            "description": "Light pink gradient text",
            "preview_url": "/static/previews/insta_story.jpg"
        },
        {
            "id": "blue_highlight",
            "name": "Blue Highlight",
            "category": "Social Media",
            "color": "#3D90D7",
            "description": "Poppins ExtraBold uppercase text with blue highlight bars and 1px black stroke",
            "preview_url": "/static/previews/blue_highlight.jpg"
        },

        
        # Educational/Informative
        {
            "id": "explainer_pro",
            "name": "Explainer Pro",
            "category": "Educational/Informative",
            "color": "#FF8C00",
            "description": "Orange highlight bars with white text",
            "preview_url": "/static/previews/explainer_pro.jpg"
        },
        {
            "id": "science_journal",
            "name": "Science Journal",
            "category": "Educational/Informative",
            "color": "#4B0082",
            "description": "Geist-Black font with highlighted terms",
            "preview_url": "/static/previews/science_journal.jpg"
        },
        
        # Gaming
        {
            "id": "streamer_pro",
            "name": "Streamer Pro",
            "category": "Gaming",
            "color": "#00FF80",
            "description": "Rajdhani Bold neon green text with tech-inspired background",
            "preview_url": "/static/previews/streamer_pro.jpg"
        },
        {
            "id": "esports_caption",
            "name": "Esports Caption",
            "category": "Gaming",
            "color": "#FF4500",
            "description": "UPPERCASE Russo One bold angular text with team color accents",
            "preview_url": "/static/previews/esports_caption.jpg"
        },
        {
            "id": "gaming_neon",
            "name": "Gaming Neon",
            "category": "Gaming",
            "color": "#007BFF",
            "description": "Lowercase multi-color neon text with vibrant glow effect",
            "preview_url": "/static/previews/gaming_neon.jpg"
        },
        
        # Cinematic/Film
        {
            "id": "film_noir",
            "name": "Film Noir",
            "category": "Cinematic/Film",
            "color": "#C0C0C0",
            "description": "Black text with thick white outline using Anton font",
            "preview_url": "/static/previews/film_noir.jpg"
        },
        {
            "id": "cinematic_quote",
            "name": "Cinematic Quote",
            "category": "Cinematic/Film",
            "color": "#E52020",
            "description": "Large elegant text with bright red highlights",
            "preview_url": "/static/previews/cinematic_quote.jpg"
        },
        {
            "id": "cinematic_futura",
            "name": "Cinematic Futura",
            "category": "Cinematic/Film",
            "color": "#F7A165",
            "description": "Futura PT Bold Oblique with orange text",
            "preview_url": "/static/previews/cinematic_futura.jpg"
        },
        
        # Comedy/Memes
        {
            "id": "meme_orange",
            "name": "Meme Orange",
            "category": "Comedy/Memes",
            "color": "#FF8C00",
            "description": "Orange text with black outline",
            "preview_url": "/static/previews/meme_orange.jpg"
        },
        
        # Trendy/Viral
        {
            "id": "meme_maker",
            "name": "Meme Maker",
            "category": "Trendy/Viral",
            "color": "#FFFFFF",
            "description": "UPPERCASE Impact text with classic white text and black outline",
            "preview_url": "/static/previews/meme_maker.jpg"
        },
        {
            "id": "viral_pop",
            "name": "Viral Pop",
            "category": "Trendy/Viral",
            "color": "#FF3399",
            "description": "UPPERCASE Gotham Black text with bright colored text and emoji accents",
            "preview_url": "/static/previews/viral_pop.jpg"
        },

        {
            "id": "green_bold",
            "name": "Aurora Bold",
            "category": "Trendy/Viral",
            "color": "#ec4899",
            "description": "Pink keywords with shadow effect",
            "preview_url": "/static/previews/green_bold.jpg"
        },
        {
            "id": "trendy_gradient",
            "name": "Trendy Gradient",
            "category": "Trendy/Viral",
            "color": "#FF69B4",
            "description": "Pink to blue gradient with Luckiest Guy font",
            "preview_url": "/static/previews/trendy_gradient.jpg"
        },
        
        # Missing templates from frontend
        {
            "id": "bold_switch",
            "name": "Bold Switch",
            "category": "Entertainment/Vlogger",
            "color": "#16D9E4",
            "description": "White text with alternating highlight colors and black stroke",
            "preview_url": "/static/previews/bold_switch.jpg"
        },
        {
            "id": "blue_highlight",
            "name": "Blue Highlight",
            "category": "Social Media",
            "color": "#3D90D7",
            "description": "Blue highlight bars with white text",
            "preview_url": "/static/previews/blue_highlight.jpg"
        },
        {
            "id": "insta_story",
            "name": "Insta Story",
            "category": "Social Media",
            "color": "#F9CEEE",
            "description": "Pink highlight style for Instagram stories",
            "preview_url": "/static/previews/insta_story.jpg"
        },
        {
            "id": "viral_pop",
            "name": "Viral Pop",
            "category": "Trendy/Viral",
            "color": "#FF3399",
            "description": "UPPERCASE Gotham Black text with bright colored text and emoji accents",
            "preview_url": "/static/previews/viral_pop.jpg"
        },

    ]
    
    return {"templates": templates}

# This route was removed to fix the duplicate route issue
# The functionality is now handled by the api_process function at line 1307

@app.route('/view/<filename>')
@log_route_details
def view_processed_video(filename):
    """View a processed video with a simple player"""
    return render_template('view.html', filename=filename)

@app.route('/static/<path:filename>')
@log_route_details
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/reprocess', methods=['POST', 'OPTIONS'])
@log_route_details
def reprocess_video():
    """API endpoint to reprocess a video with new highlight colors without re-transcribing"""
    from flask import jsonify, request
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        # Get parameters
        template_id = request.form.get('template', 'minimal_white')
        highlight_color = request.form.get('highlight_color', None)
        video_path = request.form.get('video_path', None)
        transcription_path = request.form.get('transcription_path', None)
        user_plan = request.form.get('user_plan', 'free').lower()  # Default to free plan if not specified
        
        # Check if user has credits (backend validation)
        user_id = request.form.get('user_id', None)
        user_credits = request.form.get('user_credits', None)
        
        if user_id and user_credits is not None:
            try:
                # Convert credits to integer
                user_credits = int(user_credits)
                
                # Check if user has enough credits (10 required)
                if user_credits < 10:
                    print(f"Error: User {user_id} has insufficient credits: {user_credits}")
                    return jsonify({
                        'error': True,
                        'message': 'You need at least 10 credits to process a video. Please upgrade your plan.'
                    }), 403  # 403 Forbidden
            except ValueError:
                print(f"Error: Invalid credit value: {user_credits}")
                # Continue processing even if credit check fails to avoid breaking existing functionality
                pass
        
        print(f"Reprocessing video with parameters: template={template_id}, highlight_color={highlight_color}")
        print(f"Video path: {video_path}")
        print(f"Transcription path: {transcription_path}")
        
        # Check if a video file was uploaded directly
        if 'video' in request.files:
            print("Video file was uploaded directly")
            file = request.files['video']
            
            # Save the uploaded file to a temporary location in /app/tmp
            temp_dir = "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_upload_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}_{secure_filename(file.filename)}")
            file.save(temp_upload_path)
            
            # Use this as the video path
            video_path = temp_upload_path
            print(f"Saved uploaded video to: {video_path}")
        
        if not video_path or not transcription_path:
            return jsonify({'error': 'Missing video_path or transcription_path'}), 400
        
        # Extract the filename from the video_path
        video_filename = os.path.basename(video_path)
        
        # Generate a unique output filename
        output_filename = f"processed_{uuid.uuid4().hex}_{video_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Get the original video file
        video_filename = os.path.basename(video_path)
        original_video_path = os.path.join(PROCESSED_FOLDER, video_filename)
        
        # Handle different URL formats
        print(f"Checking original video path: {original_video_path}")
        print(f"Path exists: {os.path.exists(original_video_path)}")
        
        if not os.path.exists(original_video_path):
            # Try with a direct path
            print(f"Trying direct path: {video_path}")
            if os.path.exists(video_path):
                original_video_path = video_path
                print(f"Using direct path: {original_video_path}")
            else:
                # Try with a relative path
                relative_path = video_path.replace('/processed/', '')
                alternative_path = os.path.join(PROCESSED_FOLDER, relative_path)
                print(f"Trying alternative path: {alternative_path}")
                if os.path.exists(alternative_path):
                    original_video_path = alternative_path
                    print(f"Using alternative path: {original_video_path}")
                else:
                    print(f"Warning: Could not find video file at any expected location")
        
        # Create a temporary copy of the original video without captions in /app/tmp
        temp_dir = "/app/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex}_{video_filename}")
        
        # Use the original video directly if it exists, otherwise extract from the processed video
        try:
            if os.path.exists(original_video_path):
                print(f"Using existing original video: {original_video_path}")
                
                # For basic and pro plans, we need to ensure the original video is at the right resolution
                if user_plan.lower() in ["basic", "pro"]:
                    # Load the video
                    original_clip = VideoFileClip(original_video_path)
                    original_clip.fps = original_clip.fps or 24
                    original_width, original_height = original_clip.size
                    
                    # Check if we need to resize based on current dimensions
                    needs_resize = False
                    
                    if user_plan.lower() == "basic" and (original_height < 1920 and original_width < 1920):
                        needs_resize = True
                    elif user_plan.lower() == "pro" and (original_height < 3840 and original_width < 3840):
                        # Pro plan uses 4K as maximum resolution for stability
                        needs_resize = True
                    
                    if needs_resize:
                        # Set resolution based on user plan
                        if user_plan.lower() == "basic":
                            # 4K resolution for basic plan
                            aspect_ratio = original_width / original_height
                            if aspect_ratio < 1:  # Vertical video
                                new_height = 3840
                                new_width = int(new_height * aspect_ratio)
                            else:  # Horizontal video
                                new_width = 3840
                                new_height = int(new_width / aspect_ratio)
                                
                            # Ensure dimensions are even
                            new_width, new_height = ensure_even_dimensions(new_width, new_height)
                            
                            # Resize to 4K
                            print(f"Resizing original video from {original_width}x{original_height} to 4K: {new_width}x{new_height}")
                            original_clip = original_clip.resize((new_width, new_height))
                            original_clip = original_clip.set_fps(original_clip.fps)
                            
                        elif user_plan.lower() == "pro":
                            try:
                                # Check if the original resolution is already high
                                if original_width > 3840 or original_height > 3840:
                                    # If already high resolution, just use the original size
                                    print(f"Video already has high resolution ({original_width}x{original_height}), keeping original dimensions")
                                    new_width, new_height = original_width, original_height
                                else:
                                    # 4K resolution is more stable than 8K
                                    # Using 4K as the maximum resolution to avoid memory issues
                                    aspect_ratio = original_width / original_height
                                    if aspect_ratio < 1:  # Vertical video
                                        new_height = 3840
                                        new_width = int(new_height * aspect_ratio)
                                    else:  # Horizontal video
                                        new_width = 3840
                                        new_height = int(new_width / aspect_ratio)
                                    
                                    # Ensure dimensions are even
                                    new_width, new_height = ensure_even_dimensions(new_width, new_height)
                                    
                                    # Resize to 4K (safer than 8K)
                                    print(f"Resizing original video from {original_width}x{original_height} to 4K: {new_width}x{new_height}")
                                    original_clip = original_clip.resize((new_width, new_height))
                                    original_clip = original_clip.set_fps(original_clip.fps)
                            except Exception as e:
                                print(f"Error during video resizing: {str(e)}")
                                print("Falling back to original video dimensions")
                                # If resizing fails, use the original dimensions
                                new_width, new_height = original_width, original_height
                        
                        # Set video quality based on user plan
                        video_bitrate = "8000k" if user_plan.lower() == "basic" else "12000k"
                        audio_bitrate = "192k" if user_plan.lower() == "basic" else "256k"
                        preset = "medium" if user_plan.lower() == "basic" else "slow"
                        
                        # Write the resized video directly to the temp path
                        try:
                            original_clip.write_videofile(
                                temp_video_path,
                                codec="libx264",
                                audio_codec="aac",
                                audio_bitrate=audio_bitrate,
                                bitrate=video_bitrate,
                                preset=preset,
                                temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                                remove_temp=True,
                                ffmpeg_params=[
                                    "-pix_fmt", "yuv420p",
                                    "-profile:v", "high",
                                    "-level", "4.0",
                                    "-movflags", "+faststart"
                                ]
                            )
                        finally:
                            original_clip.close()
                    else:
                        # If already at the right resolution, just copy
                        shutil.copy2(original_video_path, temp_video_path)
                else:
                    # For free plan, resize to 720p
                    original_clip = VideoFileClip(original_video_path)
                    original_clip.fps = original_clip.fps or 24
                    original_width, original_height = original_clip.size
                    
                    # Check if we need to resize
                    if original_width > 1280 or original_height > 1280:
                        # 720p resolution for free plan
                        aspect_ratio = original_width / original_height
                        if aspect_ratio < 1:  # Vertical video
                            new_height = 1280
                            new_width = int(new_height * aspect_ratio)
                        else:  # Horizontal video
                            new_width = 1280
                            new_height = int(new_width / aspect_ratio)
                            
                        # Ensure dimensions are even
                        new_width, new_height = ensure_even_dimensions(new_width, new_height)
                        
                        try:
                            # Resize to 720p
                            print(f"Resizing video from {original_width}x{original_height} to 720p: {new_width}x{new_height}")
                            resized_clip = original_clip.resize((new_width, new_height))
                            resized_clip = resized_clip.set_fps(original_clip.fps)
                            
                            # Write with basic settings
                            resized_clip.write_videofile(
                                temp_video_path,
                                codec="libx264",
                                audio_codec="aac",
                                audio_bitrate="128k",
                                bitrate="2000k",
                                preset="fast",
                                temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                                remove_temp=True,
                                ffmpeg_params=[
                                    "-pix_fmt", "yuv420p",
                                    "-profile:v", "main",
                                    "-level", "4.0",
                                    "-movflags", "+faststart"
                                ],
                                threads=2
                            )
                            resized_clip.close()
                            original_clip.close()
                        except Exception as e:
                            print(f"Error resizing free plan video: {str(e)}")
                            # If resizing fails, fall back to copying the original
                            shutil.copy2(original_video_path, temp_video_path)
                            try:
                                original_clip.close()
                            except:
                                pass
                    else:
                        # If already at or below 720p resolution, just copy the file
                        print(f"Video already at suitable resolution for free plan: {original_width}x{original_height}")
                        shutil.copy2(original_video_path, temp_video_path)
                        try:
                            original_clip.close()
                        except:
                            pass
            else:
                print(f"Original video not found, extracting from processed video: {original_video_path}")
                # Try to get the original video by extracting just the video stream without captions
                original_clip = VideoFileClip(original_video_path)
                original_clip.fps = original_clip.fps or 24
                
                # Get original dimensions
                original_width, original_height = original_clip.size
                
                # Set resolution based on user plan
                if user_plan.lower() == "basic":
                    # 1080p resolution for basic plan (1920 x 1080 for horizontal video)
                    # Maintain aspect ratio
                    aspect_ratio = original_width / original_height
                    if aspect_ratio < 1:  # Vertical video
                        new_height = 1920
                        new_width = int(new_height * aspect_ratio)
                    else:  # Horizontal video
                        new_width = 1920
                        new_height = int(new_width / aspect_ratio)
                    
                    # Ensure dimensions are even
                    new_width, new_height = ensure_even_dimensions(new_width, new_height)
                    
                    # Resize to 1080p
                    print(f"Resizing video from {original_width}x{original_height} to 1080p: {new_width}x{new_height}")
                    original_clip = original_clip.resize((new_width, new_height))
                    
                elif user_plan.lower() == "pro":
                    try:
                        # Check if the original resolution is already high
                        if original_width > 3840 or original_height > 3840:
                            # If already high resolution, just use the original size
                            print(f"Video already has high resolution ({original_width}x{original_height}), keeping original dimensions")
                            new_width, new_height = original_width, original_height
                        else:
                            # 4K resolution is more stable than 8K
                            # Using 4K as the maximum resolution to avoid memory issues
                            aspect_ratio = original_width / original_height
                            if aspect_ratio < 1:  # Vertical video
                                new_height = 3840
                                new_width = int(new_height * aspect_ratio)
                            else:  # Horizontal video
                                new_width = 3840
                                new_height = int(new_width / aspect_ratio)
                            
                            # Ensure dimensions are even
                            new_width, new_height = ensure_even_dimensions(new_width, new_height)
                            
                            # Resize to 4K (safer than 8K)
                            print(f"Resizing video from {original_width}x{original_height} to 4K: {new_width}x{new_height}")
                            original_clip = original_clip.resize((new_width, new_height))
                    except Exception as e:
                        print(f"Error during video resizing: {str(e)}")
                        print("Falling back to original video dimensions")
                        # If resizing fails, use the original dimensions
                        new_width, new_height = original_width, original_height
                    
                else:
                    # For free plan, ensure dimensions are even but keep original resolution
                    width, height = ensure_even_dimensions(original_width, original_height)
                    if width != original_width or height != original_height:
                        original_clip = original_clip.resize((width, height))
                
                # Set video quality based on user plan
                video_bitrate = "1500k"  # Default for free plan
                audio_bitrate = "128k"   # Default audio bitrate
                preset = "veryfast"      # Default preset
                
                # Higher quality for paid plans
                if user_plan.lower() == "basic":
                    # 4K quality for basic plan
                    video_bitrate = "8000k"  # Higher bitrate for 4K
                    audio_bitrate = "192k"   # Better audio quality
                    preset = "medium"        # Better quality preset
                elif user_plan.lower() == "pro":
                    # Highest quality for pro plan
                    video_bitrate = "12000k" # Highest bitrate
                    audio_bitrate = "256k"   # Best audio quality
                    preset = "slow"          # Best quality preset
                
                try:
                    original_clip.write_videofile(
                        temp_video_path, 
                        codec="libx264", 
                        audio_codec="aac",
                        audio_bitrate=audio_bitrate,
                        bitrate=video_bitrate,
                        preset=preset,
                        temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
                        remove_temp=True,
                        ffmpeg_params=[
                            "-pix_fmt", "yuv420p",
                            "-profile:v", "high",
                            "-level", "4.0",
                            "-movflags", "+faststart"
                        ]
                    )
                finally:
                    # Make sure clip is closed even if an error occurs
                    try:
                        original_clip.close()
                    except:
                        pass
        except Exception as e:
            return jsonify({'error': f'Failed to extract original video: {str(e)}'}), 500
        
        # Get the transcription data
        transcription_filename = os.path.basename(transcription_path)
        transcription_file_path = os.path.join(PROCESSED_FOLDER, transcription_filename)
        
        # Handle different URL formats for transcription
        if not os.path.exists(transcription_file_path):
            # Try with a direct path
            if os.path.exists(transcription_path):
                transcription_file_path = transcription_path
            else:
                # Try different relative path formats
                if '/api/transcription/' in transcription_path:
                    relative_path = transcription_path.replace('/api/transcription/', '')
                    transcription_file_path = os.path.join(PROCESSED_FOLDER, relative_path)
                else:
                    # Try extracting just the filename
                    transcription_file_path = os.path.join(PROCESSED_FOLDER, transcription_filename)
        
        try:
            with open(transcription_file_path, 'r') as f:
                transcription_data = json.load(f)
                segments = transcription_data.get('segments', [])
        except Exception as e:
            return jsonify({'error': f'Failed to load transcription data: {str(e)}'}), 500
        
        # Split into phrases
        phrases = split_into_phrases(segments, max_words=3)
        
        # Create a new transcription file for the reprocessed video
        new_transcription_filename = output_filename.rsplit('.', 1)[0] + '.json'
        new_transcription_path = os.path.join(PROCESSED_FOLDER, new_transcription_filename)
        
        with open(new_transcription_path, 'w') as f:
            json.dump({
                "segments": segments,
                "filename": output_filename
            }, f, indent=2)
        
        # Add captions to the video with the selected template and highlight color
        # Add watermark for free plan users
        add_watermark = user_plan.lower() == "free"
        # Get caption customization options if available
        caption_vertical_position_val = request.form.get("caption_vertical_position", 50)
        caption_font_size_val = request.form.get("caption_font_size", 100)
        
        # Convert to integers
        try:
            caption_vertical_position_val = int(caption_vertical_position_val)
            caption_font_size_val = int(caption_font_size_val)
        except ValueError:
            caption_vertical_position_val = 50
            caption_font_size_val = 100
            
        add_captions_to_video(
            temp_video_path, 
            output_path, 
            phrases, 
            template=template_id, 
            highlight_color=highlight_color, 
            add_watermark=add_watermark, 
            user_plan=user_plan,
            caption_vertical_position=caption_vertical_position_val,
            caption_font_size=caption_font_size_val
        )
        
        # Clean up temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        # Clean up uploaded file if it exists and starts with "upload_"
        if 'video' in request.files and video_path and os.path.exists(video_path) and os.path.basename(video_path).startswith("upload_"):
            os.remove(video_path)
            print(f"Removed temporary uploaded file: {video_path}")
        
        # Get the template name
        template_name = "Default"
        for t in get_templates()["templates"]:
            if t["id"] == template_id:
                template_name = t["name"]
                break
        
        # Generate response
        response = {
            'success': True,
            'filename': output_filename,
            'transcription_filename': new_transcription_filename,
            'view_url': f'/view/{output_filename}',
            'download_url': f'/processed/{output_filename}',
            'video_url': f'/processed/{output_filename}',
            'transcription_url': f'/api/transcription/{new_transcription_filename}',
            'template_name': template_name,
            'template_id': template_id,
            'highlight_color': highlight_color,
            'original_video_url': video_path
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_to_srt(segments):
    """Convert transcription segments to SRT format"""
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        # Format times as HH:MM:SS,mmm
        start_formatted = format_srt_time(start_time)
        end_formatted = format_srt_time(end_time)
        
        srt_content += f"{i+1}\n{start_formatted} --> {end_formatted}\n{text}\n\n"
    
    return srt_content

def format_srt_time(seconds):
    """Format time in seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def convert_to_ass(segments, video_width=270, video_height=480):
    """Convert transcription segments to ASS format"""
    # ASS header
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(width=video_width, height=video_height)

    # Add dialogue lines
    for segment in segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        # Format times as H:MM:SS.cc
        start_formatted = format_ass_time(start_time)
        end_formatted = format_ass_time(end_time)
        
        # Escape commas in text for ASS format
        text = text.replace(',', '\\,')
        
        ass_content += f"Dialogue: 0,{start_formatted},{end_formatted},Default,,0,0,0,,{text}\n"
    
    return ass_content

def format_ass_time(seconds):
    """Format time in seconds to ASS time format (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}"

@app.route('/api/transcription/<filename>')
@log_route_details
def get_transcription(filename):
    """API endpoint to get the transcription data for a processed video"""
    try:
        # Check if this is a download request
        download = request.args.get('download', 'false').lower() == 'true'
        format_type = request.args.get('format', 'json').lower()
        
        # Check if the file exists
        transcription_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(transcription_path):
            return jsonify({"error": "Transcription file not found"}), 404
        
        # Read the transcription data
        with open(transcription_path, 'r') as f:
            transcription_data = json.load(f)
        
        segments = transcription_data.get("segments", [])
        
        # Handle different format requests
        if format_type == 'srt':
            srt_content = convert_to_srt(segments)
            response = Response(srt_content, mimetype='text/plain')
            if download:
                response.headers["Content-Disposition"] = f"attachment; filename={filename.rsplit('.', 1)[0]}.srt"
            return response
        
        elif format_type == 'ass':
            ass_content = convert_to_ass(segments)
            response = Response(ass_content, mimetype='text/plain')
            if download:
                response.headers["Content-Disposition"] = f"attachment; filename={filename.rsplit('.', 1)[0]}.ass"
            return response
            
        elif format_type == 'captions':
            # Return captions in a format suitable for the timeline editor
            captions = []
            
            # Check if this is an edited file with captions already in the right format
            if 'captions' in transcription_data and transcription_data.get('edited', False):
                captions = transcription_data['captions']
            else:
                # Convert segments to caption format
                for i, segment in enumerate(segments):
                    caption = {
                        'id': f'caption-{i}',
                        'text': segment['text'],
                        'start': segment['start'],
                        'end': segment['end']
                    }
                    captions.append(caption)
            
            return jsonify({'captions': captions})
        
        # If it's a JSON download request, send the file directly
        elif download and format_type == 'json':
            return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
        
        # Otherwise, return as JSON
        return jsonify(transcription_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fonts', methods=['GET', 'OPTIONS'])
@log_route_details
def get_fonts():
    """Get a list of available fonts"""
    try:
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fonts')
        font_files = []
        
        # Get all font files in the Fonts directory
        for file in os.listdir(fonts_dir):
            if file.lower().endswith(('.ttf', '.otf')):
                font_files.append(file)
        
        return jsonify({
            'success': True,
            'fonts': font_files
        })
    except Exception as e:
        app.logger.error(f"Error getting fonts: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error getting fonts: {str(e)}"
        }), 500

@app.route('/api/templates/custom', methods=['GET', 'POST', 'OPTIONS'])
@log_route_details
def custom_templates():
    """API endpoint to manage custom templates"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    # Create custom templates directory if it doesn't exist
    custom_templates_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_templates')
    os.makedirs(custom_templates_dir, exist_ok=True)
    
    # Path to the JSON file storing custom template data
    templates_json_path = os.path.join(custom_templates_dir, 'templates.json')
    
    # Initialize templates file if it doesn't exist
    if not os.path.exists(templates_json_path):
        with open(templates_json_path, 'w') as f:
            json.dump({"templates": []}, f)
    
    # GET request - return all custom templates
    if request.method == 'GET':
        try:
            with open(templates_json_path, 'r') as f:
                templates_data = json.load(f)
            return jsonify({
                'success': True,
                'templates': templates_data.get('templates', [])
            })
        except Exception as e:
            app.logger.error(f"Error getting custom templates: {str(e)}")
            return jsonify({
                'success': False,
                'message': f"Error getting custom templates: {str(e)}"
            }), 500
    
    # POST request - create a new custom template
    elif request.method == 'POST':
        try:
            # Debug logging
            print("="*50)
            print("Custom template POST request received")
            print(f"Request form: {list(request.form.keys())}")
            print(f"Request form values: {dict(request.form)}")
            print(f"Request files: {list(request.files.keys())}")
            
            # Get form data
            template_name = request.form.get('name')
            highlight_color = request.form.get('color', '#00BFFF')
            font_type = request.form.get('font_type', 'existing')
            
            # Use provided template_id if available, otherwise generate one
            template_id = request.form.get('template_id')
            if not template_id:
                template_id = f"custom_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            print(f"Template ID: {template_id}")
            print(f"Template name: {template_name}")
            print(f"Highlight color: {highlight_color}")
            print(f"Font type: {font_type}")
            
            if not template_name:
                return jsonify({
                    'success': False,
                    'message': 'Template name is required'
                }), 400
            
            # Handle font selection or upload
            font_path = None
            if font_type == 'custom' and 'font_file' in request.files:
                font_file = request.files['font_file']
                if font_file.filename:
                    # Ensure filename is safe
                    font_filename = secure_filename(font_file.filename)
                    # Save the font file
                    font_path = os.path.join(custom_templates_dir, font_filename)
                    font_file.save(font_path)
                    
                    # Also copy to Fonts directory for use in processing
                    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fonts')
                    shutil.copy(font_path, os.path.join(fonts_dir, font_filename))
                    
                    font_name = font_filename
            else:
                # Use existing font
                font_name = request.form.get('font')
            
            # Create template data
            template_data = {
                'id': template_id,
                'name': template_name,
                'color': highlight_color,
                'font': font_name,
                'created_at': datetime.now().isoformat()
            }
            
            # Add to templates.json
            with open(templates_json_path, 'r') as f:
                templates_data = json.load(f)
            
            # Add isCustom flag
            template_data['isCustom'] = True
            
            templates_data['templates'].append(template_data)
            
            print(f"Saving template to {templates_json_path}")
            print(f"Template data: {template_data}")
            
            with open(templates_json_path, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            print(f"Template saved successfully")
            print("="*50)
            
            return jsonify({
                'success': True,
                'message': 'Custom template created successfully',
                'template_id': template_id,
                'template': template_data
            })
            
        except Exception as e:
            error_msg = f"Error creating custom template: {str(e)}"
            print("="*50)
            print(error_msg)
            print(f"Exception details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("="*50)
            
            app.logger.error(error_msg)
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500

@app.route('/api/edit-captions', methods=['POST', 'OPTIONS'])
@log_route_details
def edit_captions():
    """Edit captions for a video and generate a new video with the edited captions"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        # Get parameters from request
        video_url = request.form.get('video_url')
        captions_json = request.form.get('captions')
        template_id = request.form.get('template_id', 'minimal_white')
        highlight_color = request.form.get('highlight_color')
        user_id = request.form.get('user_id')
        user_plan = request.form.get('user_plan', 'free')
        
        # Parse captions
        captions = json.loads(captions_json)
        
        # Create a unique ID for the edited video
        edit_id = str(uuid.uuid4())
        
        # Extract the original filename from the video URL
        original_filename = os.path.basename(video_url.split('?')[0])
        base_filename = original_filename.split('_', 1)[1] if '_' in original_filename else original_filename
        
        # Create new filename for edited video
        edited_filename = f"{edit_id}_{base_filename}"
        
        # Get the full path to the original video
        video_path = os.path.join(app.config['PROCESSED_FOLDER'], original_filename)
        
        # If the video URL is a full URL, download it first
        if video_url.startswith(('http://', 'https://')):
            import requests
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                return jsonify({'error': 'Failed to download video'}), 500
        
        # Create output path for edited video
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], edited_filename)
        
        # Convert captions to phrases format expected by add_captions_to_video
        phrases = []
        for caption in captions:
            # Split caption text into words
            words = caption['text'].split()
            
            # Create a phrase for each word (or group of words if needed)
            for i in range(0, len(words), 3):  # Group by 3 words max
                word_group = words[i:i+3]
                phrase_text = ' '.join(word_group)
                
                # Calculate time for this phrase within the caption
                phrase_duration = (caption['end'] - caption['start']) / ((len(words) + 2) // 3)
                phrase_start = caption['start'] + (i // 3) * phrase_duration
                phrase_end = phrase_start + phrase_duration
                
                phrases.append({
                    'text': phrase_text,
                    'start': phrase_start,
                    'end': phrase_end
                })
        
        # Add watermark for free plan users
        add_watermark = user_plan.lower() == "free"
        
        # Add captions to the video with the selected template
        # Get caption customization options if available
        caption_vertical_position_val = request.json.get("caption_vertical_position", 50)
        caption_font_size_val = request.json.get("caption_font_size", 100)
        
        add_captions_to_video(
            video_path, 
            output_path, 
            phrases, 
            template=template_id, 
            highlight_color=highlight_color, 
            add_watermark=add_watermark, 
            user_plan=user_plan,
            caption_vertical_position=caption_vertical_position_val,
            caption_font_size=caption_font_size_val
        )
        
        # Generate URL for the edited video
        edited_video_url = url_for('processed_video', filename=edited_filename, _external=True)
        
        # Save the edit to user history if user_id is provided
        if user_id:
            # Here you would typically save to a database
            # For now, we'll just log it
            print(f"User {user_id} edited captions: {edited_filename}")
        
        # Save the captions as a JSON file for future editing
        transcription_filename = edited_filename.rsplit('.', 1)[0] + '.json'
        transcription_path = os.path.join(app.config['PROCESSED_FOLDER'], transcription_filename)
        
        with open(transcription_path, 'w') as f:
            json.dump({
                "segments": [],  # Original segments (empty for edited captions)
                "captions": captions,  # Store the edited captions
                "filename": edited_filename,
                "edited": True
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'video_url': edited_video_url,
            'filename': edited_filename,
            'transcription_url': url_for('get_transcription', filename=transcription_filename, _external=True)
        })
        
    except Exception as e:
        print(f"Error editing captions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to edit captions: {str(e)}'}), 500

@app.route('/api/generate-template-preview', methods=['POST', 'OPTIONS'])
@log_route_details
def generate_template_preview():
    """API endpoint to generate a preview for a custom template"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        # Get template data from request
        data = request.json
        template_id = data.get('id')
        template_name = data.get('name')
        text_color = data.get('textColor', '#FFFFFF')
        highlight_color = data.get('highlightColor', '#FF3C3C')
        font_family = data.get('fontFamily', 'Arial')
        font_size = data.get('fontSize', 32)
        
        # Create a temporary template for preview generation
        temp_template_id = f"custom_{template_id}"
        
        # Import the generate_template_previews module
        from generate_template_previews import generate_template_preview as gen_preview
        
        # Create a function to generate the preview with the custom template
        def generate_custom_preview():
            # Override the create_caption_clip function to use custom template settings
            global create_caption_clip
            original_create_caption_clip = create_caption_clip
            
            def custom_create_caption_clip(text, video_size, duration, font_size=40, template=None, current_word=None, current_word_color=None, is_profane=False, caption_vertical_position=50):
                if template == temp_template_id:
                    # Use custom template settings
                    return original_create_caption_clip(
                        text=text,
                        video_size=video_size,
                        duration=duration,
                        font_size=font_size,
                        template="minimal_white",  # Use minimal_white as base template
                        current_word=current_word,
                        current_word_color=current_word_color,
                        is_profane=is_profane,
                        custom_text_color=text_color,
                        custom_highlight_color=highlight_color,
                        custom_font=font_family,
                        caption_vertical_position=caption_vertical_position
                    )
                else:
                    # Use original function for other templates
                    return original_create_caption_clip(
                        text=text,
                        video_size=video_size,
                        duration=duration,
                        font_size=font_size,
                        template=template,
                        current_word=current_word,
                        current_word_color=current_word_color,
                        is_profane=is_profane,
                        caption_vertical_position=caption_vertical_position
                    )
            
            # Replace the create_caption_clip function temporarily
            create_caption_clip = custom_create_caption_clip
            
            try:
                # Generate the preview
                preview_path = gen_preview(temp_template_id, template_name)
                return preview_path
            finally:
                # Restore the original create_caption_clip function
                create_caption_clip = original_create_caption_clip
        
        # Generate the preview
        preview_path = generate_custom_preview()
        
        # Get the relative path for the frontend
        relative_path = preview_path.replace('frontend/public/', '/')
        
        return jsonify({
            "success": True,
            "previewUrl": relative_path,
            "message": f"Preview generated successfully for {template_name}"
        })
    except Exception as e:
        print(f"Error generating template preview: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to generate template preview"
        }), 500

@app.route('/api/edit-video', methods=['POST', 'OPTIONS'])
@log_route_details
def edit_video():
    """Edit a video with various effects and modifications"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        # Get parameters from request
        video_url = request.form.get('video_url')
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        volume = float(request.form.get('volume', 1.0))
        filter_type = request.form.get('filter', 'none')
        text_overlays_json = request.form.get('text_overlays', '[]')
        user_id = request.form.get('user_id')
        
        # Parse text overlays
        text_overlays = json.loads(text_overlays_json)
        
        # Create a unique ID for the edited video
        edit_id = str(uuid.uuid4())
        
        # Extract the original filename from the video URL
        original_filename = os.path.basename(video_url.split('?')[0])
        base_filename = original_filename.split('_', 1)[1] if '_' in original_filename else original_filename
        
        # Create new filename for edited video
        edited_filename = f"{edit_id}_{base_filename}"
        
        # Get the full path to the original video
        video_path = os.path.join(app.config['PROCESSED_FOLDER'], original_filename)
        
        # If the video URL is a full URL, download it first
        if video_url.startswith(('http://', 'https://')):
            import requests
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                return jsonify({'error': 'Failed to download video'}), 500
        
        # Create output path for edited video
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], edited_filename)
        
        # Load the video using MoviePy
        video = VideoFileClip(video_path)
        video.fps = video.fps or 24
        
        # Apply trimming if needed
        if start_time > 0 or end_time < video.duration:
            # Ensure end_time is valid
            if end_time <= 0 or end_time > video.duration:
                end_time = video.duration
                
            video = video.subclip(start_time, end_time)
            video = video.set_fps(video.fps)
        
        # Apply volume adjustment
        if volume != 1.0:
            video = video.volumex(volume)
            video = video.set_fps(video.fps)
        
        # Apply filter effects
        if filter_type != 'none':
            # Create a function to apply the selected filter
            def apply_filter(image):
                import numpy as np
                
                if filter_type == 'grayscale':
                    # Convert to grayscale
                    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    return np.stack([gray, gray, gray], axis=2)
                    
                elif filter_type == 'sepia':
                    # Apply sepia tone
                    sepia_matrix = np.array([
                        [0.393, 0.769, 0.189],
                        [0.349, 0.686, 0.168],
                        [0.272, 0.534, 0.131]
                    ])
                    sepia_img = np.dot(image[...,:3], sepia_matrix.T)
                    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
                    return sepia_img
                    
                elif filter_type == 'invert':
                    # Invert colors
                    return 255 - image
                    
                elif filter_type == 'brightness':
                    # Increase brightness
                    return np.clip(image * 1.2, 0, 255).astype(np.uint8)
                    
                return image
            
            # Apply the filter to the video
            video = video.fl_image(apply_filter)
        
        # Apply text overlays
        if text_overlays:
            from moviepy.editor import TextClip, CompositeVideoClip
            
            text_clips = []
            for overlay in text_overlays:
                # Create text clip
                txt_clip = TextClip(
                    overlay['text'],
                    fontsize=overlay['fontSize'],
                    color=overlay['color'],
                    font=overlay.get('fontFamily', 'Arial')
                )
                
                # Position the text
                txt_clip = txt_clip.set_position((overlay['x'], overlay['y']))
                
                # Set duration to match the video
                txt_clip = txt_clip.set_duration(video.duration)
                
                text_clips.append(txt_clip)
            
            # Combine video with text overlays
            if text_clips:
                video = CompositeVideoClip([video] + text_clips)
                video = video.set_fps(video.fps)
        
        # Write the edited video to file
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="128k",
            preset="veryfast",
            temp_audiofile=os.path.join("/app/tmp", f"temp-audio-{uuid.uuid4().hex}.m4a"),
            remove_temp=True,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.0",
                "-movflags", "+faststart"
            ]
        )
        
        # Close the video to release resources
        video.close()
        
        # Generate URL for the edited video
        edited_video_url = url_for('processed_video', filename=edited_filename, _external=True)
        
        # Save the edit to user history if user_id is provided
        if user_id:
            # Here you would typically save to a database
            # For now, we'll just log it
            print(f"User {user_id} edited video: {edited_filename}")
        
        return jsonify({
            'success': True,
            'video_url': edited_video_url,
            'filename': edited_filename
        })
        
    except Exception as e:
        print(f"Error editing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to edit video: {str(e)}'}), 500

# YouTube OAuth endpoints
# YouTube OAuth functionality has been removed

# Set Hugging Face token for pyannote.audio if not already set
if "HF_TOKEN" not in os.environ:
    # In production, this should be set in the environment or a .env file
    # For development, create a .env file with HF_TOKEN=your_token_here
    print("Warning: HF_TOKEN not found in environment variables. Please set it for pyannote.audio to work properly.")

# Speech enhancement feature has been removed

if __name__ == '__main__':
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    # Disable auto-reloader while keeping debug mode to prevent restarts during processing
    app.run(debug=debug_mode, use_reloader=False, host=host, port=port)
