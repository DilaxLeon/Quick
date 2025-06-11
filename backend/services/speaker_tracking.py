"""
Speaker tracking and diarization module.

This module combines YOLOv8 for visual speaker tracking and pyannote.audio for speaker diarization
to keep the active speaker centered in the frame and switch between speakers.

Note: This module uses the pyannote.audio model which requires a Hugging Face token.
The token should be set as an environment variable named 'HUGGINGFACE_TOKEN'.
"""

import os
import cv2
import numpy as np
import torch
import tempfile
import time as time_module  # Renamed to avoid variable name conflicts
import traceback
import random
import math  # Added for enhanced camera movement calculations
from ultralytics import YOLO
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess
import json
from collections import defaultdict

# Initialize models
def initialize_models():
    """Initialize YOLOv8 and pyannote.audio models."""
    # Create models directory if it doesn't exist
    os.makedirs('backend/models', exist_ok=True)
    
    # Initialize YOLOv8 model for person detection
    try:
        yolo_model = YOLO('yolov8n.pt')  # Use the smallest model for speed
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        print("Downloading YOLOv8 model...")
        yolo_model = YOLO('yolov8n.pt')  # This will download the model if not present
    
    # Initialize pyannote.audio model for speaker diarization
    try:
        # Use the Hugging Face token from environment variables
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN environment variable is not set")
            print("Please set the HF_TOKEN environment variable with your HuggingFace token")
            diarization_pipeline = None
        else:
            print(f"Using HuggingFace token: {hf_token[:5]}...")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
    except Exception as e:
        print(f"Error loading pyannote.audio model: {e}")
        print("Please check that your HUGGINGFACE_TOKEN is valid and you have access to the model")
        diarization_pipeline = None
    
    return yolo_model, diarization_pipeline

# Extract audio from video
def extract_audio(video_path):
    """Extract audio from video file."""
    audio_path = tempfile.mktemp(suffix='.wav')
    
    try:
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "16000"])
        video.close()
    except Exception as e:
        print(f"Error extracting audio with moviepy: {e}")
        # Fallback to ffmpeg directly
        try:
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', 
                audio_path, '-y'
            ]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error extracting audio with ffmpeg: {e}")
            return None
    
    return audio_path

# Perform speaker diarization
def perform_diarization(audio_path, diarization_pipeline):
    """Perform speaker diarization on audio file."""
    if diarization_pipeline is None:
        print("Diarization pipeline not initialized")
        return None
    
    try:
        # Run diarization
        diarization = diarization_pipeline(audio_path)
        
        # Convert diarization result to a list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return segments
    except Exception as e:
        print(f"Error performing diarization: {e}")
        return None

# Detect people in video frames
def detect_people(video_path, yolo_model, sample_rate=1):
    """
    Detect people in video frames using YOLOv8.
    
    Args:
        video_path: Path to the video file
        yolo_model: YOLOv8 model
        sample_rate: Process every Nth frame to save time
        
    Returns:
        Dictionary mapping frame indices to detected people
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Dictionary to store detected people for each frame
    people_detections = {}
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_idx % sample_rate == 0:
            # Run YOLOv8 detection
            results = yolo_model(frame, classes=0)  # class 0 is person
            
            # Extract person detections
            people = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    
                    # Only include high-confidence detections
                    if conf > 0.5:
                        # Calculate center point
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Store detection with bounding box and center point
                        people.append({
                            "bbox": [x1, y1, x2, y2],
                            "center": [center_x, center_y],
                            "confidence": float(conf)
                        })
            
            # Store detections for this frame
            people_detections[frame_idx] = people
        
        frame_idx += 1
        
        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
    
    cap.release()
    return people_detections, fps

# Match speakers with detected people
def match_speakers_to_people(diarization_segments, people_detections, fps):
    """
    Match diarization segments with detected people.
    When multiple speakers are active simultaneously, switches between speakers
    every 5 seconds. When only one speaker remains, focuses on that speaker.
    
    Args:
        diarization_segments: List of speaker segments from diarization
        people_detections: Dictionary mapping frame indices to detected people
        fps: Frames per second of the video
        
    Returns:
        Dictionary mapping frame indices to the active speaker's position
    """
    if not diarization_segments or not people_detections:
        return None
    
    # Create a mapping of frame index to active speaker
    active_speaker_frames = {}
    
    # For each frame, determine the active speaker
    # Keep track of the current primary speaker and when they started
    current_primary_speaker = None
    current_primary_start_frame = 0
    speaker_switch_interval = int(5 * fps)  # Switch every 5 seconds (in frames)
    
    for frame_idx in sorted(people_detections.keys()):
        time_seconds = frame_idx / fps
        
        # Find all active speakers at this time
        active_speakers = []
        for segment in diarization_segments:
            if segment["start"] <= time_seconds <= segment["end"]:
                active_speakers.append({
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "duration": time_seconds - segment["start"]  # How long they've been speaking
                })
        
        # If we have multiple speakers, choose one to focus on
        active_speaker = None
        if active_speakers:
            if len(active_speakers) == 1:
                # Only one speaker, focus on them
                active_speaker = active_speakers[0]["speaker"]
                current_primary_speaker = active_speaker
                current_primary_start_frame = frame_idx
            else:
                # Multiple speakers active simultaneously
                # Check if it's time to switch speakers (every 5 seconds)
                frames_with_current_speaker = frame_idx - current_primary_start_frame
                
                if current_primary_speaker is None or frames_with_current_speaker >= speaker_switch_interval:
                    # Time to switch speakers - rotate through the available speakers
                    if current_primary_speaker is None:
                        # First frame with multiple speakers, start with any speaker
                        active_speaker = active_speakers[0]["speaker"]
                    else:
                        # Find the current speaker's index in the active speakers list
                        current_idx = -1
                        for i, s in enumerate(active_speakers):
                            if s["speaker"] == current_primary_speaker:
                                current_idx = i
                                break
                        
                        # Move to the next speaker (cycling back to the beginning if needed)
                        next_idx = (current_idx + 1) % len(active_speakers)
                        active_speaker = active_speakers[next_idx]["speaker"]
                    
                    # Update the primary speaker and start time
                    current_primary_speaker = active_speaker
                    current_primary_start_frame = frame_idx
                else:
                    # Continue with the current primary speaker if they're still active
                    if any(s["speaker"] == current_primary_speaker for s in active_speakers):
                        active_speaker = current_primary_speaker
                    else:
                        # Current primary speaker is no longer active, choose a new one
                        active_speaker = active_speakers[0]["speaker"]
                        current_primary_speaker = active_speaker
                        current_primary_start_frame = frame_idx
        
        # If there's an active speaker and people detected in this frame
        if active_speaker and people_detections[frame_idx]:
            active_speaker_frames[frame_idx] = {
                "speaker": active_speaker,
                "people": people_detections[frame_idx]
            }
    
    # Track speakers across frames to maintain consistency
    speaker_positions = defaultdict(list)
    
    # First pass: collect all positions for each speaker
    for frame_idx, data in active_speaker_frames.items():
        speaker = data["speaker"]
        people = data["people"]
        
        # If only one person is detected, assume they're the speaker
        if len(people) == 1:
            speaker_positions[speaker].append({
                "frame": frame_idx,
                "position": people[0]["center"]
            })
    
    # Second pass: assign the most likely person to each speaker
    speaker_tracking = {}
    
    for frame_idx in sorted(active_speaker_frames.keys()):
        data = active_speaker_frames[frame_idx]
        speaker = data["speaker"]
        people = data["people"]
        
        # If we have previous positions for this speaker
        if speaker in speaker_positions and speaker_positions[speaker]:
            # Calculate the average position of this speaker from previous frames
            prev_positions = [p["position"] for p in speaker_positions[speaker][-10:]]  # Use last 10 positions
            if prev_positions:
                avg_x = sum(pos[0] for pos in prev_positions) / len(prev_positions)
                avg_y = sum(pos[1] for pos in prev_positions) / len(prev_positions)
                
                # Find the person closest to the average position
                closest_person = None
                min_distance = float('inf')
                
                for person in people:
                    center_x, center_y = person["center"]
                    distance = ((center_x - avg_x) ** 2 + (center_y - avg_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person
                
                if closest_person:
                    speaker_tracking[frame_idx] = {
                        "speaker": speaker,
                        "position": closest_person["center"],
                        "bbox": closest_person["bbox"]
                    }
                    
                    # Update the speaker's position history
                    speaker_positions[speaker].append({
                        "frame": frame_idx,
                        "position": closest_person["center"]
                    })
        
        # If no previous positions or couldn't find a match, just use the most confident detection
        if frame_idx not in speaker_tracking and people:
            most_confident = max(people, key=lambda p: p["confidence"])
            speaker_tracking[frame_idx] = {
                "speaker": speaker,
                "position": most_confident["center"],
                "bbox": most_confident["bbox"]
            }
            
            # Update the speaker's position history
            speaker_positions[speaker].append({
                "frame": frame_idx,
                "position": most_confident["center"]
            })
    
    return speaker_tracking

# Generate crop parameters for each frame
def generate_crop_parameters(speaker_tracking, video_width, video_height, fps, smooth_window=90):
    """
    Generate professional-looking crop parameters to keep the active speaker centered.
    Enhanced for smoother transitions between multiple speakers.
    
    Args:
        speaker_tracking: Dictionary mapping frame indices to active speaker positions
        video_width: Width of the video
        video_height: Height of the video
        fps: Frames per second of the video
        smooth_window: Window size for smoothing the crop parameters (increased for smoother movement)
        
    Returns:
        Dictionary mapping frame indices to crop parameters
    """
    # Target aspect ratio (9:16)
    target_ratio = 9 / 16
    
    # Calculate the width of the crop window based on the target aspect ratio
    crop_width = min(video_width, int(video_height * target_ratio))
    crop_height = video_height
    
    # Ensure dimensions are even (required for H.264 encoding)
    if crop_width % 2 != 0:
        crop_width -= 1
    if crop_height % 2 != 0:
        crop_height -= 1
    
    # Initialize crop parameters for all frames
    all_frames = max(speaker_tracking.keys()) + 1 if speaker_tracking else 0
    crop_params = {}
    
    # Default crop parameters (center of the frame)
    default_center_x = video_width / 2
    
    # Enhanced camera movement parameters for professional look with improved smoothness
    dead_zone_size = crop_width * 0.1  # 10% of crop width - larger dead zone for more stability
    max_speed = crop_width * 0.012  # Reduced maximum camera movement speed (1.2% of crop width)
    min_speed = crop_width * 0.001  # Lower minimum speed for more subtle movements
    inertia_factor = 0.94  # Higher inertia for smoother starts/stops (was 0.92)
    look_ahead_frames = int(fps * 1.5)  # Look ahead 1.5 seconds to better anticipate movements
    
    # First pass: assign crop parameters based on speaker positions
    for frame_idx in range(all_frames):
        if frame_idx in speaker_tracking:
            # Get the speaker's position
            position = speaker_tracking[frame_idx]["position"]
            center_x = position[0]
            
            # Ensure the crop window stays within the video bounds
            min_x = crop_width / 2
            max_x = video_width - crop_width / 2
            center_x = max(min_x, min(center_x, max_x))
            
            crop_params[frame_idx] = {
                "center_x": center_x,
                "width": crop_width,
                "height": crop_height,
                "raw_center_x": center_x,  # Store the raw position for reference
                "is_keyframe": True,  # Mark frames with actual speaker detection as keyframes
                "speaker_id": speaker_tracking[frame_idx].get("speaker", "unknown")  # Store speaker ID for transition handling
            }
        else:
            # If no speaker tracking for this frame, use the previous frame's parameters
            # or default to the center if this is the first frame
            if frame_idx > 0 and frame_idx - 1 in crop_params:
                prev_params = crop_params[frame_idx - 1].copy()
                crop_params[frame_idx] = prev_params
                # Keep the raw_center_x value from the previous frame
                if "raw_center_x" in prev_params:
                    crop_params[frame_idx]["raw_center_x"] = prev_params["raw_center_x"]
                else:
                    crop_params[frame_idx]["raw_center_x"] = prev_params["center_x"]
                crop_params[frame_idx]["is_keyframe"] = False  # Not a keyframe
            else:
                crop_params[frame_idx] = {
                    "center_x": default_center_x,
                    "width": crop_width,
                    "height": crop_height,
                    "raw_center_x": default_center_x,
                    "is_keyframe": False,
                    "speaker_id": "unknown"
                }
    
    # Second pass: apply professional camera movement with enhanced inertia and look-ahead
    smoothed_params = {}
    current_camera_pos = default_center_x
    camera_velocity = 0.0  # Initial camera velocity
    last_significant_movement_frame = 0
    
    # Find speaker transitions (when speaker ID changes)
    speaker_transitions = []
    prev_speaker = None
    
    for frame_idx in range(all_frames):
        if frame_idx in crop_params and crop_params[frame_idx].get("is_keyframe", False):
            current_speaker = crop_params[frame_idx].get("speaker_id", "unknown")
            if prev_speaker is not None and current_speaker != prev_speaker:
                # Speaker has changed - mark as a transition point
                speaker_transitions.append(frame_idx)
            prev_speaker = current_speaker
    
    # Find scene changes (significant speaker position changes)
    scene_changes = []
    prev_pos = None
    
    for frame_idx in range(all_frames):
        if frame_idx in crop_params and crop_params[frame_idx].get("is_keyframe", False):
            current_pos = crop_params[frame_idx]["raw_center_x"]
            if prev_pos is not None:
                # If position change is significant (more than 25% of frame width), mark as scene change
                if abs(current_pos - prev_pos) > (video_width * 0.25):
                    scene_changes.append(frame_idx)
            prev_pos = current_pos
    
    # Process each frame with enhanced cinematic movement
    for frame_idx in range(all_frames):
        # Check if this is a scene change or speaker transition that needs special handling
        is_scene_change = frame_idx in scene_changes
        is_speaker_transition = frame_idx in speaker_transitions
        
        # Look ahead to anticipate speaker movement (longer look-ahead for better anticipation)
        future_positions = []
        future_keyframes = []
        
        for i in range(1, look_ahead_frames + 1):
            future_idx = frame_idx + i
            if future_idx < all_frames and future_idx in crop_params:
                future_positions.append(crop_params[future_idx]["raw_center_x"])
                if crop_params[future_idx].get("is_keyframe", False):
                    future_keyframes.append((i, crop_params[future_idx]["raw_center_x"]))
        
        # Calculate target position with enhanced anticipation of future movement
        current_target = crop_params[frame_idx]["raw_center_x"]
        
        # Enhanced anticipation logic
        if future_positions:
            # Weight future positions with decreasing importance
            future_weights = [max(0, 1 - (i / look_ahead_frames)**1.5) for i in range(len(future_positions))]
            weighted_future = sum(pos * weight for pos, weight in zip(future_positions, future_weights))
            total_weight = sum(future_weights)
            
            if total_weight > 0:
                # Prioritize actual speaker detections (keyframes) in the future
                if future_keyframes:
                    # Sort by distance in frames (closest first)
                    future_keyframes.sort(key=lambda x: x[0])
                    # Get the closest future keyframe
                    frames_away, keyframe_pos = future_keyframes[0]
                    # Calculate anticipation factor based on distance
                    # Closer keyframes have stronger influence
                    keyframe_factor = max(0.1, 0.5 * (1 - frames_away / look_ahead_frames))
                    # Blend current target with keyframe and general future prediction
                    anticipation_factor = 0.45  # Increased from 0.4 for better anticipation
                    general_future = weighted_future / total_weight
                    target_pos = (1 - anticipation_factor) * current_target + \
                                anticipation_factor * ((1 - keyframe_factor) * general_future + keyframe_factor * keyframe_pos)
                else:
                    # No keyframes, use weighted average of future positions
                    anticipation_factor = 0.4  # Increased from 0.35
                    target_pos = (1 - anticipation_factor) * current_target + anticipation_factor * (weighted_future / total_weight)
            else:
                target_pos = current_target
        else:
            target_pos = current_target
        
        # Apply enhanced dead zone with adaptive size
        # Dead zone is larger when camera is already moving slowly (creates natural pauses)
        current_speed = abs(camera_velocity)
        adaptive_dead_zone = dead_zone_size * (1.0 + 0.5 * (1.0 - min(1.0, current_speed / max_speed)))
        
        distance_to_target = target_pos - current_camera_pos
        if abs(distance_to_target) < adaptive_dead_zone and not (is_scene_change or is_speaker_transition):
            # Target is within dead zone, gradually reduce velocity instead of immediate stop
            # This creates more natural deceleration
            camera_velocity *= 0.9  # Gradual slowdown
            if abs(camera_velocity) < min_speed:
                camera_velocity = 0  # Stop completely when very slow
        else:
            # Calculate desired velocity with enhanced cinematic curve
            # Use cubic easing for acceleration/deceleration (more professional)
            direction = 1 if distance_to_target > 0 else -1
            adjusted_distance = abs(distance_to_target) - (adaptive_dead_zone if not (is_scene_change or is_speaker_transition) else 0)
            
            # Normalize distance as percentage of screen width for better control
            normalized_distance = min(1.0, adjusted_distance / (video_width * 0.5))
            
            # Apply cubic easing curve for more cinematic movement
            # Slower start, faster middle, slower end
            eased_factor = normalized_distance ** 2 * (3 - 2 * normalized_distance)
            
            # Special handling for scene changes and speaker transitions
            if is_scene_change:
                # For scene changes, move more quickly but still with easing
                target_velocity = direction * max(min_speed, min(max_speed * 1.5, adjusted_distance * 0.15))
                # Reset inertia to allow quicker response
                inertia_factor = 0.7
                last_significant_movement_frame = frame_idx
            elif is_speaker_transition:
                # For speaker transitions, move at a moderate pace - smoother than scene changes
                # but faster than regular movement
                target_velocity = direction * max(min_speed, min(max_speed * 1.2, adjusted_distance * 0.12))
                # Use moderate inertia for smooth but responsive transitions
                inertia_factor = 0.8
                last_significant_movement_frame = frame_idx
            else:
                # Normal movement with enhanced easing
                target_velocity = direction * max(min_speed, min(max_speed, max_speed * eased_factor))
                
                # Gradually restore higher inertia after significant movements
                frames_since_significant = frame_idx - last_significant_movement_frame
                if frames_since_significant < fps * 1.5:  # Within 1.5 seconds of significant movement
                    # Gradually increase inertia back to maximum
                    progress = min(1.0, frames_since_significant / (fps * 1.5))
                    adaptive_inertia = 0.8 + (0.14 * progress)  # 0.8 to 0.94
                    inertia_factor = adaptive_inertia
                else:
                    inertia_factor = 0.94  # Default high inertia for smooth movement
        
        # Apply inertia with enhanced smoothing - camera velocity changes more gradually
        camera_velocity = camera_velocity * inertia_factor + target_velocity * (1 - inertia_factor)
        
        # Apply additional damping to very small movements to prevent micro-jitters
        if abs(camera_velocity) < (min_speed * 0.5):
            camera_velocity *= 0.75  # Additional damping for tiny movements (was 0.8)
        
        # Update camera position
        current_camera_pos += camera_velocity
        
        # Enhanced edge handling with easing
        min_x = crop_width / 2
        max_x = video_width - crop_width / 2
        
        # Create a soft boundary near edges (12% of crop width)
        edge_margin = crop_width * 0.12  # Increased from 10% to 12%
        
        # If approaching left edge
        if current_camera_pos < (min_x + edge_margin):
            # Calculate how far into the margin we are (0 to 1)
            edge_factor = (current_camera_pos - min_x) / edge_margin
            # Apply cubic easing to slow down gracefully near the edge
            if edge_factor > 0:
                edge_ease = edge_factor ** 2 * (3 - 2 * edge_factor)
                # Adjust position with easing (slows down as it approaches edge)
                current_camera_pos = min_x + (edge_margin * edge_ease)
            else:
                current_camera_pos = min_x
        
        # If approaching right edge
        elif current_camera_pos > (max_x - edge_margin):
            # Calculate how far into the margin we are (0 to 1)
            edge_factor = (max_x - current_camera_pos) / edge_margin
            # Apply cubic easing to slow down gracefully near the edge
            if edge_factor > 0:
                edge_ease = edge_factor ** 2 * (3 - 2 * edge_factor)
                # Adjust position with easing (slows down as it approaches edge)
                current_camera_pos = max_x - (edge_margin * edge_ease)
            else:
                current_camera_pos = max_x
        
        # Store the smoothed camera position
        smoothed_center_x = current_camera_pos
        
        # Calculate crop coordinates with precise pixel alignment
        x1 = max(0, int(smoothed_center_x - crop_width / 2))
        y1 = 0
        x2 = min(video_width, int(smoothed_center_x + crop_width / 2))
        y2 = crop_height
        
        # Ensure the width and height are even numbers (required for H.264 encoding)
        width = x2 - x1
        if width % 2 != 0:
            # Adjust x2 to make width even
            if x2 < video_width:
                x2 += 1
            else:
                x1 -= 1
        
        height = y2 - y1
        if height % 2 != 0:
            # Adjust y2 to make height even
            if y2 < video_height:
                y2 += 1
            else:
                y1 -= 1
        
        # Store smoothed parameters
        smoothed_params[frame_idx] = {
            "center_x": smoothed_center_x,
            "width": crop_width,
            "height": crop_height,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "camera_velocity": camera_velocity,  # Store for debugging
            "is_scene_change": is_scene_change,  # Mark scene changes for special handling
            "is_speaker_transition": is_speaker_transition  # Mark speaker transitions
        }
    
    return smoothed_params

# Apply dynamic cropping to video using the tracking.py approach
def apply_dynamic_cropping_tracking(video_path, output_path, active_speaker_segments, fps, professional_camera=True):
    """
    Apply dynamic cropping to video to keep the active speaker centered using the tracking.py approach.
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        active_speaker_segments: List of speaker segments from diarization
        fps: Frames per second of the video
        professional_camera: Whether to use professional camera movement
        
    Returns:
        Path to the cropped video
    """
    try:
        # Load YOLO model
        yolo_model, _ = initialize_models()
        if yolo_model is None:
            print("Failed to load YOLO model, falling back to standard processing")
            return None

        # Load the video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate dimensions for 9:16 aspect ratio
        target_aspect_ratio = 9/16  # Width/Height for vertical video

        # Calculate the width of the 9:16 crop window (maintaining original height)
        crop_width = int(height * target_aspect_ratio)

        # If the video is already narrower than 9:16, we don't need to crop
        if width <= crop_width:
            print("Video is already narrower than 9:16, no need for speaker tracking")
            return None

        # Create a temporary output file with timestamp
        temp_output_path = os.path.join(os.path.dirname(output_path), f'temp_speaker_tracked_{int(time_module.time())}.mp4')

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (crop_width, height))

        # Initialize variables for tracking with enhanced professional settings
        prev_center_x = width // 2  # Start with center of frame
        # Increased smoothing factor for more cinematic movement
        smoothing_factor = 0.98 if professional_camera else 0.92  # Higher value = smoother camera movement
        
        # For performance optimization, we'll process every nth frame
        if total_frames > 3000:  # For very long videos
            frame_skip = 3  # Reduced from 4 for better tracking
        elif total_frames > 1000:  # For medium length videos
            frame_skip = 2  # Reduced from 3 for better tracking
        else:
            frame_skip = 1  # Process every frame for shorter videos for best quality
        
        # Store detection history to handle frames with no detections
        detection_history = []
        max_history = 45  # Increased history size for smoother transitions (was 30)
        
        # Enhanced weighted history for better prediction
        weighted_history = []  # Store (position, weight) pairs
        max_weighted_history = 60  # Longer history for better prediction (was 45)
        
        # Variables for speaker persistence with improved settings
        current_speaker_id = None  # ID to track the current speaker
        speaker_frame_count = 0  # Count of frames with the current speaker
        min_speaker_frames = 20  # Increased from 15 - stay with speaker longer for stability
        
        # Variables for handling temporary speaker disappearance
        speaker_lost_count = 0  # Count of frames where current speaker is not detected
        max_speaker_lost_frames = 45  # Increased from 30 - wait longer before switching
        
        # Camera movement control variables for professional look
        camera_velocity = 0.0  # Current camera movement speed
        target_velocity = 0.0  # Target camera movement speed
        inertia_factor = 0.95  # Camera inertia (higher = smoother movement) - increased from 0.92
        max_velocity = width * 0.012  # Maximum camera speed (1.2% of width per frame) - reduced from 1.5%
        min_velocity = width * 0.0008  # Minimum noticeable camera speed - reduced for smoother micro-movements
        
        # Scene change detection
        last_significant_movement = 0  # Frame index of last significant camera movement
        scene_change_cooldown = int(fps * 1.5)  # Minimum frames between scene changes (1.5 seconds)
        
        # Cinematic composition variables
        lead_space_factor = 0.15  # Space in front of speaker (15% of crop width)
        composition_weight = 0.3  # How much to consider composition vs. centering

        print(f"Processing video with YOLO and speaker diarization: {total_frames} frames")

        # Process each frame
        frame_count = 0
        current_time = 0.0
        active_speaker = None
        
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # Only process every nth frame with YOLO for performance
            process_with_yolo = (frame_count % frame_skip == 0)

            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

            # Find the active speaker from diarization results if available
            if active_speaker_segments is not None:
                # Find all active speakers at the current time
                current_speakers = []
                for segment in active_speaker_segments:
                    if segment['start'] <= current_time <= segment['end']:
                        current_speakers.append(segment['speaker'])
                
                # If we have active speakers
                if current_speakers:
                    # If only one speaker is active, focus on them
                    if len(current_speakers) == 1:
                        active_speaker = current_speakers[0]
                    else:
                        # Multiple speakers are active
                        # Check if we need to switch speakers (every 5 seconds)
                        switch_interval_frames = int(5 * fps)  # 5 seconds in frames
                        
                        # If we don't have a previously selected active speaker or it's time to switch
                        if active_speaker is None or frame_count % switch_interval_frames == 0:
                            # If we have a previous speaker, find its index and move to the next one
                            if active_speaker in current_speakers:
                                current_idx = current_speakers.index(active_speaker)
                                next_idx = (current_idx + 1) % len(current_speakers)
                                active_speaker = current_speakers[next_idx]
                            else:
                                # No previous speaker or not in current list, choose one
                                active_speaker = random.choice(current_speakers)
                            
                            print(f"Frame {frame_count}: Switched to speaker '{active_speaker}' from {len(current_speakers)} active speakers")
                else:
                    active_speaker = None

            # Find the largest person detection (likely the main speaker)
            largest_box = None
            largest_area = 0
            active_speaker_box = None

            if process_with_yolo:
                # Run YOLOv8 inference on the frame
                results = yolo_model(frame, classes=0)  # class 0 is person

                # Store all detected persons in this frame
                detected_persons = []

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Calculate area of the bounding box
                        area = (x2 - x1) * (y2 - y1)

                        # Store this detection
                        detected_persons.append({
                            'box': (x1, y1, x2, y2),
                            'area': area,
                            'center_x': (x1 + x2) // 2
                        })

                        # Check if this is the largest person detected so far
                        if area > largest_area:
                            largest_area = area
                            largest_box = (x1, y1, x2, y2)

                # Sort detected persons by area (largest first)
                detected_persons.sort(key=lambda p: p['area'], reverse=True)

                # Determine which person to track based on speaker diarization and persistence rules
                selected_person = None

                # If we have a current speaker, increment their frame count
                if current_speaker_id is not None:
                    speaker_frame_count += 1

                # If we have detected persons
                if detected_persons:
                    # If we have active speaker information from diarization
                    if active_speaker is not None:
                        # If we have a previously selected person, try to maintain focus on them
                        if current_speaker_id is not None and speaker_frame_count < 30:  # Stick with current person for at least 30 frames
                            # Try to find the same person we were tracking before
                            if detection_history and len(detected_persons) > 1:
                                prev_center_x_from_history = detection_history[-1]
                                
                                # Find the person closest to the previous center
                                min_distance = float('inf')
                                closest_person = None
                                
                                for person in detected_persons:
                                    distance = abs(person['center_x'] - prev_center_x_from_history)
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_person = person
                                
                                # If the closest person is within a reasonable distance, keep tracking them
                                if closest_person and min_distance < (width * 0.2):  # 20% of frame width as threshold
                                    selected_person = closest_person
                                    # Continue with the same person for stability
                                    if frame_count % 300 == 0:  # Log less frequently to reduce console spam
                                        print(f"Frame {frame_count}: Maintaining focus on current speaker for stability")
                                else:
                                    # If we can't find the same person, use the largest one
                                    selected_person = detected_persons[0]
                            else:
                                # Only one person detected, assume they're the active speaker
                                selected_person = detected_persons[0]
                        else:
                            # No current speaker or we've tracked them long enough, use the largest person
                            selected_person = detected_persons[0]
                            if len(detected_persons) > 1 and frame_count % 300 == 0:  # Log less frequently
                                print(f"Frame {frame_count}: Multiple people detected, using largest as active speaker '{active_speaker}'")
                    # If no active speaker from diarization or diarization not available
                    elif current_speaker_id is None or speaker_frame_count >= min_speaker_frames:
                        # Select the largest person as the new speaker
                        selected_person = detected_persons[0]

                        # Only log when we're actually switching speakers (not on first detection)
                        if current_speaker_id is not None:
                            print(f"Frame {frame_count}: Switching to new speaker after {speaker_frame_count} frames")

                        current_speaker_id = id(selected_person)  # Use object id as unique identifier
                        speaker_frame_count = 0  # Reset frame count for new speaker
                    else:
                        # We have a current speaker who hasn't been tracked for minimum frames
                        # Try to find them among detected persons (using position similarity)
                        current_speaker_found = False

                        # If we have previous detection history, use it to find the current speaker
                        if detection_history:
                            prev_center_x_from_history = detection_history[-1]

                            # Find the person closest to the previous center
                            min_distance = float('inf')
                            closest_person = None

                            for person in detected_persons:
                                distance = abs(person['center_x'] - prev_center_x_from_history)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_person = person

                            # If the closest person is within a reasonable distance, consider it the same speaker
                            if closest_person and min_distance < (width * 0.2):  # 20% of frame width as threshold
                                selected_person = closest_person
                                current_speaker_found = True

                        # If we couldn't find the current speaker, use the largest person
                        if not current_speaker_found:
                            selected_person = detected_persons[0]

                # If we selected a person, update tracking information
                if selected_person:
                    largest_box = selected_person['box']
                    person_center_x = selected_person['center_x']
                    detection_history.append(person_center_x)
                    # Keep history at max size
                    if len(detection_history) > max_history:
                        detection_history.pop(0)

                    # Reset the speaker lost count since we found a person
                    speaker_lost_count = 0

            # Enhanced professional camera movement logic
            if largest_box:
                # Get the center of the detected person
                x1, y1, x2, y2 = largest_box
                person_center_x = (x1 + x2) // 2
                
                # Apply cinematic composition - add lead space in the direction the person is facing
                # Estimate facing direction based on position change
                facing_right = True  # Default assumption
                if len(detection_history) > 5:
                    # Compare current position to average of last few positions
                    recent_avg = sum(detection_history[-5:]) / 5
                    facing_right = person_center_x >= recent_avg
                
                # Add lead space in the direction the person is facing (rule of thirds)
                lead_space = crop_width * lead_space_factor
                if facing_right:
                    # Person facing right, add space to the right
                    composed_center_x = person_center_x - (lead_space * composition_weight)
                else:
                    # Person facing left, add space to the left
                    composed_center_x = person_center_x + (lead_space * composition_weight)
                
                # Blend centered position with composed position
                target_center_x = (person_center_x * (1 - composition_weight)) + (composed_center_x * composition_weight)
                
                # Calculate movement distance to determine if this is a significant movement
                movement_distance = abs(target_center_x - prev_center_x)
                significant_movement = movement_distance > (width * 0.25)  # 25% of frame width
                
                # Check if we can make a scene change (not in cooldown period)
                can_scene_change = (frame_count - last_significant_movement) > scene_change_cooldown
                
                # Handle scene changes (significant movements)
                if significant_movement and can_scene_change:
                    # This is a scene change - use faster but still smooth transition
                    print(f"Frame {frame_count}: Significant camera movement detected - scene change")
                    last_significant_movement = frame_count
                    
                    # For scene changes, use a special transition curve
                    # Start faster, then ease out (cinematic transition)
                    adaptive_smoothing = 0.75  # Lower smoothing for faster initial movement
                    
                    # Reset velocity for quicker response to the new position
                    camera_velocity = (target_center_x - prev_center_x) * 0.1
                else:
                    # Normal movement - use enhanced adaptive smoothing
                    # More smoothing for larger movements (counterintuitive but creates professional damping)
                    normalized_distance = min(1.0, movement_distance / (width * 0.3))
                    
                    # Cubic easing curve for professional damping
                    eased_factor = normalized_distance ** 2 * (3 - 2 * normalized_distance)
                    
                    # Calculate target velocity based on distance and easing
                    direction = 1 if (target_center_x > prev_center_x) else -1
                    target_velocity = direction * min(max_velocity, max_velocity * eased_factor)
                    
                    # Apply inertia to camera velocity (gradual acceleration/deceleration)
                    camera_velocity = camera_velocity * inertia_factor + target_velocity * (1 - inertia_factor)
                    
                    # Dampen very small movements to prevent micro-jitters
                    if abs(camera_velocity) < min_velocity:
                        camera_velocity *= 0.8
                    
                    # Use extremely high smoothing for very small movements (creates stable shots)
                    if movement_distance < (crop_width * 0.03):  # 3% of crop width
                        adaptive_smoothing = 0.99  # Very high smoothing for tiny movements
                    else:
                        # Normal adaptive smoothing
                        adaptive_smoothing = min(0.985, smoothing_factor + (normalized_distance * 0.02))
                
                # Apply the calculated smoothing
                center_x = int(adaptive_smoothing * prev_center_x + (1 - adaptive_smoothing) * target_center_x)
                
                # Enhanced smoothing for speaker transitions
                # Check if we've recently switched speakers (within the last second)
                recent_speaker_switch = False
                if active_speaker_segments is not None and len(active_speaker_segments) > 1:
                    # Calculate time in seconds for the last 1 second of frames
                    one_second_ago = current_time - 1.0
                    
                    # Check if there was a speaker change in the last second
                    prev_speaker = None
                    for t in np.arange(one_second_ago, current_time, 0.1):  # Check every 0.1 seconds
                        t_speakers = []
                        for segment in active_speaker_segments:
                            if segment['start'] <= t <= segment['end']:
                                t_speakers.append(segment['speaker'])
                        
                        # Get the active speaker at this time point
                        t_active = t_speakers[0] if len(t_speakers) == 1 else (
                            active_speaker if active_speaker in t_speakers else 
                            (t_speakers[0] if t_speakers else None)
                        )
                        
                        # Check for speaker change
                        if prev_speaker is not None and t_active is not None and prev_speaker != t_active:
                            recent_speaker_switch = True
                            break
                        
                        prev_speaker = t_active
                
                # Apply extra smoothing during speaker transitions
                if recent_speaker_switch:
                    # Use higher smoothing factor during transitions
                    transition_smoothing = 0.98
                    center_x = int(transition_smoothing * prev_center_x + (1 - transition_smoothing) * center_x)
                
                # Add to weighted history with timestamp and position
                weighted_history.append((frame_count, person_center_x, 1.0))  # Full weight for actual detections
                if len(weighted_history) > max_weighted_history:
                    weighted_history.pop(0)
                
                # Add to regular history
                detection_history.append(person_center_x)
                if len(detection_history) > max_history:
                    detection_history.pop(0)
                
                # Update previous center
                prev_center_x = center_x
                
                # Reset speaker lost count since we found them
                speaker_lost_count = 0

            elif detection_history or weighted_history:
                # Enhanced prediction when person is temporarily not detected
                
                # If we have a current speaker, increment the lost count
                if current_speaker_id is not None:
                    speaker_lost_count += 1

                    # Log that we've lost the speaker (but less frequently to reduce console spam)
                    if speaker_lost_count == 1:
                        print(f"Frame {frame_count}: Speaker temporarily lost, using enhanced prediction")

                    # If we've lost the speaker for too many frames, reset and allow switching
                    if speaker_lost_count >= max_speaker_lost_frames:
                        print(f"Frame {frame_count}: Speaker lost for {speaker_lost_count} frames, allowing switch to new speaker")
                        current_speaker_id = None
                        speaker_frame_count = min_speaker_frames  # Allow immediate switch
                        speaker_lost_count = 0
                
                # Use enhanced weighted history with time decay for better prediction
                if weighted_history:
                    current_frame_time = frame_count
                    total_weight = 0
                    weighted_sum = 0
                    
                    for frame_time, pos, base_weight in weighted_history:
                        # Calculate time-based weight decay
                        # More recent positions have higher weight, with exponential decay
                        time_diff = current_frame_time - frame_time
                        time_factor = max(0.1, math.exp(-time_diff / (fps * 0.5)))  # Half-life of 0.5 seconds
                        
                        # Calculate final weight combining base weight and time factor
                        weight = base_weight * time_factor
                        
                        weighted_sum += pos * weight
                        total_weight += weight
                    
                    # Calculate predicted position
                    if total_weight > 0:
                        predicted_center_x = weighted_sum / total_weight
                    else:
                        # Fallback to simple average if weights are too small
                        predicted_center_x = sum(pos for _, pos, _ in weighted_history) / len(weighted_history)
                    
                    # Add predicted position to history with reduced weight
                    # This helps maintain continuity while acknowledging it's a prediction
                    weighted_history.append((frame_count, predicted_center_x, 0.5))  # Half weight for predictions
                    if len(weighted_history) > max_weighted_history:
                        weighted_history.pop(0)
                    
                    # Apply very high smoothing for predicted positions to prevent jitter
                    # Higher smoothing when speaker has been lost longer
                    prediction_smoothing = min(0.99, smoothing_factor + (speaker_lost_count / max_speaker_lost_frames) * 0.05)
                    center_x = int(prediction_smoothing * prev_center_x + (1 - prediction_smoothing) * predicted_center_x)
                    
                    # Gradually reduce camera velocity when using predictions
                    # This creates a natural "settling" effect when the speaker is lost
                    camera_velocity *= 0.95
                    
                elif detection_history:
                    # Fallback to simpler history-based prediction if no weighted history
                    total_weight = 0
                    weighted_sum = 0

                    for i, pos in enumerate(detection_history):
                        # Exponential weighting - more recent detections have higher weight
                        weight = (i + 1) ** 2.5  # Increased exponent for stronger recency bias
                        weighted_sum += pos * weight
                        total_weight += weight

                    avg_center_x = weighted_sum / total_weight if total_weight > 0 else sum(detection_history) / len(detection_history)

                    # Apply enhanced smoothing with the weighted average
                    center_x = int(smoothing_factor * prev_center_x + (1 - smoothing_factor) * avg_center_x)
                
                # Update previous center
                prev_center_x = center_x
            else:
                # If no person detected and no history, use the previous center
                # Gradually slow down any existing camera movement
                camera_velocity *= 0.9
                center_x = prev_center_x + int(camera_velocity)
                prev_center_x = center_x

            # Enhanced crop boundary calculation with cinematic edge handling
            # Calculate initial crop boundaries
            half_width = crop_width // 2
            
            # Apply camera velocity for smoother movement
            center_x += int(camera_velocity)
            
            # Calculate crop boundaries
            left = center_x - half_width
            right = center_x + half_width

            # Enhanced edge handling with cubic easing for professional look
            edge_buffer = int(crop_width * 0.12)  # Increased from 5% to 12% for smoother edge transitions
            
            # Create a soft boundary near edges
            if left < edge_buffer:
                # Calculate how far into the buffer zone we are (0 to 1)
                edge_progress = left / edge_buffer if left > 0 else 0
                
                # Apply cubic easing for professional edge handling
                # This creates a natural deceleration as the camera approaches the edge
                edge_ease = edge_progress ** 3  # Cubic easing for smoother approach
                
                # Calculate adjusted position with easing
                adjusted_left = int(edge_buffer * edge_ease)
                
                # Apply adjustment
                left = adjusted_left
                right = left + crop_width
                
                # Reduce camera velocity near edges
                camera_velocity *= 0.8
                
            elif right > (width - edge_buffer):
                # Calculate how far into the buffer zone we are (0 to 1)
                edge_progress = (width - right) / edge_buffer if right < width else 0
                
                # Apply cubic easing for professional edge handling
                edge_ease = edge_progress ** 3  # Cubic easing for smoother approach
                
                # Calculate adjusted position with easing
                adjusted_right = width - int(edge_buffer * edge_ease)
                
                # Apply adjustment
                right = adjusted_right
                left = right - crop_width
                
                # Reduce camera velocity near edges
                camera_velocity *= 0.8

            # Final boundary check to ensure we're within frame
            if left < 0:
                left = 0
                right = crop_width
                camera_velocity = 0  # Stop camera movement at boundary
            elif right > width:
                right = width
                left = width - crop_width
                camera_velocity = 0  # Stop camera movement at boundary

            # Crop the frame
            cropped_frame = frame[:, left:right]

            # Ensure the cropped frame has the correct dimensions
            if cropped_frame.shape[1] != crop_width:
                # Use high-quality interpolation for resizing
                cropped_frame = cv2.resize(cropped_frame, (crop_width, height),
                                          interpolation=cv2.INTER_CUBIC)

            # Write the frame
            out.write(cropped_frame)

        # Release resources
        video.release()
        out.release()

        # Convert the OpenCV output to a proper MP4 with audio using MoviePy
        print("Converting output and adding audio...")
        temp_cv2_output = temp_output_path
        final_output_path = output_path

        # Load the original video to get the audio
        original_clip = VideoFileClip(video_path)

        # Load the processed video
        processed_clip = VideoFileClip(temp_cv2_output)

        # Add the original audio to the processed video
        if original_clip.audio is not None:
            processed_clip = processed_clip.set_audio(original_clip.audio)

        # Write the final video with professional quality settings
        processed_clip.write_videofile(
            final_output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='40000k',  # Increased bitrate for higher quality
            preset='veryslow' if professional_camera else 'medium',  # Slowest preset for best quality
            threads=4,
            ffmpeg_params=[
                '-crf', '15',  # Lower CRF for higher quality (was 17)
                '-profile:v', 'high',  # Use high profile for better quality
                '-level', '4.2',  # Compatible level for most devices
                '-movflags', '+faststart',  # Optimize for web streaming
                '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
                # Add denoising for smoother gradients (helps with compression artifacts)
                '-vf', 'hqdn3d=4:3:6:4'  # Subtle denoise settings
            ]
        )

        # Clean up
        original_clip.close()
        processed_clip.close()

        # Remove the temporary file
        if os.path.exists(temp_cv2_output):
            os.remove(temp_cv2_output)
            
        return final_output_path
        
    except Exception as e:
        print(f"Error in apply_dynamic_cropping_tracking: {str(e)}")
        traceback.print_exc()
        return None

# Original apply_dynamic_cropping function (keeping for compatibility)
def apply_dynamic_cropping(video_path, output_path, crop_params, fps):
    """
    Apply dynamic cropping to video to keep the active speaker centered.
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        crop_params: Dictionary mapping frame indices to crop parameters
        fps: Frames per second of the video
        
    Returns:
        Path to the cropped video
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    # Extract audio
    audio_path = os.path.join(temp_dir, "audio.aac")
    subprocess.run([
        "ffmpeg", "-i", video_path, 
        "-vn", "-acodec", "copy", 
        audio_path
    ], check=True)
    
    # Process video frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in crop_params:
            params = crop_params[frame_idx]
            x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
            
            # Crop the frame
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Verify dimensions are even
            h, w = cropped_frame.shape[:2]
            if w % 2 != 0 or h % 2 != 0:
                # Adjust dimensions to be even
                new_w = w - (w % 2)
                new_h = h - (h % 2)
                if new_w > 0 and new_h > 0:
                    cropped_frame = cropped_frame[:new_h, :new_w]
                    print(f"Adjusted frame dimensions from {w}x{h} to {new_w}x{new_h}")
            
            # Save the cropped frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, cropped_frame)
        
        frame_idx += 1
        
        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")
    
    cap.release()
    
    # Check the dimensions of the first frame to ensure they're even
    first_frame_path = os.path.join(temp_dir, "frame_000000.jpg")
    if os.path.exists(first_frame_path):
        first_frame = cv2.imread(first_frame_path)
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            print(f"First frame dimensions: {w}x{h}")
            if w % 2 != 0 or h % 2 != 0:
                print(f"WARNING: Dimensions are not even: {w}x{h}")
    
    # Combine frames into video
    frames_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
    subprocess.run([
        "ffmpeg", "-r", str(fps), 
        "-i", frames_pattern, 
        "-i", audio_path, 
        "-c:v", "libx264", "-crf", "23", 
        "-preset", "medium", 
        "-c:a", "copy", 
        "-shortest", 
        "-pix_fmt", "yuv420p",  # Explicitly set pixel format
        output_path
    ], check=True)
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    return output_path

# Main function to process video with speaker tracking
def process_video_with_speaker_tracking(video_path, output_path, professional_camera=True):
    """
    Process video with speaker tracking and diarization using professional cinematography techniques.
    
    This function applies advanced camera movement algorithms to create smooth, 
    professional-looking tracking that follows active speakers like a skilled cameraman.
    It includes features like anticipatory movement, cinematic transitions, and
    natural acceleration/deceleration curves.
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        professional_camera: Whether to use professional camera movement (default: True)
                            When True, applies enhanced smoothing, cinematic transitions,
                            and professional composition techniques.
        
    Returns:
        Path to the processed video
    """
    try:
        # Initialize models
        yolo_model, diarization_pipeline = initialize_models()
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate dimensions for 9:16 aspect ratio
        target_ratio = 9 / 16
        crop_width = min(width, int(height * target_ratio))
        
        # If the video is already narrower than 9:16, we don't need to crop
        if width <= crop_width:
            print("Video is already narrower than 9:16, no need for speaker tracking")
            return None
        
        # Extract audio for speaker diarization
        audio_path = extract_audio(video_path)
        if not audio_path:
            print("Failed to extract audio, continuing with visual tracking only")
            active_speaker_segments = None
        else:
            # Perform speaker diarization
            active_speaker_segments = perform_diarization(audio_path, diarization_pipeline)
            if not active_speaker_segments:
                print("Failed to perform diarization, continuing with visual tracking only")
                active_speaker_segments = None
        
        # Apply dynamic cropping using the tracking.py approach
        processed_video = apply_dynamic_cropping_tracking(
            video_path, 
            output_path, 
            active_speaker_segments, 
            fps, 
            professional_camera=professional_camera
        )
        
        return processed_video
        
    except Exception as e:
        print(f"Error in process_video_with_speaker_tracking: {str(e)}")
        traceback.print_exc()
        return None