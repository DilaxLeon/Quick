import os
import sys
import tempfile
from moviepy.editor import VideoClip, TextClip, CompositeVideoClip, ImageClip
from moviepy.config import change_settings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from app import create_caption_clip

# Configure MoviePy to use the correct ImageMagick path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# Create necessary directories
# Use absolute path to the frontend/public/previews directory
# This ensures previews are saved in the correct location regardless of where the script is run from
# The frontend directory is at the same level as the App directory, not inside it
PROJECT_ROOT = 'd:/Quickcap v9'
PREVIEW_FOLDER = os.path.join(PROJECT_ROOT, 'frontend', 'public', 'previews')
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

# Define video dimensions for 9:16 aspect ratio
WIDTH = 480
HEIGHT = 854

# Duration of each preview in seconds (shorter for faster loading)
DURATION = 5

# Sample text for previews
SAMPLE_TEXTS = [
    "This is a caption template",
    "Perfect for your videos",
    "Easy to customize",
    "Boost engagement",
    "Improve accessibility",
    "Increase watch time",
    "Great for social media",
    "Works on all platforms",
    "Professional captions",
    "Stand out from the crowd",
    "Grab attention",
    "Highlight key points",
    "Make videos pop",
    "Trendy styles",
    "Modern designs"
]

def create_black_background(size, duration):
    """Create a black background clip"""
    def make_frame(t):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return img
    
    return VideoClip(make_frame, duration=duration)

# Removed static preview generation function as per requirements

def generate_template_preview(template_id, template_name):
    """Generate a preview video for a specific template"""
    print(f"Generating preview for {template_name} ({template_id})...")
    
    # Create a black background clip
    background = create_black_background((WIDTH, HEIGHT), DURATION)
    
    # Create caption clips for each sample text
    caption_clips = []
    
    # Special handling for Word by Word templates
    if template_id in ["yellow_impact", "bold_white", "bold_green"]:
        # Use a single sentence and show words one by one
        sentence = "This is a word by word caption style for your videos"
        words = sentence.split()
        
        # Calculate timing for each word
        word_duration = 0.3  # Duration for each word
        total_words = len(words)
        
        for i, word in enumerate(words):
            start_time = i * word_duration
            
            # Create caption clip for this word with larger font
            caption = create_caption_clip(
                text=word,
                video_size=(WIDTH, HEIGHT),
                duration=word_duration,
                font_size=65,  # Larger font for better visibility
                template=template_id
            )
            
            # Set the start time for this word
            caption = caption.set_start(start_time)
            # Position at 60% of the vertical screen height
            position_y = int(HEIGHT * 0.6)
            caption = caption.set_position(('center', position_y))
            
            caption_clips.append(caption)
    else:
        # Standard template preview with phrase-by-phrase captions
        # Use a faster transition between captions (0.5 second per caption)
        caption_duration = 0.5
        
        # Select a subset of texts to fit within the duration
        max_captions = min(len(SAMPLE_TEXTS), int(DURATION / caption_duration))
        selected_texts = SAMPLE_TEXTS[:max_captions]
        
        for i, text in enumerate(selected_texts):
            start_time = i * caption_duration
            
            # Create caption clip
            caption = create_caption_clip(
                text=text,
                video_size=(WIDTH, HEIGHT),
                duration=caption_duration,
                font_size=40,
                template=template_id
            )
            
            # Set the start time for this caption
            caption = caption.set_start(start_time)
            
            caption_clips.append(caption)
    
    # Combine background and captions
    final_clip = CompositeVideoClip([background] + caption_clips, size=(WIDTH, HEIGHT))
    
    # Define output path for WebM
    webm_path = os.path.join(PREVIEW_FOLDER, f"{template_id}.webm")
    
    # Write directly to WebM format
    final_clip.write_videofile(
        webm_path,
        codec="libvpx",
        audio=False,
        fps=15,
        ffmpeg_params=[
            "-b:v", "150k",
            "-crf", "30",
            "-deadline", "realtime",
            "-cpu-used", "8",
            "-vf", "scale=200:-1:flags=lanczos"
        ]
    )
    
    print(f"Preview generated: {webm_path}")
    return webm_path

def generate_all_previews():
    """Generate previews for all templates"""
    # Template categories and their templates (copied from TemplateSelector.jsx)
    template_categories = [
        {
            "name": 'Word by Word',
            "templates": [
                { "id": 'yellow_impact', "name": 'Word by Word Yellow' },
                { "id": 'bold_white', "name": 'Word by Word White' },
                { "id": 'bold_green', "name": 'Word by Word Green' },
            ]
        },
        {
            "name": 'Premium',
            "templates": [
                { "id": 'prism_fusion', "name": 'Prism Fusion' },
                { "id": 'neon_heartbeat', "name": 'Neon Heartbeat' },
                { "id": 'premium_yellow', "name": 'Premium Yellow' },
                { "id": 'premium_orange', "name": 'Premium Orange' },
            ]
        },
        {
            "name": 'Classic/Minimal',
            "templates": [
                { "id": 'minimal_white', "name": 'Minimal White' },
                { "id": 'elegant_pink', "name": 'Elegant Pink' },
            ]
        },
        {
            "name": 'Entertainment/Vlogger',
            "templates": [
                { "id": 'creator_highlight', "name": 'Creator Highlight' },
                { "id": 'bold_sunshine', "name": 'Bold Sunshine' },
                { "id": 'mrbeast', "name": 'MrBeast' },
                { "id": 'creator_clean', "name": 'Creator Clean' },
                { "id": 'reaction_pop', "name": 'Reaction Pop' },
                { "id": 'bold_switch', "name": 'Bold Switch' },
            ]
        },
        {
            "name": 'Social Media',
            "templates": [
                { "id": 'tiktok', "name": 'TikTok' },
                { "id": 'insta_story', "name": 'Insta Story' },
                { "id": 'blue_highlight', "name": 'Blue Highlight' },
            ]
        },
        {
            "name": 'Educational/Informative',
            "templates": [

                { "id": 'explainer_pro', "name": 'Explainer Pro' },
                { "id": 'science_journal', "name": 'Science Journal' },
            ]
        },
        {
            "name": 'Gaming',
            "templates": [
                { "id": 'streamer_pro', "name": 'Streamer Pro' },
                { "id": 'esports_caption', "name": 'Esports Caption' },
                { "id": 'gaming_neon', "name": 'Gaming Neon' },
            ]
        },
        {
            "name": 'Cinematic/Film',
            "templates": [
                { "id": 'film_noir', "name": 'Film Noir' },
                { "id": 'cinematic_quote', "name": 'Cinematic Quote' },
                { "id": 'cinematic_futura', "name": 'Cinematic Futura' },
            ]
        },
        {
            "name": 'Comedy/Memes',
            "templates": [
                { "id": 'meme_orange', "name": 'Meme Orange' },
            ]
        },
        {
            "name": 'Trendy/Viral',
            "templates": [
                { "id": 'viral_pop', "name": 'Viral Pop' },

                { "id": 'green_bold', "name": 'Green Bold' },
                { "id": 'trendy_gradient', "name": 'Trendy Gradient' },
            ]
        },
    ]
    
    # Generate previews for all templates
    generated_previews = []
    for category in template_categories:
        for template in category["templates"]:
            try:
                preview_path = generate_template_preview(template["id"], template["name"])
                generated_previews.append({
                    "id": template["id"],
                    "name": template["name"],
                    "category": category["name"],
                    "preview_path": preview_path
                })
            except Exception as e:
                print(f"Error generating preview for {template['name']}: {e}")
    
    print(f"Generated {len(generated_previews)} template previews")
    return generated_previews

if __name__ == "__main__":
    # Generate the Word by Word template previews
    try:
        print("\nGenerating preview for Word by Word Yellow...")
        generate_template_preview('yellow_impact', 'Word by Word Yellow')
    except Exception as e:
        print(f"Error generating preview for Word by Word Yellow: {e}")
        
    try:
        print("\nGenerating preview for Word by Word White...")
        generate_template_preview('bold_white', 'Word by Word White')
    except Exception as e:
        print(f"Error generating preview for Word by Word White: {e}")
    
    # Generate preview for our new Cinematic Futura template
    try:
        print("\nGenerating preview for Cinematic Futura...")
        generate_template_preview('cinematic_futura', 'Cinematic Futura')
    except Exception as e:
        print(f"Error generating preview for Cinematic Futura: {e}")
        
    # Generate preview for Neon Heartbeat with reduced glow
    try:
        print("\nGenerating preview for Neon Heartbeat...")
        generate_template_preview('neon_heartbeat', 'Neon Heartbeat')
    except Exception as e:
        print(f"Error generating preview for Neon Heartbeat: {e}")
        
    # Generate preview for Neon Pulse with reduced glow and enhanced light effect
    try:
        print("\nGenerating preview for Neon Pulse...")
        generate_template_preview('neon_pulse', 'Neon Pulse')
    except Exception as e:
        print(f"Error generating preview for Neon Pulse: {e}")
        
    # Generate preview for Blue Highlight template
    try:
        print("\nGenerating preview for Blue Highlight...")
        generate_template_preview('blue_highlight', 'Blue Highlight')
    except Exception as e:
        print(f"Error generating preview for Blue Highlight: {e}")
        

    # Generate all previews in WebM format
    try:
        print("\nGenerating all template previews in WebM format...")
        generate_all_previews()
    except Exception as e:
        print(f"Error generating previews: {e}")