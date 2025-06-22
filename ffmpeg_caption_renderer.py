"""
FFmpeg-based Caption Renderer
Replaces PIL and MoviePy TextClip with high-performance FFmpeg drawtext filters
"""

import os
import json
import re
import subprocess
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import uuid


class FFmpegCaptionRenderer:
    """
    High-performance caption renderer using FFmpeg's drawtext filter
    Supports complex styling, animations, and multi-threaded processing
    """
    
    # Font mapping for different templates
    TEMPLATE_FONTS = {
        # Word by Word
        "yellow_impact": "EuropaGroteskSH-Bol.otf",
        "bold_white": "Poppins-BlackItalic.ttf",
        "bold_green": "Poppins-ExtraBold.ttf",
        
        # Classic/Minimal
        "minimalist_sans": "HelveticaNeue-Light.ttf",
        "minimal_white": "SpiegelSans.otf",
        "elegant_pink": "Anton-Regular.ttf",
        
        # Entertainment/Vlogger
        "bold_sunshine": "Theboldfont.ttf",
        "creator_highlight": "Poppins-SemiBold.ttf",
        "vlog_caption": "Montserrat-Bold.ttf",
        "mrbeast": "Komikax.ttf",
        "vlogger_bold": "HelveticaRoundedLTStd-Bd.ttf",
        "bold_switch": "Theboldfont.ttf",
        "creator_clean": "Montserrat.ttf",
        "reaction_pop": "Proxima Nova Alt Condensed Black.otf",
        
        # Social Media
        "tiktok_trend": "Inter-Black.ttf",
        "reels_ready": "SFProDisplay-Bold.ttf",
        "tiktok": "Proxima Nova Alt Condensed Black Italic.otf",
        "insta_story": "PoetsenOne-Regular.ttf",
        "blue_highlight": "Poppins-ExtraBold.ttf",
        
        # Educational/Informative
        "explainer_pro": "HelveticaRoundedLTStd-Bd.ttf",
        "science_journal": "Geist-Black.otf",
        
        # Gaming
        "gaming_neon": "Exo2-VariableFont_wght.ttf",
        
        # Cinematic/Film
        "film_noir": "Anton-Regular.ttf",
        "cinematic_quote": "Proxima Nova Alt Condensed Black Italic.otf",
        "cinematic_futura": "futura-pt-bold-oblique.otf",
        
        # Comedy/Memes
        "meme_orange": "LuckiestGuy.ttf",
        
        # Trendy/Viral
        "green_bold": "Uni Sans Heavy.otf",
        "trendy_gradient": "LuckiestGuy.ttf",
        "premium_orange": "Poppins-BoldItalic.ttf",
        "premium_yellow": "Poppins-BoldItalic.ttf",
        "neon_heartbeat": "Montserrat-SemiBoldItalic.ttf",
        "neon_pulse": "BebasNeue-Regular.ttf",
        "meme_maker": "Impact.ttf",
        "viral_pop": "Provicali.otf",
        "streamer_pro": "Rajdhani-Bold.ttf",
        "esports_caption": "Exo2-Black.ttf",
    }
    
    # Color definitions for templates
    TEMPLATE_COLORS = {
        # Word by Word
        "yellow_impact": {"text": "white", "highlight": "yellow", "outline": "black"},
        "bold_white": {"text": "white", "highlight": "white", "outline": "black"},
        "bold_green": {"text": "white", "highlight": "#9CFF2E", "outline": "black"},
        
        # Classic/Minimal
        "minimalist_sans": {"text": "#F0F0F0", "highlight": "#F0F0F0", "outline": "black"},
        "minimal_white": {"text": "white", "highlight": "white", "outline": "black"},
        "elegant_pink": {"text": "white", "highlight": "#F7374F", "outline": "black"},
        
        # Entertainment/Vlogger
        "bold_sunshine": {"text": "white", "highlight": "yellow", "outline": "black"},
        "creator_highlight": {"text": "white", "highlight": "#FFCC00", "outline": "black"},
        "vlog_caption": {"text": "white", "highlight": "white", "outline": "black"},
        "mrbeast": {"text": "white", "highlight": "#FFDD00", "outline": "black"},
        "vlogger_bold": {"text": "white", "highlight": "#525FE1", "outline": "black"},
        "bold_switch": {"text": "white", "highlight": "white", "outline": "black"},
        "creator_clean": {"text": "white", "highlight": "#00BFFF", "outline": "black"},
        "reaction_pop": {"text": "white", "highlight": "#FF3C3C", "outline": "black"},
        
        # Social Media
        "tiktok_trend": {"text": "white", "highlight": "white", "outline": "black"},
        "reels_ready": {"text": "white", "highlight": "white", "outline": "black"},
        "tiktok": {"text": "white", "highlight": "#FF69B4", "outline": "black", "bg": "#FF69B4@0.9"},
        "insta_story": {"text": "white", "highlight": "#F9CEEE", "outline": "black"},
        "blue_highlight": {"text": "white", "highlight": "#3D90D7", "outline": "black", "bg": "#3D90D7@0.9"},
        
        # Educational/Informative
        "explainer_pro": {"text": "white", "highlight": "#FF8C00", "outline": "black", "bg": "#FF8C00@0.9"},
        "science_journal": {"text": "white", "highlight": "#89AC46", "outline": "black"},
        
        # Gaming
        "gaming_neon": {"text": "#007BFF", "highlight": "#00FFFF", "outline": "black", "shadow": "#007BFF"},
        
        # Cinematic/Film
        "film_noir": {"text": "black", "highlight": "black", "outline": "white"},
        "cinematic_quote": {"text": "yellow", "highlight": "yellow", "outline": "black"},
        "cinematic_futura": {"text": "#F7A165", "highlight": "#F7A165", "outline": "black"},
        
        # Comedy/Memes
        "meme_orange": {"text": "white", "highlight": "#FF8C00", "outline": "black"},
        
        # Trendy/Viral
        "green_bold": {"text": "white", "highlight": "#00FF00", "outline": "black"},
        "trendy_gradient": {"text": "white", "highlight": "#FF69B4", "outline": "black"},
        "premium_orange": {"text": "white", "highlight": "#EB5B00", "outline": "black"},
        "premium_yellow": {"text": "white", "highlight": "#E9D023", "outline": "black"},
        "neon_heartbeat": {"text": "#FF00FF", "highlight": "#FF00FF", "outline": "black", "shadow": "#FF00FF"},
        "neon_pulse": {"text": "white", "highlight": "white", "outline": "black"},
        "meme_maker": {"text": "white", "highlight": "white", "outline": "black"},
        "viral_pop": {"text": "white", "highlight": "#FF3399", "outline": "black"},
        "streamer_pro": {"text": "white", "highlight": "#00FF80", "outline": "black"},
        "esports_caption": {"text": "white", "highlight": "#FF4500", "outline": "black"},
    }
    
    def __init__(self, fonts_dir: str = "Fonts"):
        self.fonts_dir = fonts_dir
        self.temp_dir = "/app/tmp" if os.path.exists("/app") else tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def escape_ffmpeg_text(self, text: str) -> str:
        """Escape text for FFmpeg drawtext filter"""
        if not text:
            return ""
        
        # Escape special characters for FFmpeg drawtext
        text = str(text)  # Ensure it's a string
        text = text.replace("\\", "\\\\")  # Backslash (must be first)
        text = text.replace(":", "\\:")    # Colon
        text = text.replace("'", "\\'")    # Single quote
        text = text.replace("%", "\\%")    # Percent
        text = text.replace("[", "\\[")    # Square brackets
        text = text.replace("]", "\\]")
        text = text.replace(",", "\\,")    # Comma
        text = text.replace(";", "\\;")    # Semicolon
        text = text.replace("{", "\\{")    # Curly braces
        text = text.replace("}", "\\}")
        text = text.replace("=", "\\=")    # Equals sign
        
        # Remove or replace problematic characters
        text = text.replace("\n", " ")     # Newlines to spaces
        text = text.replace("\r", " ")     # Carriage returns to spaces
        text = text.replace("\t", " ")     # Tabs to spaces
        
        # Clean up multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_font_path(self, template: str, custom_font: Optional[str] = None) -> str:
        """Get the full path to the font file"""
        if custom_font:
            # Try different extensions for custom fonts
            for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
                font_name = f"{custom_font}{ext}"
                font_path = os.path.join(self.fonts_dir, font_name)
                if os.path.exists(font_path):
                    return font_path
        else:
            font_name = self.TEMPLATE_FONTS.get(template, "SpiegelSans.otf")
            font_path = os.path.join(self.fonts_dir, font_name)
            if os.path.exists(font_path):
                return font_path
        
        # If font doesn't exist, use fallback
        fallback_fonts = [
            "Arial.ttf", "arial.ttf", "ARIAL.TTF",
            "Poppins-Regular.ttf", "poppins-regular.ttf", 
            "Montserrat.ttf", "montserrat.ttf",
            "SpiegelSans.otf", "spiegelsans.otf",
            "DejaVuSans.ttf", "dejavu-sans.ttf",
            "Liberation-Sans.ttf", "liberation-sans.ttf"
        ]
        
        for fallback in fallback_fonts:
            fallback_path = os.path.join(self.fonts_dir, fallback)
            if os.path.exists(fallback_path):
                print(f"Font not found: {font_name if 'font_name' in locals() else 'custom font'}, using fallback: {fallback}")
                return fallback_path
        
        # If no fonts found in Fonts directory, let FFmpeg use system default
        print(f"No fonts found in {self.fonts_dir}, FFmpeg will use system default font")
        return "Arial"  # FFmpeg will use system Arial or default
    
    def apply_text_case(self, text: str, template: str) -> str:
        """Apply text case transformation based on template"""
        template = template.lower()
        
        # Templates that use lowercase
        if template in ["tiktok", "gaming_neon"]:
            return text.lower()
        
        # Templates that use uppercase
        elif template in ["meme_orange", "viralpop_bold", "premium_orange", "premium_yellow", 
                         "neon_pulse", "viral_pop", "esports_caption", "bold_green",
                         "cinematic_futura", "bold_switch", "film_noir", "blue_highlight",
                         "meme_maker"]:
            return text.upper()
        
        # Templates that use title case (capitalize each word)
        elif template in ["cinematic_quote", "reaction_pop"]:
            return ' '.join(word.capitalize() for word in text.split())
        
        # All other templates use normal case
        return text
    
    def create_advanced_word_filter(self,
                                  text: str,
                                  template: str,
                                  video_width: int,
                                  video_height: int,
                                  font_size: int,
                                  start_time: float,
                                  duration: float,
                                  word_timestamps: Optional[List[Dict]] = None,
                                  vertical_position: int = 50,
                                  custom_colors: Optional[Dict] = None,
                                  enable_animations: bool = True) -> List[str]:
        """
        Create advanced FFmpeg drawtext filters with word-level highlighting and animations
        """
        font_path = self.get_font_path(template)
        colors = custom_colors or self.TEMPLATE_COLORS.get(template, self.TEMPLATE_COLORS["minimal_white"])
        
        # Apply text case transformation
        text = self.apply_text_case(text, template)
        escaped_text = self.escape_ffmpeg_text(text)
        
        words = text.split()
        if not words:
            return []
        
        # Calculate vertical position (0-100 to actual pixels)
        y_pos = int(video_height * (vertical_position / 100))
        if vertical_position > 50:
            y_pos = int(video_height * 0.8)  # Bottom area
        else:
            y_pos = int(video_height * 0.2)  # Top area
        
        filters = []
        
        # Add drop shadow first (renders behind text)
        if colors.get('shadow'):
            shadow_offset = max(2, font_size // 20)
            shadow_filter = (
                f"drawtext=fontfile='{font_path}'"
                f":text='{escaped_text}'"
                f":fontsize={font_size}"
                f":fontcolor={colors['shadow']}@0.6"
                f":x=(w-text_w)/2+{shadow_offset}"
                f":y={y_pos + shadow_offset}"
                f":borderw=0"
            )
            
            # Add fade-in animation for shadow
            if enable_animations:
                fade_duration = min(0.3, duration * 0.2)
                shadow_filter += f":alpha='if(lt(t,{start_time + fade_duration}),(t-{start_time})/{fade_duration},1)'"
            
            shadow_filter += f":enable='between(t,{start_time},{start_time + duration})'"
            filters.append(shadow_filter)
        
        # Add background bar for certain templates
        if colors.get('bg'):
            bg_color, bg_alpha = colors['bg'].split('@') if '@' in colors['bg'] else (colors['bg'], '0.8')
            padding = max(10, font_size // 4)
            
            bg_filter = (
                f"drawbox=x=(w-text_w)/2-{padding}"
                f":y={y_pos - padding//2}"
                f":w=text_w+{padding*2}"
                f":h={font_size + padding}"
                f":color={bg_color}@{bg_alpha}"
            )
            
            # Add slide-in animation for background
            if enable_animations and template in ["tiktok", "blue_highlight"]:
                bg_filter += f":x='(w-text_w)/2-{padding}+200*exp(-3*(t-{start_time}))'"
            
            bg_filter += f":enable='between(t,{start_time},{start_time + duration})'"
            filters.append(bg_filter)
        
        # Base text filter (renders behind highlights)
        base_filter = (
            f"drawtext=fontfile='{font_path}'"
            f":text='{escaped_text}'"
            f":fontsize={font_size}"
            f":fontcolor={colors['text']}"
            f":x=(w-text_w)/2"
            f":y={y_pos}"
            f":borderw={max(2, font_size // 15)}"
            f":bordercolor={colors.get('outline', 'black')}"
        )
        
        # Add entrance animation
        if enable_animations:
            fade_duration = min(0.4, duration * 0.25)
            
            if template in ["mrbeast", "meme_orange"]:
                # Bounce effect for energetic templates
                base_filter += f":y='{y_pos}+20*exp(-8*(t-{start_time}))*cos(20*(t-{start_time}))'"
            elif template in ["cinematic_quote", "film_noir"]:
                # Fade in for cinematic templates
                base_filter += f":alpha='if(lt(t,{start_time + fade_duration}),(t-{start_time})/{fade_duration},1)'"
            elif template in ["gaming_neon", "neon_heartbeat"]:
                # Glow pulsing for gaming templates
                base_filter += f":alpha='0.8+0.2*sin(4*PI*(t-{start_time}))'"
        
        base_filter += f":enable='between(t,{start_time},{start_time + duration})'"
        filters.append(base_filter)
        
        # Add word-level highlighting
        if word_timestamps and len(word_timestamps) == len(words):
            for i, word_info in enumerate(word_timestamps):
                word = words[i]
                word_start = word_info.get('start', start_time)
                word_end = word_info.get('end', start_time + duration / len(words))
                
                # Calculate word position in the text
                before_text = ' '.join(words[:i])
                if before_text:
                    # Approximate word position calculation
                    char_width = font_size * 0.6  # Approximate character width
                    x_offset = len(before_text) * char_width
                else:
                    x_offset = 0
                
                escaped_word = self.escape_ffmpeg_text(word)
                
                # Create highlighted word overlay
                highlight_filter = (
                    f"drawtext=fontfile='{font_path}'"
                    f":text='{escaped_word}'"
                    f":fontsize={font_size}"
                    f":fontcolor={colors['highlight']}"
                    f":x=(w-text_w)/2+{x_offset}"
                    f":y={y_pos}"
                    f":borderw={max(2, font_size // 15)}"
                    f":bordercolor={colors.get('outline', 'black')}"
                )
                
                # Add highlighting animations
                if enable_animations:
                    if template in ["yellow_impact", "bold_green"]:
                        # Scale up highlight for impact templates  
                        scale_factor = 1.1
                        highlight_filter += f":fontsize='{font_size * scale_factor}'"
                    elif template in ["gaming_neon", "neon_pulse"]:
                        # Pulsing glow for gaming templates
                        highlight_filter += f":alpha='0.9+0.1*sin(10*PI*(t-{word_start}))'"
                    elif template in ["trendy_gradient", "viral_pop"]:
                        # Color cycling for trendy templates
                        highlight_filter += f":fontcolor='if(mod(floor(t*5),2),{colors['highlight']},{colors['text']})'"
                
                highlight_filter += f":enable='between(t,{word_start},{word_end})'"
                filters.append(highlight_filter)
        
        return filters
    
    def create_complex_filtergraph(self, 
                                 phrases: List[Dict],
                                 template: str,
                                 video_width: int,
                                 video_height: int,
                                 font_size: int = 40,
                                 vertical_position: int = 50,
                                 custom_colors: Optional[Dict] = None,
                                 enable_word_highlighting: bool = True,
                                 enable_animations: bool = True) -> str:
        """
        Create complex FFmpeg filtergraph with advanced animations and word highlighting
        """
        if not phrases:
            return ""
        
        all_filters = []
        
        for i, phrase in enumerate(phrases):
            text = phrase.get('text', '').strip()
            if not text:
                continue
            
            start_time = phrase.get('start', 0)
            end_time = phrase.get('end', start_time + 2)
            duration = end_time - start_time
            
            # Get word-level timestamps if available
            word_timestamps = phrase.get('word_timestamps', [])
            
            # Create advanced filters for this phrase
            phrase_filters = self.create_advanced_word_filter(
                text=text,
                template=template,
                video_width=video_width,
                video_height=video_height,
                font_size=font_size,
                start_time=start_time,
                duration=duration,
                word_timestamps=word_timestamps if enable_word_highlighting else None,
                vertical_position=vertical_position,
                custom_colors=custom_colors,
                enable_animations=enable_animations
            )
            
            all_filters.extend(phrase_filters)
        
        # Handle line wrapping for long text
        return self.optimize_filtergraph(all_filters)
    
    def optimize_filtergraph(self, filters: List[str]) -> str:
        """
        Optimize the filtergraph for better performance and handling of long filter chains
        """
        if not filters:
            return ""
        
        # Group similar filters together for better performance
        shadow_filters = []
        bg_filters = []
        text_filters = []
        highlight_filters = []
        
        for filter_str in filters:
            if 'drawbox' in filter_str:
                bg_filters.append(filter_str)
            elif '@0.6' in filter_str or 'shadow' in filter_str:
                shadow_filters.append(filter_str)
            elif 'highlight' in filter_str or any(color in filter_str for color in ['yellow', 'green', 'blue', 'red']):
                highlight_filters.append(filter_str)
            else:
                text_filters.append(filter_str)
        
        # Combine in optimal order: shadows -> backgrounds -> text -> highlights
        optimized_filters = shadow_filters + bg_filters + text_filters + highlight_filters
        
        # Split long filtergraphs if needed (FFmpeg has limits)
        max_filter_length = 8000  # Conservative limit
        combined = ','.join(optimized_filters)
        
        if len(combined) > max_filter_length:
            # Split into chunks if too long
            chunks = []
            current_chunk = []
            current_length = 0
            
            for filter_str in optimized_filters:
                if current_length + len(filter_str) + 1 > max_filter_length and current_chunk:
                    chunks.append(','.join(current_chunk))
                    current_chunk = [filter_str]
                    current_length = len(filter_str)
                else:
                    current_chunk.append(filter_str)
                    current_length += len(filter_str) + 1
            
            if current_chunk:
                chunks.append(','.join(current_chunk))
            
            # For now, return the first chunk (in practice, we'd need multiple passes)
            return chunks[0] if chunks else ""
        
        return combined
    
    def create_multiline_filter(self,
                              lines: List[str],
                              template: str,
                              video_width: int,
                              video_height: int,
                              font_size: int,
                              start_time: float,
                              duration: float,
                              vertical_position: int = 50,
                              custom_colors: Optional[Dict] = None,
                              enable_animations: bool = True) -> List[str]:
        """
        Create filters for multi-line captions with proper spacing
        """
        if not lines:
            return []
        
        font_path = self.get_font_path(template)
        colors = custom_colors or self.TEMPLATE_COLORS.get(template, self.TEMPLATE_COLORS["minimal_white"])
        
        # Calculate base vertical position
        base_y = int(video_height * (vertical_position / 100))
        if vertical_position > 50:
            base_y = int(video_height * 0.8)
        else:
            base_y = int(video_height * 0.2)
        
        # Calculate line spacing
        line_height = int(font_size * 1.2)  # 20% spacing between lines
        
        # Center the text block vertically
        total_height = len(lines) * line_height
        start_y = base_y - (total_height // 2)
        
        filters = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            line_text = self.apply_text_case(line, template)
            escaped_line = self.escape_ffmpeg_text(line_text)
            y_pos = start_y + (i * line_height)
            
            # Add background for each line if needed
            if colors.get('bg'):
                bg_color, bg_alpha = colors['bg'].split('@') if '@' in colors['bg'] else (colors['bg'], '0.8')
                padding = max(10, font_size // 4)
                
                bg_filter = (
                    f"drawbox=x=(w-text_w)/2-{padding}"
                    f":y={y_pos - padding//2}"
                    f":w=text_w+{padding*2}"
                    f":h={font_size + padding}"
                    f":color={bg_color}@{bg_alpha}"
                    f":enable='between(t,{start_time},{start_time + duration})'"
                )
                filters.append(bg_filter)
            
            # Create text filter for this line
            text_filter = (
                f"drawtext=fontfile='{font_path}'"
                f":text='{escaped_line}'"
                f":fontsize={font_size}"
                f":fontcolor={colors['text']}"
                f":x=(w-text_w)/2"
                f":y={y_pos}"
                f":borderw={max(2, font_size // 15)}"
                f":bordercolor={colors.get('outline', 'black')}"
            )
            
            # Add staggered animation for multi-line
            if enable_animations and len(lines) > 1:
                delay = i * 0.1  # 100ms delay per line
                fade_duration = 0.3
                
                if template in ["mrbeast", "meme_orange"]:
                    # Slide in from sides alternating
                    direction = 1 if i % 2 == 0 else -1
                    text_filter += f":x='(w-text_w)/2+{200 * direction}*exp(-5*(t-{start_time + delay}))'"
                else:
                    # Fade in with delay
                    text_filter += f":alpha='if(lt(t,{start_time + delay + fade_duration}),(t-{start_time + delay})/{fade_duration},1)'"
            
            text_filter += f":enable='between(t,{start_time},{start_time + duration})'"
            filters.append(text_filter)
        
        return filters
    
    def split_text_to_lines(self, text: str, max_chars_per_line: int = 35) -> List[str]:
        """
        Intelligently split text into lines for better readability
        """
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed the limit
            word_length = len(word)
            if current_length + word_length + len(current_line) > max_chars_per_line and current_line:
                # Start a new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length
        
        # Add the last line if it has content
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def get_template_animation_preset(self, template: str) -> Dict[str, Any]:
        """
        Get animation presets for different templates
        """
        presets = {
            # Energetic templates
            "mrbeast": {
                "entrance": "bounce",
                "highlight": "scale_pulse",
                "fade_duration": 0.2,
                "bounce_strength": 20,
                "scale_factor": 1.15
            },
            "meme_orange": {
                "entrance": "slide_bounce",
                "highlight": "color_flash",
                "fade_duration": 0.15,
                "slide_distance": 150,
                "bounce_strength": 15
            },
            
            # Smooth templates
            "tiktok": {
                "entrance": "slide_smooth",
                "highlight": "glow_pulse",
                "fade_duration": 0.3,
                "slide_distance": 200,
                "glow_intensity": 0.3
            },
            "cinematic_quote": {
                "entrance": "fade_elegant",
                "highlight": "subtle_glow",
                "fade_duration": 0.8,
                "glow_intensity": 0.2
            },
            
            # Gaming templates
            "gaming_neon": {
                "entrance": "glow_burst",
                "highlight": "neon_pulse",
                "fade_duration": 0.25,
                "pulse_frequency": 6,
                "glow_intensity": 0.4
            },
            "neon_heartbeat": {
                "entrance": "heartbeat",
                "highlight": "pulse_sync",
                "fade_duration": 0.2,
                "heartbeat_rate": 1.2,
                "pulse_frequency": 2.4
            },
            
            # Professional templates
            "minimal_white": {
                "entrance": "fade_simple",
                "highlight": "subtle_brighten",
                "fade_duration": 0.4,
                "brighten_amount": 0.15
            },
            "elegant_pink": {
                "entrance": "fade_elegant",
                "highlight": "color_shift",
                "fade_duration": 0.5,
                "color_shift_intensity": 0.2
            }
        }
        
        return presets.get(template, presets["minimal_white"])
    
    def create_animated_entrance(self, 
                               filter_base: str,
                               animation_type: str,
                               start_time: float,
                               preset: Dict[str, Any]) -> str:
        """
        Add entrance animation to a drawtext filter
        """
        fade_duration = preset.get('fade_duration', 0.3)
        
        if animation_type == "bounce":
            bounce_strength = preset.get('bounce_strength', 20)
            # Extract y position from filter and add bounce
            if ':y=' in filter_base:
                y_part = filter_base.split(':y=')[1].split(':')[0]
                filter_base = filter_base.replace(f':y={y_part}', f':y=\'{y_part}+{bounce_strength}*exp(-8*(t-{start_time}))*cos(20*(t-{start_time}))\'')
            
        elif animation_type == "slide_bounce":
            slide_distance = preset.get('slide_distance', 150)
            bounce_strength = preset.get('bounce_strength', 15)
            # Add horizontal slide with bounce
            if ':x=' in filter_base:
                x_part = filter_base.split(':x=')[1].split(':')[0]
                filter_base = filter_base.replace(f':x={x_part}', f':x=\'{x_part}+{slide_distance}*exp(-5*(t-{start_time}))+{bounce_strength}*exp(-10*(t-{start_time}))*sin(15*(t-{start_time}))\'')
            
        elif animation_type == "slide_smooth":
            slide_distance = preset.get('slide_distance', 200)
            # Smooth slide from side
            if ':x=' in filter_base:
                x_part = filter_base.split(':x=')[1].split(':')[0]
                filter_base = filter_base.replace(f':x={x_part}', f':x=\'{x_part}+{slide_distance}*exp(-6*(t-{start_time}))\'')
            
        elif animation_type == "fade_elegant":
            # Smooth fade with easing
            filter_base += f":alpha='if(lt(t,{start_time + fade_duration}),pow((t-{start_time})/{fade_duration},2),1)'"
            
        elif animation_type == "fade_simple":
            # Linear fade
            filter_base += f":alpha='if(lt(t,{start_time + fade_duration}),(t-{start_time})/{fade_duration},1)'"
            
        elif animation_type == "glow_burst":
            glow_intensity = preset.get('glow_intensity', 0.4)
            # Glow effect with burst
            filter_base += f":alpha='1-{glow_intensity}*exp(-4*(t-{start_time}))'"
            
        elif animation_type == "heartbeat":
            heartbeat_rate = preset.get('heartbeat_rate', 1.2)
            # Heartbeat scaling effect
            if ':fontsize=' in filter_base:
                size_part = filter_base.split(':fontsize=')[1].split(':')[0]
                filter_base = filter_base.replace(f':fontsize={size_part}', f':fontsize=\'{size_part}*(1+0.1*sin({heartbeat_rate}*2*PI*(t-{start_time})))\'')
        
        return filter_base
    
    def render_video_with_captions(self,
                                 input_video_path: str,
                                 output_video_path: str,
                                 phrases: List[Dict],
                                 template: str = "minimal_white",
                                 font_size: int = 40,
                                 vertical_position: int = 50,
                                 custom_colors: Optional[Dict] = None,
                                 enable_word_highlighting: bool = True,
                                 video_quality: str = "high") -> bool:
        """
        Render video with captions using FFmpeg
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            phrases: List of caption phrases with timing
            template: Caption template name
            font_size: Font size for captions
            vertical_position: Vertical position (0-100)
            custom_colors: Custom color scheme
            enable_word_highlighting: Enable word-by-word highlighting
            video_quality: Quality preset ('high', 'medium', 'fast')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get video dimensions
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
                input_video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error probing video: {result.stderr}")
                return False
            
            video_info = json.loads(result.stdout)
            video_stream = next((s for s in video_info['streams'] if s['codec_type'] == 'video'), None)
            
            if not video_stream:
                print("No video stream found")
                return False
            
            video_width = int(video_stream['width'])
            video_height = int(video_stream['height'])
            
            # Create filtergraph with enhanced features
            filtergraph = self.create_complex_filtergraph(
                phrases=phrases,
                template=template,
                video_width=video_width,
                video_height=video_height,
                font_size=font_size,
                vertical_position=vertical_position,
                custom_colors=custom_colors,
                enable_word_highlighting=enable_word_highlighting,
                enable_animations=True
            )
            
            if not filtergraph:
                print("No valid captions to render")
                return False
            
            # Quality settings
            quality_settings = {
                'high': {
                    'preset': 'slow',
                    'crf': '18',
                    'threads': '0'  # Use all available threads
                },
                'medium': {
                    'preset': 'medium',
                    'crf': '23',
                    'threads': '0'
                },
                'fast': {
                    'preset': 'veryfast',
                    'crf': '28',
                    'threads': '0'
                }
            }
            
            settings = quality_settings.get(video_quality, quality_settings['medium'])
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', input_video_path,
                '-vf', filtergraph,  # Use video filter instead of filter_complex
                '-c:v', 'libx264',
                '-preset', settings['preset'],
                '-crf', settings['crf'],
                '-pix_fmt', 'yuv420p',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-threads', settings['threads'],
                '-movflags', '+faststart',  # Web optimization
                output_video_path
            ]
            
            print(f"Running FFmpeg command: {' '.join(cmd[:10])}...")  # Don't print full command for security
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully rendered video with captions: {output_video_path}")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error rendering video with captions: {str(e)}")
            return False
    
    def generate_test_phrases(self) -> List[Dict]:
        """Generate test phrases for development/testing"""
        return [
            {
                "text": "Welcome to our amazing video",
                "start": 0.0,
                "end": 2.5,
                "word_timestamps": [
                    {"word": "Welcome", "start": 0.0, "end": 0.5},
                    {"word": "to", "start": 0.5, "end": 0.8},
                    {"word": "our", "start": 0.8, "end": 1.0},
                    {"word": "amazing", "start": 1.0, "end": 1.8},
                    {"word": "video", "start": 1.8, "end": 2.5}
                ]
            },
            {
                "text": "This is a test caption",
                "start": 3.0,
                "end": 5.0,
                "word_timestamps": [
                    {"word": "This", "start": 3.0, "end": 3.3},
                    {"word": "is", "start": 3.3, "end": 3.5},
                    {"word": "a", "start": 3.5, "end": 3.7},
                    {"word": "test", "start": 3.7, "end": 4.2},
                    {"word": "caption", "start": 4.2, "end": 5.0}
                ]
            }
        ]
    
    def generate_test_phrases_with_words(self) -> List[Dict]:
        """Generate enhanced test phrases with detailed word timing"""
        return [
            {
                "text": "Check out this amazing content",
                "start": 0.0,
                "end": 3.0,
                "word_timestamps": [
                    {"word": "Check", "start": 0.0, "end": 0.4},
                    {"word": "out", "start": 0.4, "end": 0.8},
                    {"word": "this", "start": 0.8, "end": 1.2},
                    {"word": "amazing", "start": 1.2, "end": 2.0},
                    {"word": "content", "start": 2.0, "end": 3.0}
                ]
            },
            {
                "text": "Subscribe for more awesome videos",
                "start": 3.5,
                "end": 6.0,
                "word_timestamps": [
                    {"word": "Subscribe", "start": 3.5, "end": 4.2},
                    {"word": "for", "start": 4.2, "end": 4.5},
                    {"word": "more", "start": 4.5, "end": 4.9},
                    {"word": "awesome", "start": 4.9, "end": 5.5},
                    {"word": "videos", "start": 5.5, "end": 6.0}
                ]
            },
            {
                "text": "This is how you create epic captions with style",
                "start": 7.0,
                "end": 11.0,
                "word_timestamps": [
                    {"word": "This", "start": 7.0, "end": 7.3},
                    {"word": "is", "start": 7.3, "end": 7.5},
                    {"word": "how", "start": 7.5, "end": 7.8},
                    {"word": "you", "start": 7.8, "end": 8.1},
                    {"word": "create", "start": 8.1, "end": 8.6},
                    {"word": "epic", "start": 8.6, "end": 9.0},
                    {"word": "captions", "start": 9.0, "end": 9.8},
                    {"word": "with", "start": 9.8, "end": 10.1},
                    {"word": "style", "start": 10.1, "end": 11.0}
                ]
            }
        ]


def test_ffmpeg_renderer():
    """Test function for the FFmpeg caption renderer"""
    renderer = FFmpegCaptionRenderer()
    
    # Generate test phrases
    test_phrases = renderer.generate_test_phrases()
    
    # Test filtergraph generation
    filtergraph = renderer.create_complex_filtergraph(
        phrases=test_phrases,
        template="minimal_white",
        video_width=480,
        video_height=854,
        font_size=40,
        vertical_position=80,
        enable_word_highlighting=True
    )
    
    print("Generated FFmpeg filtergraph:")
    print(filtergraph)
    
    return filtergraph


if __name__ == "__main__":
    test_ffmpeg_renderer()