import io
import numpy as np
from PIL import Image
from modules import shared

def get_live_preview_size(width, height):
    """Calculate preview dimensions based on settings"""
    if not shared.opts.live_preview_auto_scale:
        max_size = shared.opts.live_preview_size
        scale = max_size / max(width, height)
        return int(width * scale), int(height * scale)
        
    # Auto scale based on target resolution
    scale = 512 / max(width, height) 
    return int(width * scale), int(height * scale)

def create_preview_image(image, width=None, height=None):
    """Convert PIL Image or numpy array to JPEG bytes for preview"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Get target preview size
    preview_width, preview_height = get_live_preview_size(
        width or image.width,
        height or image.height
    )
    
    # Resize and convert to JPEG bytes
    preview = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    preview.save(output, format='JPEG', quality=shared.opts.live_preview_quality)
    return output.getvalue()
