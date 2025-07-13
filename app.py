from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance
import os
from io import BytesIO
import numpy as np
import cv2
from rembg import remove, new_session
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import logging

# for edge handling : 

from PIL import ImageChops,ImageFilter

def fade_edges(image: Image.Image, blur_radius=2) -> Image.Image:
    """Fade edges of the alpha channel slightly."""
    if image.mode != 'RGBA':
        return image
    r, g, b, a = image.split()
    a_blurred = a.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.merge('RGBA', (r, g, b, a_blurred))


# Placeholder for missing modules (to be implemented or approximated)
class EfficientBackbone(nn.Module):
    def __init__(self):
        super(EfficientBackbone, self).__init__()
        # Simplified placeholder for efficientnet-b0 (1280 channels at output)
        self.conv = nn.Conv2d(4, 1280, kernel_size=3, padding=1)  # 4 channels input
        self.relu = nn.ReLU()

    def forward(self, fg, bg):
        # Dummy forward pass to mimic multi-scale features
        x = self.relu(self.conv(fg))  # Process fg only for now
        return x, x, x, x, x  # Return 5 dummy feature maps

class CascadeArgumentRegressor(nn.Module):
    def __init__(self, in_channels, mid_channels, stages, out_channels):
        super(CascadeArgumentRegressor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels)
        )

    def forward(self, x):
        x = self.pool(x)  # [1, 1280, 256, 256] -> [1, 1280, 1, 1]
        x = x.view(x.size(0), -1)  # [1, 1280]
        return self.layers(x)

class FilterPerformer(nn.Module):
    def __init__(self, filter_types):
        super(FilterPerformer, self).__init__()
        self.filter_types = filter_types

    def restore(self, comp, mask, arguments, orig_size):
        # Apply amplified adjustments at original size
        adjusted = comp
        for arg, ftype in zip(arguments, self.filter_types):
            arg_value = arg.item() * 2.0  # Amplify adjustments for visibility
            if ftype == "TEMPERATURE":  # Adjust color temperature (simplified)
                adjusted = tf.adjust_hue(adjusted, arg_value * 0.1)
            elif ftype == "BRIGHTNESS":
                adjusted = tf.adjust_brightness(adjusted, 1 + arg_value * 0.5)
            elif ftype == "CONTRAST":
                adjusted = tf.adjust_contrast(adjusted, 1 + arg_value * 0.5)
            elif ftype == "SATURATION":
                adjusted = tf.adjust_saturation(adjusted, 1 + arg_value * 0.5)
            elif ftype == "HIGHLIGHT" or ftype == "SHADOW":
                # Simplified: adjust brightness for highlights/shadows
                adjusted = tf.adjust_brightness(adjusted, 1 + arg_value * 0.2)
        # Upscale with bicubic interpolation for better quality
        adjusted = F.interpolate(adjusted, size=orig_size, mode='bicubic', align_corners=False)
        return adjusted

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define Harmonizer model
class Harmonizer(nn.Module):
    def __init__(self):
        super(Harmonizer, self).__init__()
        self.filter_types = [
            "TEMPERATURE",
            "BRIGHTNESS",
            "CONTRAST",
            "SATURATION",
            "HIGHLIGHT",
            "SHADOW",
        ]

        self.backbone = EfficientBackbone()
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        self.performer = FilterPerformer(self.filter_types)

    def predict_arguments(self, comp, mask):
        # Convert and prepare tensors
        comp_tensor = tf.to_tensor(comp.convert('RGB')).unsqueeze(0)  # [1, 3, H, W]
        mask_tensor = tf.to_tensor(mask.convert('L')).unsqueeze(0)     # [1, 1, H, W]
        orig_size = comp_tensor.shape[2:]  # Preserve original size
        logger.info(f"Initial comp_tensor shape: {comp_tensor.shape}, mask_tensor shape: {mask_tensor.shape}")

        # Resize to model input size
        # Resize to higher input size to reduce blurring
        target_size = (720, 720)  # or even (640, 640)
        comp_tensor = F.interpolate(comp_tensor, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        mask_tensor = F.interpolate(mask_tensor, size=target_size, mode='bilinear', align_corners=False)


        # Create 4-channel inputs
        fg = torch.cat((comp_tensor, mask_tensor), dim=1)  # [1, 4, 256, 256]
        bg = torch.cat((comp_tensor, 1 - mask_tensor), dim=1)  # [1, 4, 256, 256]
        logger.info(f"fg shape before backbone: {fg.shape}, bg shape before backbone: {bg.shape}")

        # Pass through backbone
        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(fg, bg)
        logger.info(f"enc32x shape: {enc32x.shape}")
        arguments = self.regressor(enc32x)
        return arguments, orig_size

    def restore_image(self, comp, mask, arguments):
        # Convert and prepare tensors, ensuring 3-channel comp
        comp_tensor = tf.to_tensor(comp.convert('RGB')).unsqueeze(0)  # [1, 3, H, W]
        mask_tensor = tf.to_tensor(mask.convert('L')).unsqueeze(0)     # [1, 1, H, W]
        orig_size = comp_tensor.shape[2:]  # Preserve original size
        logger.info(f"comp_tensor shape in restore: {comp_tensor.shape}, mask_tensor shape in restore: {mask_tensor.shape}")

        # Resize to model input size for processing
        comp_tensor = F.interpolate(comp_tensor, size=(720, 720), mode='bilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=(720, 720), mode='bilinear', align_corners=False)

        # Ensure arguments are in the correct format
        arguments = arguments.chunk(len(self.filter_types), dim=1)  # Split into individual arguments
        adjusted = self.performer.restore(comp_tensor, mask_tensor, arguments, orig_size)
        return adjusted

# Initialize Harmonizer model with pre-trained weights
HARMONIZER_MODEL_PATH = 'Harmonizer/pretrained/harmonizer.pth'
harmonizer = None
try:
    harmonizer = Harmonizer()
    if os.path.exists(HARMONIZER_MODEL_PATH):
        state_dict = torch.load(HARMONIZER_MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        harmonizer.load_state_dict(new_state_dict, strict=False)
        harmonizer.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        harmonizer.to(device)
        logger.info("Harmonizer model loaded successfully from pretrained/harmonizer.pth.")
    else:
        logger.warning("Pre-trained Harmonizer model not found. Using untrained model.")
        harmonizer.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        harmonizer.to(device)
except Exception as e:
    logger.error(f"Failed to initialize harmonizer: {str(e)}. Harmonization will use fallback.")

def refine_alpha_mask(alpha_mask_pil):
    """
    Applies a slight Gaussian blur to the alpha mask for smoother edges.
    """
    alpha_np = np.array(alpha_mask_pil)
    blurred_alpha_np = cv2.GaussianBlur(alpha_np, (5, 5), 1)
    return Image.fromarray(blurred_alpha_np)

def remove_background_with_session(image, session):
    """
    Removes the background from an image using a specific rembg session.
    Ensures non-zero dimensions by avoiding empty crops.
    """
    image = image.convert("RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    input_bytes = buffer.getvalue()
    output_bytes = remove(input_bytes, session=session)
    result_image = Image.open(BytesIO(output_bytes)).convert("RGBA")
    *rgb_channels, alpha_channel = result_image.split()
    refined_alpha_channel = refine_alpha_mask(alpha_channel)
    result_image = Image.merge('RGBA', (*rgb_channels, refined_alpha_channel))
    bbox = result_image.getbbox()
    if bbox:
        cropped_image = result_image.crop(bbox)
        # Ensure minimum size if cropped result is too small
        if cropped_image.size[0] == 0 or cropped_image.size[1] == 0:
            logger.warning("Cropped image has zero dimensions; using original size with transparent background.")
            new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            new_image.paste(result_image, bbox)
            return new_image
        return cropped_image
    else:
        logger.warning("No valid bounding box found; using original size with transparent background.")
        return Image.new('RGBA', image.size, (0, 0, 0, 0)).paste(result_image, (0, 0))

def create_composite_mask(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Creates a binary mask for harmonization based on the alpha channels of person1 and person2.
    """
    bg_width, bg_height = background_img.size
    mask = Image.new('L', (bg_width, bg_height), 0)
    person1_alpha = person1_img.split()[3]
    # Ensure minimum dimensions
    w1 = max(1, int(w1))
    h1 = max(1, int(h1))
    person1_resized = person1_alpha.resize((w1, h1), Image.Resampling.LANCZOS)
    mask.paste(person1_resized, (int(x1), int(y1)), person1_resized)
    person2_alpha = person2_img.split()[3]
    w2 = max(1, int(w2))
    h2 = max(1, int(h2))
    person2_resized = person2_alpha.resize((w2, h2), Image.Resampling.LANCZOS)
    mask.paste(person2_resized, (int(x2), int(y2)), person2_resized)
    mask_np = np.array(mask)
    binary_mask = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_mask)

def harmonize_image(image, mask, harmonizer_model):
    """
    Applies harmonization using the Harmonizer model.
    """
    if harmonizer_model is None:
        logger.warning("Harmonizer model not loaded; applying fallback adjustment.")
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)
    try:
        with torch.no_grad():
            arguments, orig_size = harmonizer_model.predict_arguments(image, mask)
            harmonized_img = harmonizer_model.restore_image(image, mask, arguments)
            harmonized_img = tf.to_pil_image(harmonized_img.squeeze(0))  # Convert tensor to PIL image
        logger.info(f"Harmonization applied with arguments: {arguments}, original size: {orig_size}")
        return harmonized_img
    except Exception as e:
        logger.error(f"Harmonization failed with exception: {str(e)}")
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)

def merge_images(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2):
    """Merge person images onto background at given positions/sizes."""
    person1_resized = person1_img.resize((int(w1), int(h1)), Image.Resampling.LANCZOS)
    person2_resized = person2_img.resize((int(w2), int(h2)), Image.Resampling.LANCZOS)
    result = background_img.copy()
    result.paste(person1_resized, (int(x1), int(y1)), person1_resized)
    result.paste(person2_resized, (int(x2), int(y2)), person2_resized)
    return result

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        logger.info("Starting image processing...")
        person1_file = request.files.get('person1')
        person2_file = request.files.get('person2')
        background_file = request.files.get('background')

        if not all([person1_file, person2_file, background_file]):
            return jsonify({'error': 'All three images are required'}), 400

        person1_path = os.path.join(UPLOAD_FOLDER, 'person1_no_bg.png')
        person2_path = os.path.join(UPLOAD_FOLDER, 'person2_no_bg.png')
        background_path = os.path.join(UPLOAD_FOLDER, 'background.png')
        composite_path = os.path.join(UPLOAD_FOLDER, 'composite.png')

        for path in [person1_path, person2_path, background_path, composite_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Error removing old file {path}: {e}")

        person1_img = Image.open(person1_file).convert("RGBA")
        person2_img = Image.open(person2_file).convert("RGBA")
        background_img = Image.open(background_file).convert("RGBA")

        session1 = new_session(model_name="u2net_human_seg")
        session2 = new_session(model_name="u2net_human_seg")

        logger.info("Removing background from Person 1...")
        person1_no_bg = remove_background_with_session(person1_img, session1)
        if person1_no_bg.size[0] == 0 or person1_no_bg.size[1] == 0:
            raise ValueError("Person 1 image has zero width or height after background removal")

        logger.info("Removing background from Person 2...")
        person2_no_bg = remove_background_with_session(person2_img, session2)
        if person2_no_bg.size[0] == 0 or person2_no_bg.size[1] == 0:
            raise ValueError("Person 2 image has zero width or height after background removal")

        bg_width, bg_height = background_img.size
        max_person_width = bg_width * 0.4
        max_person_height = bg_height * 0.6

        p1_width, p1_height = person1_no_bg.size
        p1_scale = min(max_person_width / p1_width, max_person_height / p1_height, 1)
        w1 = int(p1_width * p1_scale)
        h1 = int(p1_height * p1_scale)
        if w1 <= 0 or h1 <= 0:
            raise ValueError("Person 1 scaled dimensions are invalid (w1 or h1 <= 0)")

        p2_width, p2_height = person2_no_bg.size
        p2_scale = min(max_person_width / p2_width, max_person_height / p2_height, 1)
        w2 = int(p2_width * p2_scale)
        h2 = int(p2_height * p2_scale)
        if w2 <= 0 or h2 <= 0:
            raise ValueError("Person 2 scaled dimensions are invalid (w2 or h2 <= 0)")
        person1_no_bg = fade_edges(remove_background_with_session(person1_img, session1))
        person2_no_bg = fade_edges(remove_background_with_session(person2_img, session2))

        person1_no_bg.save(person1_path, 'PNG')
        person2_no_bg.save(person2_path, 'PNG')
        background_img.save(background_path, 'PNG')

        x1 = int((bg_width / 2) - (w1 / 2) - (w1 / 4))
        y1 = int(bg_height - h1 - 20)
        x2 = int((bg_width / 2) - (w2 / 2) + (w2 / 4))
        y2 = int(bg_height - h2 - 20)

        composite = merge_images(
            person1_no_bg, person2_no_bg, background_img,
            x1, y1, w1, h1, x2, y2, w2, h2
        )
        composite.save(composite_path, 'PNG')

        logger.info("Image processing and merging completed!")

        return jsonify({
            'person1_url': f'/{person1_path}',
            'person2_url': f'/{person2_path}',
            'background_url': f'/{background_path}',
            'composite_url': f'/{composite_path}',
            'initial_x1': x1, 'initial_y1': y1, 'initial_w1': w1, 'initial_h1': h1,
            'initial_x2': x2, 'initial_y2': y2, 'initial_w2': w2, 'initial_h2': h2,
            'bg_width': bg_width, 'bg_height': bg_height
        })

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/adjust', methods=['POST'])
def adjust_images():
    try:
        logger.info("Adjusting image composition...")
        data = request.get_json()
        x1 = int(data.get('x1', 50))
        y1 = int(data.get('y1', 50))
        w1 = int(data.get('w1', 100))
        h1 = int(data.get('h1', 100))
        x2 = int(data.get('x2', 200))
        y2 = int(data.get('y2', 50))
        w2 = int(data.get('w2', 100))
        h2 = int(data.get('h2', 100))

        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            raise ValueError("Invalid dimensions for adjustment (w1, h1, w2, or h2 <= 0)")

        person1_img = Image.open('static/uploads/person1_no_bg.png').convert("RGBA")
        person2_img = Image.open('static/uploads/person2_no_bg.png').convert("RGBA")
        background_img = Image.open('static/uploads/background.png').convert("RGBA")
        composite_path = os.path.join(UPLOAD_FOLDER, 'composite.png')

        if os.path.exists(composite_path):
            try:
                os.remove(composite_path)
            except PermissionError:
                logger.warning("Could not delete composite.png due to file lock. This might cause issues.")

        composite = merge_images(person1_img, person2_img, background_img,
                                 x1, y1, w1, h1, x2, y2, w2, h2)
        composite.save(composite_path, 'PNG')

        logger.info("Image adjustment completed!")

        return jsonify({'composite_url': f'/{composite_path}'})
    except Exception as e:
        logger.error(f"Error during adjustment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance_composite():
    try:
        logger.info("Processing final composite image...")
        composite_path = os.path.join(UPLOAD_FOLDER, 'composite.png')
        final_path = os.path.join(UPLOAD_FOLDER, 'enhanced_composite.png')

        if not os.path.exists(composite_path):
            logger.error("Composite image not found")
            return jsonify({'error': 'Composite image not found'}), 400

        composite_img = Image.open(composite_path).convert("RGBA")
        data = request.get_json()
        x1 = int(data.get('x1', 50))
        y1 = int(data.get('y1', 50))
        w1 = int(data.get('w1', 100))
        h1 = int(data.get('h1', 100))
        x2 = int(data.get('x2', 200))
        y2 = int(data.get('y2', 50))
        w2 = int(data.get('w2', 100))
        h2 = int(data.get('h2', 100))

        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            raise ValueError("Invalid dimensions for enhancement (w1, h1, w2, or h2 <= 0)")

        if os.path.exists(final_path):
            try:
                os.remove(final_path)
            except PermissionError:
                logger.warning("Could not delete enhanced_composite.png due to file lock. This might cause issues.")

        composite_img.save(final_path, 'PNG')
        logger.info("Final image processed!")

        return jsonify({
            'enhanced_url': f'/{final_path}',
            'message': 'Image processed. Would you like to apply harmonization now?'  # Updated message
        })
    except Exception as e:
        logger.error(f"Error during final processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply_harmonizer', methods=['POST'])
def apply_harmonizer():
    try:
        logger.info("Applying harmonization to enhanced image...")
        enhanced_path = os.path.join(UPLOAD_FOLDER, 'enhanced_composite.png')
        harmonized_path = os.path.join(UPLOAD_FOLDER, 'harmonized_composite.png')
        person1_img = Image.open('static/uploads/person1_no_bg.png').convert("RGBA")
        person2_img = Image.open('static/uploads/person2_no_bg.png').convert("RGBA")
        background_img = Image.open('static/uploads/background.png').convert("RGBA")
        data = request.get_json()
        x1 = int(data.get('x1', 50))
        y1 = int(data.get('y1', 50))
        w1 = int(data.get('w1', 100))
        h1 = int(data.get('h1', 100))
        x2 = int(data.get('x2', 200))
        y2 = int(data.get('y2', 50))
        w2 = int(data.get('w2', 100))
        h2 = int(data.get('h2', 100))

        # Ensure minimum dimensions
        w1 = max(1, w1)
        h1 = max(1, h1)
        w2 = max(1, w2)
        h2 = max(1, h2)

        if not os.path.exists(enhanced_path):
            logger.error("Enhanced image not found")
            return jsonify({'error': 'Enhanced image not found'}), 400

        enhanced_img = Image.open(enhanced_path).convert("RGBA")

        # Step 1: Harmonize person1 with background
        logger.info("Creating mask for person1...")
        mask1 = create_composite_mask(person1_img, Image.new('RGBA', background_img.size), background_img, x1, y1, w1, h1, 0, 0, 0, 0)
        logger.info("Harmonizing person1 with background...")
        harmonized_img1 = harmonize_image(enhanced_img, mask1, harmonizer)

        # Step 2: Use harmonized result as new background and add person2
        logger.info("Creating mask for person2...")
        mask2 = create_composite_mask(person2_img, Image.new('RGBA', background_img.size), harmonized_img1, x2, y2, w2, h2, 0, 0, 0, 0)
        logger.info("Harmonizing person2 with updated background...")
        harmonized_img2 = harmonize_image(harmonized_img1, mask2, harmonizer)

        from PIL import ImageEnhance

        # Boost color saturation slightly (1.0 = original, >1 = more vibrant)
        color_enhancer = ImageEnhance.Color(harmonized_img2)
        harmonized_img2 = color_enhancer.enhance(1.2)  # Try 1.1 to 1.5 for stronger boost


        if os.path.exists(harmonized_path):
            try:
                os.remove(harmonized_path)
            except PermissionError:
                logger.warning("Could not delete harmonized_composite.png due to file lock. This might cause issues.")

        from PIL import ImageFilter

        harmonized_img2 = harmonized_img2.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        harmonized_img2 = harmonize_image(harmonized_img1, mask2, harmonizer)

        # Color enhancement
        harmonized_img2 = ImageEnhance.Color(harmonized_img2).enhance(1.2)

        # Optional enhancements for brightness and contrast
        harmonized_img2 = ImageEnhance.Brightness(harmonized_img2).enhance(1.05)
        harmonized_img2 = ImageEnhance.Contrast(harmonized_img2).enhance(1.05)

        # Save with high quality
        harmonized_img2.save(harmonized_path, 'PNG', quality=95, dpi=(300, 300))
        logger.info("Harmonization applied! Saved to %s", harmonized_path)

        return jsonify({
            'harmonized_url': f'/{harmonized_path}'
        })
    except Exception as e:
        logger.error(f"Harmonization process failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)