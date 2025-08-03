from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter
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
from filelock import FileLock
import threading
import secrets
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import time
import uuid
import atexit

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Limit concurrent image processing
process_semaphore = threading.Semaphore(2)

# Store filenames for the session and their cleanup timers
session_files = {}
session_cleanup_timers = {}

# Load Harmonizer model once
HARMONIZER_MODEL_PATH = os.getenv('HARMONIZER_MODEL_PATH', 'Harmonizer/pretrained/harmonizer.pth')
harmonizer = None
try:
    class EfficientBackbone(nn.Module):
        def __init__(self):
            super(EfficientBackbone, self).__init__()
            self.conv = nn.Conv2d(4, 1280, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, fg, bg):
            x = self.relu(self.conv(fg))
            return x, x, x, x, x

    class CascadeArgumentRegressor(nn.Module):
        def __init__(self, in_channels, mid_channels, stages, out_channels):
            super(CascadeArgumentRegressor, self).__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.layers = nn.Sequential(
                nn.Linear(in_channels, mid_channels),
                nn.ReLU(),
                nn.Linear(mid_channels, out_channels)
            )

        def forward(self, x):
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.layers(x)

    class FilterPerformer(nn.Module):
        def __init__(self, filter_types):
            super(FilterPerformer, self).__init__()
            self.filter_types = filter_types

        def restore(self, comp, mask, arguments, orig_size):
            adjusted = comp
            selected_filters = ["BRIGHTNESS", "CONTRAST", "SATURATION"]
            for arg, ftype in zip(arguments, self.filter_types):
                if ftype in selected_filters:
                    arg_value = arg.item() * 2.0
                    if ftype == "BRIGHTNESS":
                        adjusted = tf.adjust_brightness(adjusted, 1 + arg_value * 0.5)
                    elif ftype == "CONTRAST":
                        adjusted = tf.adjust_contrast(adjusted, 1 + arg_value * 0.5)
                    elif ftype == "SATURATION":
                        adjusted = tf.adjust_saturation(adjusted, 1 + arg_value * 0.5)
            adjusted = F.interpolate(adjusted, size=orig_size, mode='bilinear', align_corners=False)
            return adjusted

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
            comp_tensor = tf.to_tensor(comp.convert('RGB')).unsqueeze(0)
            mask_tensor = tf.to_tensor(mask.convert('L')).unsqueeze(0)
            orig_size = comp_tensor.shape[2:]
            target_size = (256, 256)
            comp_tensor = F.interpolate(comp_tensor, size=target_size, mode='bilinear', align_corners=False, antialias=True)
            mask_tensor = F.interpolate(mask_tensor, size=target_size, mode='bilinear', align_corners=False)
            fg = torch.cat((comp_tensor, mask_tensor), dim=1)
            bg = torch.cat((comp_tensor, 1 - mask_tensor), dim=1)
            enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(fg, bg)
            arguments = self.regressor(enc32x)
            return arguments, orig_size

        def restore_image(self, comp, mask, arguments):
            comp_tensor = tf.to_tensor(comp.convert('RGB')).unsqueeze(0)
            mask_tensor = tf.to_tensor(mask.convert('L')).unsqueeze(0)
            orig_size = comp_tensor.shape[2:]
            comp_tensor = F.interpolate(comp_tensor, size=(2048, 2048), mode='bilinear', align_corners=False)
            mask_tensor = F.interpolate(mask_tensor, size=(2048, 2048), mode='bilinear', align_corners=False)
            arguments = arguments.chunk(len(self.filter_types), dim=1)
            adjusted = self.performer.restore(comp_tensor, mask_tensor, arguments, orig_size)
            adjusted_pil = tf.to_pil_image(adjusted.squeeze(0))
            if comp.mode == 'RGBA':
                adjusted_pil = Image.merge('RGBA', (adjusted_pil.split()[0], adjusted_pil.split()[1], adjusted_pil.split()[2], comp.split()[3]))
            return adjusted_pil

    harmonizer = Harmonizer()
    if os.path.exists(HARMONIZER_MODEL_PATH):
        state_dict = torch.load(HARMONIZER_MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        harmonizer.load_state_dict(new_state_dict, strict=False)
        harmonizer.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        harmonizer.to(device)
        logger.info("Harmonizer model loaded successfully.")
    else:
        logger.warning("Pre-trained Harmonizer model not found. Using untrained model.")
        harmonizer.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        harmonizer.to(device)
except Exception as e:
    logger.error(f"Failed to initialize harmonizer: {str(e)}. Using fallback.")
    harmonizer = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_remove(path, max_retries=3, delay=1.0):
    """Remove a file with retries."""
    lock_path = path + '.lock'
    with FileLock(lock_path):
        for attempt in range(max_retries):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted {path}")
                    return True
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Could not delete {path}: {str(e)}")
                time.sleep(delay)
        logger.error(f"Failed to delete {path} after {max_retries} attempts")
        return False

def save_with_validation(img, file_path):
    """Save image and validate integrity."""
    lock_path = file_path + '.lock'
    with FileLock(lock_path):
        img.save(file_path, 'PNG')
        try:
            with Image.open(file_path) as test_img:
                test_img.load()
            logger.info(f"Saved and validated {file_path}")
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {str(e)}")
            os.remove(file_path) if os.path.exists(file_path) else None
            raise ValueError(f"Corrupted file saved at {file_path}")

def refine_alpha_mask(alpha_mask_pil):
    alpha_np = np.array(alpha_mask_pil)
    kernel = np.ones((3, 3), np.uint8)
    eroded_alpha = cv2.erode(alpha_np, kernel, iterations=1)
    blurred_alpha_np = cv2.GaussianBlur(eroded_alpha, (5, 5), 1)
    blurred_alpha_np = np.clip(blurred_alpha_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred_alpha_np)

def remove_background_with_session(image, session):
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
        if cropped_image.size[0] == 0 or cropped_image.size[1] == 0:
            logger.warning("Cropped image has zero dimensions; using original size.")
            new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            new_image.paste(result_image, bbox)
            return new_image
        return cropped_image
    else:
        logger.warning("No valid bounding box; using original size.")
        new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        new_image.paste(result_image, (0, 0))
        return new_image

def create_composite_mask(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2):
    bg_width, bg_height = background_img.size
    mask = Image.new('L', (bg_width, bg_height), 0)
    person1_alpha = person1_img.split()[3]
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
    if harmonizer_model is None:
        logger.warning("Harmonizer model not loaded; applying fallback.")
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)
    try:
        with torch.no_grad():
            arguments, orig_size = harmonizer_model.predict_arguments(image, mask)
            harmonized_img = harmonizer_model.restore_image(image, mask, arguments)
        logger.info(f"Harmonization applied with arguments: {arguments}")
        return harmonized_img
    except Exception as e:
        logger.error(f"Harmonization failed: {str(e)}")
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)

def merge_images(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2):
    person1_resized = person1_img.resize((int(w1), int(h1)), Image.Resampling.LANCZOS)
    person2_resized = person2_img.resize((int(w2), int(h2)), Image.Resampling.LANCZOS)
    result = background_img.copy()
    result.paste(person1_resized, (int(x1), int(y1)), person1_resized.split()[3])
    result.paste(person2_resized, (int(x2), int(y2)), person2_resized.split()[3])
    return result

def cleanup_session(session_id):
    """Clean up all files associated with a specific session."""
    if session_id in session_files:
        logger.info(f"Starting cleanup for session {session_id}")
        for filename in list(session_files[session_id].values()):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                if safe_remove(file_path):
                    logger.info(f"Successfully deleted {file_path} for session {session_id}")
                else:
                    logger.warning(f"Failed to delete {file_path} for session {session_id}")
            else:
                logger.warning(f"File {file_path} not found for session {session_id}")
        del session_files[session_id]
        if session_id in session_cleanup_timers:
            session_cleanup_timers[session_id].cancel()
            del session_cleanup_timers[session_id]
        logger.info(f"Completed cleanup for session {session_id}")

def schedule_cleanup(session_id, delay_seconds=3600):
    """Schedule cleanup of session files after a delay."""
    def cleanup_task():
        time.sleep(delay_seconds)
        if session_id in session_files:
            cleanup_session(session_id)
    timer = threading.Timer(delay_seconds, cleanup_task)
    timer.daemon = True
    timer.start()
    session_cleanup_timers[session_id] = timer
    logger.info(f"Scheduled cleanup for session {session_id} in {delay_seconds} seconds")

# Register cleanup on app shutdown
@atexit.register
def cleanup_on_shutdown():
    logger.info("Shutting down, cleaning up all sessions...")
    for session_id in list(session_files.keys()):
        cleanup_session(session_id)

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    with process_semaphore:
        try:
            logger.info("Starting image processing...")
            person1_file = request.files.get('person1')
            person2_file = request.files.get('person2')
            background_file = request.files.get('background')

            if not all([person1_file, person2_file, background_file]):
                return jsonify({'error': 'All three images are required'}), 400

            for file in [person1_file, person2_file, background_file]:
                if not allowed_file(file.filename):
                    return jsonify({'error': f'Invalid file extension for {file.filename}'}), 400

            person1_filename = secure_filename(person1_file.filename)
            person2_filename = secure_filename(person2_file.filename)
            background_filename = secure_filename(background_file.filename)

            # Generate unique session ID and filenames
            session_id = str(uuid.uuid4())
            person1_filename_random = f'person1_{secrets.token_hex(8)}.png'
            person2_filename_random = f'person2_{secrets.token_hex(8)}.png'
            background_filename_random = f'background_{secrets.token_hex(8)}.png'
            composite_filename_random = f'composite_{secrets.token_hex(8)}.png'

            person1_path = os.path.join(app.config['UPLOAD_FOLDER'], person1_filename_random)
            person2_path = os.path.join(app.config['UPLOAD_FOLDER'], person2_filename_random)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename_random)
            composite_path = os.path.join(app.config['UPLOAD_FOLDER'], composite_filename_random)

            # Store filenames in session_files
            session_files[session_id] = {
                'person1': person1_filename_random,
                'person2': person2_filename_random,
                'background': background_filename_random,
                'composite': composite_filename_random
            }

            for path in [person1_path, person2_path, background_path, composite_path]:
                safe_remove(path)

            for file, name in [(person1_file, 'person1'), (person2_file, 'person2'), (background_file, 'background')]:
                file.seek(0)
                try:
                    with Image.open(file) as img:
                        img.verify()
                except Exception as e:
                    return jsonify({'error': f'Invalid {name} image: {str(e)}'}), 400
                file.seek(0)

            person1_img = Image.open(person1_file).convert("RGBA")
            person1_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            person2_img = Image.open(person2_file).convert("RGBA")
            person2_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            background_img = Image.open(background_file).convert("RGBA")
            background_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

            session1 = new_session(model_name="u2net_human_seg")
            session2 = new_session(model_name="u2net_human_seg")

            logger.info("Removing background from Person 1...")
            person1_no_bg = remove_background_with_session(person1_img, session1)
            if person1_no_bg.size[0] == 0 or person1_no_bg.size[1] == 0:
                raise ValueError("Person 1 image has zero dimensions")

            logger.info("Removing background from Person 2...")
            person2_no_bg = remove_background_with_session(person2_img, session2)
            if person2_no_bg.size[0] == 0 or person2_no_bg.size[1] == 0:
                raise ValueError("Person 2 image has zero dimensions")

            bg_width, bg_height = background_img.size
            max_person_width = bg_width * 0.4
            max_person_height = bg_height * 0.6

            p1_width, p1_height = person1_no_bg.size
            p1_scale = min(max_person_width / p1_width, max_person_height / p1_height, 1)
            w1 = int(p1_width * p1_scale)
            h1 = int(p1_height * p1_scale)
            if w1 <= 0 or h1 <= 0:
                raise ValueError("Person 1 scaled dimensions are invalid")

            p2_width, p2_height = person2_no_bg.size
            p2_scale = min(max_person_width / p2_width, max_person_height / p2_height, 1)
            w2 = int(p2_width * p2_scale)
            h2 = int(p2_height * p2_scale)
            if w2 <= 0 or h2 <= 0:
                raise ValueError("Person 2 scaled dimensions are invalid")

            save_with_validation(person1_no_bg, person1_path)
            save_with_validation(person2_no_bg, person2_path)
            save_with_validation(background_img, background_path)

            x1 = int((bg_width / 2) - (w1 / 2) - (w1 / 4))
            y1 = int(bg_height - h1 - 20)
            x2 = int((bg_width / 2) - (w2 / 2) + (w2 / 4))
            y2 = int(bg_height - h2 - 20)

            composite = merge_images(
                person1_no_bg, person2_no_bg, background_img,
                x1, y1, w1, h1, x2, y2, w2, h2
            )
            save_with_validation(composite, composite_path)

            logger.info("Image processing completed!")
            return jsonify({
                'session_id': session_id,
                'person1_url': f'/{person1_path}',
                'person2_url': f'/{person2_path}',
                'background_url': f'/{background_path}',
                'composite_url': f'/{composite_path}',
                'initial_x1': x1, 'initial_y1': y1, 'initial_w1': w1, 'initial_h1': h1,
                'initial_x2': x2, 'initial_y2': y2, 'initial_w2': w2, 'initial_h2': h2,
                'bg_width': bg_width, 'bg_height': bg_height
            })

        except Exception as e:
            logger.error(f"Error during upload: {str(e)}")
            if session_id in session_files:
                cleanup_session(session_id)
            return jsonify({'error': str(e)}), 500

@app.route('/adjust', methods=['POST'])
def adjust_images():
    with process_semaphore:
        try:
            logger.info("Adjusting image composition...")
            data = request.get_json()
            logger.info(f"Received adjust request data: {data}")
            session_id = data.get('session_id')
            if not session_id:
                raise ValueError("Missing session_id in request data")
            if session_id not in session_files:
                raise ValueError(f"Session ID {session_id} not found in session_files")

            x1 = int(data.get('x1', 50))
            y1 = int(data.get('y1', 50))
            w1 = int(data.get('w1', 100))
            h1 = int(data.get('h1', 100))
            x2 = int(data.get('x2', 200))
            y2 = int(data.get('y2', 50))
            w2 = int(data.get('w2', 100))
            h2 = int(data.get('h2', 100))

            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                raise ValueError("Invalid dimensions")

            person1_filename = session_files[session_id]['person1']
            person2_filename = session_files[session_id]['person2']
            background_filename = session_files[session_id]['background']
            composite_filename_random = f'composite_{secrets.token_hex(8)}.png'

            person1_path = os.path.join(app.config['UPLOAD_FOLDER'], person1_filename)
            person2_path = os.path.join(app.config['UPLOAD_FOLDER'], person2_filename)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            composite_path = os.path.join(app.config['UPLOAD_FOLDER'], composite_filename_random)

            # Update session_files with the new composite filename
            session_files[session_id]['composite'] = composite_filename_random

            for path in [person1_path, person2_path, background_path]:
                if not os.path.exists(path):
                    raise ValueError(f"Input file {path} not found")

            with Image.open(person1_path) as person1_img:
                person1_img = person1_img.convert("RGBA")
                person1_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(person2_path) as person2_img:
                person2_img = person2_img.convert("RGBA")
                person2_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(background_path) as background_img:
                background_img = background_img.convert("RGBA")
                background_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

            # Remove the old composite if it exists
            old_composite_path = os.path.join(app.config['UPLOAD_FOLDER'], session_files[session_id].get('composite', ''))
            if old_composite_path and os.path.exists(old_composite_path):
                safe_remove(old_composite_path)
                logger.info(f"Deleted old composite {old_composite_path} for session {session_id}")

            composite = merge_images(person1_img, person2_img, background_img,
                                    x1, y1, w1, h1, x2, y2, w2, h2)
            save_with_validation(composite, composite_path)

            logger.info("Image adjustment completed!")
            return jsonify({
                'session_id': session_id,
                'composite_url': f'/{composite_path}'
            })

        except Exception as e:
            logger.error(f"Error during adjustment: {str(e)}")
            if session_id in session_files:
                cleanup_session(session_id)
            return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance_composite():
    with process_semaphore:
        try:
            logger.info("Processing enhanced composite image...")
            data = request.get_json()
            logger.info(f"Received enhance request data: {data}")
            session_id = data.get('session_id')
            if not session_id or session_id not in session_files:
                raise ValueError("Invalid or missing session ID")

            composite_filename = session_files[session_id]['composite']
            final_filename_random = f'enhanced_composite_{secrets.token_hex(8)}.png'

            composite_path = os.path.join(app.config['UPLOAD_FOLDER'], composite_filename)
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename_random)

            session_files[session_id]['enhanced'] = final_filename_random

            if not os.path.exists(composite_path):
                raise ValueError("Composite image not found")

            with Image.open(composite_path) as composite_img:
                composite_img = composite_img.convert("RGBA")
                composite_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

            x1 = int(data.get('x1', 50))
            y1 = int(data.get('y1', 50))
            w1 = int(data.get('w1', 100))
            h1 = int(data.get('h1', 100))
            x2 = int(data.get('x2', 200))
            y2 = int(data.get('y2', 50))
            w2 = int(data.get('w2', 100))
            h2 = int(data.get('h2', 100))

            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                raise ValueError("Invalid dimensions")

            safe_remove(final_path)
            save_with_validation(composite_img, final_path)
            logger.info("Enhanced image processed!")

            # Proceed directly to harmonization
            person1_filename = session_files[session_id]['person1']
            person2_filename = session_files[session_id]['person2']
            background_filename = session_files[session_id]['background']
            harmonized_filename_random = f'harmonized_composite_{secrets.token_hex(8)}.png'

            person1_path = os.path.join(app.config['UPLOAD_FOLDER'], person1_filename)
            person2_path = os.path.join(app.config['UPLOAD_FOLDER'], person2_filename)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            harmonized_path = os.path.join(app.config['UPLOAD_FOLDER'], harmonized_filename_random)

            session_files[session_id]['harmonized'] = harmonized_filename_random

            for path in [person1_path, person2_path, background_path, final_path]:
                if not os.path.exists(path):
                    raise ValueError(f"Input file {path} not found")

            with Image.open(person1_path) as person1_img:
                person1_img = person1_img.convert("RGBA")
                person1_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(person2_path) as person2_img:
                person2_img = person2_img.convert("RGBA")
                person2_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(background_path) as background_img:
                background_img = background_img.convert("RGBA")
                background_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(final_path) as enhanced_img:
                enhanced_img = enhanced_img.convert("RGBA")
                enhanced_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

            mask = create_composite_mask(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2)
            harmonized_img = harmonize_image(enhanced_img, mask, harmonizer)

            safe_remove(harmonized_path)
            harmonized_img = harmonized_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            save_with_validation(harmonized_img, harmonized_path)
            logger.info("Harmonization applied!")

            # Schedule cleanup after 1 hour (3600 seconds)
            schedule_cleanup(session_id)

            return jsonify({
                'session_id': session_id,
                'enhanced_url': f'/{final_path}',
                'harmonized_url': f'/{harmonized_path}',
                'message': 'Enhanced and harmonized images are processed'
            })

        except Exception as e:
            logger.error(f"Error during enhancement or harmonization: {str(e)}")
            if session_id in session_files:
                cleanup_session(session_id)
            return jsonify({'error': str(e), 'message': ''}), 500

@app.route('/apply_harmonizer', methods=['POST'])
def apply_harmonizer():
    with process_semaphore:
        try:
            logger.info("Applying harmonization...")
            data = request.get_json()
            logger.info(f"Received harmonizer request data: {data}")
            session_id = data.get('session_id')
            if not session_id or session_id not in session_files:
                raise ValueError("Invalid or missing session ID")

            person1_filename = session_files[session_id]['person1']
            person2_filename = session_files[session_id]['person2']
            background_filename = session_files[session_id]['background']
            enhanced_filename = session_files[session_id].get('enhanced')
            harmonized_filename_random = f'harmonized_composite_{secrets.token_hex(8)}.png'

            person1_path = os.path.join(app.config['UPLOAD_FOLDER'], person1_filename)
            person2_path = os.path.join(app.config['UPLOAD_FOLDER'], person2_filename)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
            harmonized_path = os.path.join(app.config['UPLOAD_FOLDER'], harmonized_filename_random)

            session_files[session_id]['harmonized'] = harmonized_filename_random

            for path in [person1_path, person2_path, background_path, enhanced_path]:
                if not os.path.exists(path):
                    raise ValueError(f"Input file {path} not found")

            with Image.open(person1_path) as person1_img:
                person1_img = person1_img.convert("RGBA")
                person1_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(person2_path) as person2_img:
                person2_img = person2_img.convert("RGBA")
                person2_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(background_path) as background_img:
                background_img = background_img.convert("RGBA")
                background_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            with Image.open(enhanced_path) as enhanced_img:
                enhanced_img = enhanced_img.convert("RGBA")
                enhanced_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

            x1 = int(data.get('x1', 50))
            y1 = int(data.get('y1', 50))
            w1 = int(data.get('w1', 100))
            h1 = int(data.get('h1', 100))
            x2 = int(data.get('x2', 200))
            y2 = int(data.get('y2', 50))
            w2 = int(data.get('w2', 100))
            h2 = int(data.get('h2', 100))

            w1 = max(1, w1)
            h1 = max(1, h1)
            w2 = max(1, w2)
            h2 = max(1, h2)

            mask = create_composite_mask(person1_img, person2_img, background_img, x1, y1, w1, h1, x2, y2, w2, h2)
            harmonized_img = harmonize_image(enhanced_img, mask, harmonizer)

            safe_remove(harmonized_path)
            harmonized_img = harmonized_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            save_with_validation(harmonized_img, harmonized_path)
            logger.info("Harmonization applied!")

            # Schedule cleanup after 1 hour (3600 seconds)
            schedule_cleanup(session_id)

            return jsonify({
                'session_id': session_id,
                'harmonized_url': f'/{harmonized_path}',
                'message': 'Harmonized image is processed'
            })

        except Exception as e:
            logger.error(f"Harmonization failed: {str(e)}")
            if session_id in session_files:
                cleanup_session(session_id)
            return jsonify({'error': str(e), 'message': ''}), 500

@app.route('/cleanup', methods=['DELETE'])
def cleanup_session_endpoint():
    """Handle client-initiated cleanup request."""
    session_id = request.args.get('session_id')
    if not session_id or session_id not in session_files:
        logger.warning(f"Invalid or missing session ID for cleanup: {session_id}")
        return jsonify({'message': 'Invalid session ID'}), 400
    cleanup_session(session_id)
    return jsonify({'message': f'Session {session_id} cleaned up'}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)


