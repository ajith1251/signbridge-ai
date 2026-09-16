import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import string
import threading
import time
import uuid
import logging
import json
from collections import deque
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

app = Flask(__name__)

# Kannada news CSV location (optional examples corpus - NOT CNN training data).
# Supports both the nested folder and a sibling top-level folder layout.
_KANNADA_NEWS_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kannada_dataset'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'kannada_dataset'),
]
_KANNADA_NEWS_FILES = ['train.csv', 'valid.csv']

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Sentiment analysis pipeline
sentiment_analyzer = None
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )
except Exception as e:
    logger.error(f"Failed to load sentiment analysis model: {e}")
    sentiment_analyzer = None

# ISL detection configuration
try:
    model = load_model('model.h5')
    print('Model loaded successfully')
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise
ISL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
cap = None
current_prediction = "Prediction: None"
prediction_history = deque(maxlen=5)
word = []
last_prediction = None
last_time = time.time()
lock = threading.Lock()
COMMON_WORDS = ['HELLO', 'THANK', 'YOU', 'GOOD', 'BYE']
predicted_sentence = ''
NAVIGATION_GESTURES = {'N': 'next', 'B': 'back', 'S': 'submit'}

# --- Structured prediction state (Phase 4) -------------------------------
# Populated directly from CNN inference inside generate_frames(); read by
# GET /api/prediction. Never parsed back from the human-readable string.
ACCEPT_CONFIDENCE = 0.70   # acceptance threshold for a "confident" sign
LOW_CONFIDENCE = 0.45      # below this: treated as noise / hold steady

prediction_state = {
    'status': 'ready',            # ready|detecting|no_hand|low_confidence|camera_off|error
    'letter': None,               # accepted letter or None
    'confidence': 0.0,            # 0-100 (top-1 confidence from the CNN)
    'top_predictions': [],        # [{letter, confidence}, ...] top-3
    'word_buffer': '',            # letters accumulated so far
    'sentence': '',               # completed COMMON word, if any
    'timestamp': None,            # epoch seconds of last update
    'message': 'Camera ready — press Start Camera to begin.',
}


def _update_prediction_state(status, letter=None, confidence=0.0, top=None,
                             message=None):
    """Write one snapshot of recognition state under the global lock."""
    prediction_state.update({
        'status': status,
        'letter': letter,
        'confidence': round(float(confidence) * 100.0, 1),
        'top_predictions': top or [],
        'word_buffer': ''.join(word),
        'sentence': predicted_sentence,
        'timestamp': time.time(),
    })
    if message:
        prediction_state['message'] = message

# --- end Phase 4 state ----------------------------------------------------

# Tutorial configuration
TUTORIAL_STEPS = [
    {'letter': 'A', 'prompt': 'Show letter A'},
    {'letter': 'B', 'prompt': 'Show letter B'},
    {'letter': 'C', 'prompt': 'Show letter C'},
    {'letter': 'D', 'prompt': 'Show letter D'},
    {'letter': 'E', 'prompt': 'Show letter E'}
]

# Feedback rules for ISL letters
FEEDBACK_RULES = {
    'A': {
        'correct': 'Correct! You signed "A" perfectly.',
        'incorrect': 'For "A", close your fingers into a fist with your thumb on the side.'
    },
    'B': {
        'correct': 'Great job! Your "B" is spot-on.',
        'incorrect': 'For "B", keep fingers together, palm out, and thumb extended.'
    },
    'C': {
        'correct': 'Nice! Your "C" is clear.',
        'incorrect': 'For "C", curve your fingers to form a "C" shape, palm facing out.'
    },
    'D': {
        'correct': 'Well done! Your "D" is correct.',
        'incorrect': 'For "D", extend index finger up, thumb holding other fingers.'
    },
    'E': {
        'correct': 'Excellent! You nailed "E".',
        'incorrect': 'For "E", tuck fingers down with thumb over them, palm out.'
    }
}

# Track tutorial progress
tutorial_progress = {'completed_steps': 0, 'correct_attempts': 0, 'total_attempts': 0}

# ISL configuration
from isl_assets import (
    ISL_LABELS as _ISL_LABELS,
    OUTPUT_DIR as ISL_OUTPUT_DIR,
    TARGET_SIZE,
    MAX_CHARS_PER_LINE,
    assets_available,
    load_letter_image,
    build_composite,
    missing_assets,
    cleanup_old_outputs,
)
import bilingual

# ISL_LABELS below keeps the original in-module definition; isl_assets agrees.
assert _ISL_LABELS == ISL_LABELS

ASSETS_READY = assets_available()
if not ASSETS_READY:
    logger.warning(
        "ISL alphabet assets missing: %s. "
        "Run 'python generate_alphabet_assets.py' to generate them from the bundled training data.",
        ', '.join(missing_assets())
    )

def clean_input(text):
    """Keep A-Z, 0-9 and spaces; everything else is rejected/removed."""
    allowed_chars = set(string.ascii_letters + string.digits + " ")
    cleaned = ''.join(ch for ch in text if ch in allowed_chars)
    return cleaned.upper()

def text_to_isl_images(text):
    """Render each A-Z character to a tile using the generated local assets."""
    text = clean_input(text)
    images = []
    for char in text:
        if char == " ":
            images.append(np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255)
            continue
        if char not in ISL_LABELS:
            logger.warning(f"Skipping unsupported character: {char}")
            continue
        img = load_letter_image(char, TARGET_SIZE)
        if img is None:
            logger.warning(f"No reference asset for letter: {char}")
            continue
        images.append(img)
    return images

def generate_isl_image(text):
    isl_images = text_to_isl_images(text)
    if not isl_images:
        logger.warning(f"No ISL images generated for text: {text}")
        return None
    combined_image = build_composite(isl_images)
    ISL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = f"isl_{uuid.uuid4().hex}.png"
    output_path = ISL_OUTPUT_DIR / output_file
    cv2.imwrite(str(output_path), combined_image)
    # Opportunistic cleanup of stale generated images.
    try:
        removed = cleanup_old_outputs(keep_seconds=3600)
        if removed:
            logger.info(f"Cleaned {removed} old generated image(s)")
    except Exception as e:
        logger.warning(f"Output cleanup failed: {e}")
    return output_file

def detect_hand(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    skin_ratio = skin_pixels / total_pixels
    if skin_ratio > 0.15:
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                return True
    return False

def generate_frames():
    global cap, current_prediction, prediction_history, word, last_prediction, last_time, predicted_sentence
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            with lock:
                _update_prediction_state('error', message='Webcam unavailable — check that a camera is connected and not in use by another app.')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # MJPEG generator started -> camera is live
    with lock:
        _update_prediction_state('detecting', message='Camera live — show a sign in the green box.')

    while True:
        with lock:
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                _update_prediction_state('error', message='Camera frame could not be read — try stopping and starting again.')
                break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        roi_width, roi_height = 300, 300
        # Center the ROI box
        roi_x = (width - roi_width) // 2
        roi_y = (height - roi_height) // 2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), 
                     (0, 255, 0), 2)
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        action = ''
        if detect_hand(roi):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (50, 50))
            gray = gray.reshape(1, 50, 50, 1)
            gray = gray.astype('float32') / 255.0
            prediction = model.predict(gray, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Top-3 straight from the raw softmax output (never re-parsed
            # from display strings).
            top_idx = np.argsort(prediction[0])[::-1][:3]
            top_predictions = [
                {'letter': ISL_LABELS[int(i)], 'confidence': round(float(prediction[0][int(i)]) * 100.0, 1)}
                for i in top_idx
            ]

            with lock:
                if confidence > ACCEPT_CONFIDENCE:
                    current_pred = ISL_LABELS[predicted_class]
                    current_prediction = f"Prediction: {current_pred} ({confidence*100:.1f}%)"
                    prediction_history.append(f"{current_pred} ({confidence*100:.1f}%)")
                    hold_duration = 2.0  # seconds
                    if (len(word) == 0 or current_pred != word[-1]) or (time.time() - last_time > hold_duration):
                        word.append(current_pred)
                        last_time = time.time()
                    last_prediction = current_pred
                    if ''.join(word) in COMMON_WORDS:
                        predicted_sentence = ''.join(word)
                        word.clear()
                        logger.info(f"Sentence predicted: {predicted_sentence}")
                    _update_prediction_state(
                        'detecting', letter=current_pred, confidence=confidence,
                        top=top_predictions, message='Great sign detected')
                    if current_pred in NAVIGATION_GESTURES:
                        action = NAVIGATION_GESTURES[current_pred]
                elif confidence > LOW_CONFIDENCE:
                    current_prediction = "Prediction: Low Confidence"
                    last_prediction = None
                    _update_prediction_state(
                        'low_confidence', confidence=confidence,
                        top=top_predictions, message='Hold steady — sign not clear enough yet')
                else:
                    current_prediction = "Prediction: Low Confidence"
                    last_prediction = None
                    _update_prediction_state(
                        'low_confidence', confidence=confidence,
                        top=top_predictions, message='Hold steady — the sign is too unclear')
        else:
            with lock:
                current_prediction = "Prediction: No Hand Detected"
                last_prediction = None
                _update_prediction_state(
                    'no_hand', message='No hand detected — show your hand in the green box')
        cv2.putText(frame, "Show ISL sign in the green box", (50, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/isl_detection')
def isl_detection():
    return render_template('asl_detection.html')

@app.route('/speech_to_isl')
def speech_to_isl():
    return render_template('speech_to_isl.html')

@app.route('/text_to_isl')
def text_to_isl():
    return render_template('text_to_isl.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with lock:
        current_pred = current_prediction.split(': ')[1][0] if 'Prediction: ' in current_prediction and current_prediction.split(': ')[1][0] in ISL_LABELS else ''
        # Extract confidence from strings like "Prediction: A (95.3%)";
        # fall back to 0 when the format has no confidence part.
        try:
            confidence = float(current_prediction.split('(')[1].split('%')[0])
        except (IndexError, ValueError):
            confidence = 0.0
        return jsonify({
            'current': current_prediction,
            'history': list(prediction_history),
            'confidence': confidence,
            'word': ''.join(word),
            'action': NAVIGATION_GESTURES.get(current_pred, '')
        })

@app.route('/clear_word', methods=['POST'])
def clear_word():
    global word, predicted_sentence
    with lock:
        word = []
        predicted_sentence = ''
        logger.info("Word and sentence cleared")
    return jsonify({'status': 'success'})

@app.route('/stop_camera')
def stop_camera():
    global cap
    with lock:
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None
            logger.info("Camera stopped")
        _update_prediction_state(
            'camera_off', message='Camera off — press Start Camera to begin.')
    return "Camera stopped"

@app.route('/api/prediction')
def api_prediction():
    """
    Structured, machine-readable recognition state for the dashboard UI.
    Values come directly from CNN inference in generate_frames() - the
    human-readable prediction string is never parsed.

    status: ready | detecting | no_hand | low_confidence | camera_off | error
    """
    with lock:
        return jsonify({
            'status': prediction_state['status'],
            'letter': prediction_state['letter'],
            'confidence': prediction_state['confidence'],
            'top_predictions': prediction_state['top_predictions'],
            'word_buffer': prediction_state['word_buffer'],
            'sentence': prediction_state['sentence'],
            'timestamp': prediction_state['timestamp'],
            'message': prediction_state['message'],
        })

@app.route('/process_speech', methods=['POST'])
def process_speech():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    cleaned_text = clean_input(text)
    if not any(ch.isalnum() for ch in cleaned_text):
        return jsonify({'error': 'No valid letters found'}), 400
    output_file = generate_isl_image(cleaned_text)
    if output_file:
        return jsonify({'text': cleaned_text, 'image': f'/static/isl_output/{output_file}'})
    return jsonify({'error': 'No valid ISL images generated'}), 400

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    cleaned_text = clean_input(text)
    if not any(ch.isalnum() for ch in cleaned_text):
        return jsonify({'error': 'No valid letters found'}), 400
    output_file = generate_isl_image(cleaned_text)
    if output_file:
        return jsonify({'text': cleaned_text, 'image': f'/static/isl_output/{output_file}'})
    return jsonify({'error': 'No valid ISL images generated'}), 400

@app.route('/process_bilingual_text', methods=['POST'])
def process_bilingual_text():
    """
    Bilingual Text-to-ISL: English, Kannada or mixed text ->
    Roman transliteration -> ISL alphabet finger-spelling sequence.

    This is Kannada *text* to ISL finger-spelling - it is NOT Kannada Sign
    Language gesture recognition.
    Request JSON:  {"text": "<any unicode text>", "mode": "auto"|"en"|"kn"}
    Response JSON: {original_text, language, romanized_text, isl_sequence,
                    image, length}
    Errors:        400 with {"error": "..."} on empty or unsupported input.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    mode = (data.get('mode') or 'auto').lower()

    if not text:
        return jsonify({'error': 'No text provided. Type English or Kannada text first.'}), 400
    if len(text) > 500:
        return jsonify({'error': 'Text too long (maximum 500 characters).'}), 400
    if mode not in ('auto', 'en', 'kn'):
        return jsonify({'error': f'Unknown mode: {mode}. Use auto, en or kn.'}), 400

    # Force an explicit mode by preprocessing the text for that script.
    if mode == 'en':
        language = 'en'
        # Strip non-Latin characters; keep A-Z a-z digits spaces.
        latin_only = ''.join(ch for ch in text if ch in string.ascii_letters + string.digits + ' ')
        roman_text = latin_only.strip().upper()
    elif mode == 'kn':
        language = 'kn'
        roman_text, _ = bilingual.prepare_for_isl(text)
        if not bilingual._KANNADA_RE.search(text):
            return jsonify({'error': 'Kannada mode selected, but the text contains no Kannada script. Switch to Auto Detect or English.'}), 400
    else:
        roman_text, language = bilingual.prepare_for_isl(text)
    if not roman_text or not any(ch.isalnum() for ch in roman_text):
        return jsonify({'error': 'No finger-spellable characters found (A-Z). Unsupported characters were removed.'}), 400
    sequence = bilingual.isl_letter_sequence(roman_text)
    output_file = generate_isl_image(roman_text)
    if not output_file:
        return jsonify({'error': 'Could not generate the ISL image sequence.'}), 500
    return jsonify({
        'original_text': text,
        'language': language,
        'romanized_text': roman_text,
        'isl_sequence': sequence,
        'image': f'/static/isl_output/{output_file}',
        'length': len([s for s in sequence if s != 'space'])
    })

@app.route('/kannada_news_examples')
def kannada_news_examples():
    """
    Return a small sample of real Kannada headlines from the optional local
    kannada_dataset CSV files (text examples only - never model training data).
    Returns {"available": false, "message": ...} when the corpus is absent.
    """
    base_dir = None
    for candidate in _KANNADA_NEWS_DIRS:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate) and any(
            os.path.isfile(os.path.join(candidate, f)) for f in _KANNADA_NEWS_FILES
        ):
            base_dir = candidate
            break
    if base_dir is None:
        return jsonify({
            'available': False,
            'message': 'The optional Kannada news corpus is not installed locally. '
                       'You can still type any Kannada text above.',
            'examples': []
        })
    import csv
    import random
    examples = []
    for filename in _KANNADA_NEWS_FILES:
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = [row[0] for row in reader if row and row[0].strip()]
            # Skip header if present (first cell often 'headline')
            if rows and rows[0].lower().strip() in ('headline', 'text'):
                rows = rows[1:]
            sample = random.sample(rows, min(3, len(rows)))
            examples.extend({'headline': h.strip(), 'source': filename} for h in sample)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            logger.warning(f'Could not read Kannada news file {filename}: {e}')
    if not examples:
        return jsonify({'available': False,
                        'message': 'Kannada news files were found but could not be read.',
                        'examples': []})
    random.shuffle(examples)
    return jsonify({'available': True, 'message': '', 'examples': examples[:3]})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json.get('feedback')
    if not feedback:
        return jsonify({'error': 'No feedback provided'}), 400
    sentiment_label = 'Unknown'
    sentiment_score = 0.0
    if sentiment_analyzer:
        try:
            sentiment_result = sentiment_analyzer(feedback)[0]
            sentiment_label = sentiment_result['label'].capitalize()
            sentiment_score = sentiment_result['score']
            if sentiment_label == 'Positive' and sentiment_score < 0.7:
                sentiment_label = 'Neutral'
            elif sentiment_label == 'Negative' and sentiment_score < 0.7:
                sentiment_label = 'Neutral'
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            sentiment_label = 'Unknown'
            sentiment_score = 0.0
    feedback_entry = {
        'id': str(uuid.uuid4()),
        'feedback': feedback,
        'sentiment': sentiment_label,
        'sentiment_score': sentiment_score
    }
    with open('feedback.json', 'a') as f:
        json.dump(feedback_entry, f)
        f.write('\n')
    return jsonify({'status': 'success'})

@app.route('/get_feedback')
def get_feedback():
    feedback_list = []
    if os.path.exists('feedback.json'):
        with open('feedback.json', 'r') as f:
            for line in f:
                feedback_list.append(json.loads(line))
    return jsonify(feedback_list)

@app.route('/get_tutorial_step/<int:step>')
def get_tutorial_step(step):
    global tutorial_progress
    if step >= len(TUTORIAL_STEPS):
        progress = {
            'completed': tutorial_progress['completed_steps'],
            'total': len(TUTORIAL_STEPS),
            'accuracy': (tutorial_progress['correct_attempts'] / tutorial_progress['total_attempts'] * 100) if tutorial_progress['total_attempts'] > 0 else 0
        }
        logger.info(f"Tutorial complete: {progress}")
        return jsonify({'done': True, 'progress': progress})
    # Check current prediction
    with lock:
        current_pred = current_prediction.split(': ')[1][0] if 'Prediction: ' in current_prediction and current_prediction.split(': ')[1][0] in ISL_LABELS else ''
        try:
            confidence = float(current_prediction.split('(')[1].split('%')[0])
        except (IndexError, ValueError):
            confidence = 0.0
    tutorial_progress['total_attempts'] += 1
    feedback = ''
    if current_pred == TUTORIAL_STEPS[step]['letter'] and confidence > 65:  # Lowered threshold
        feedback = FEEDBACK_RULES[TUTORIAL_STEPS[step]['letter']]['correct']
        tutorial_progress['completed_steps'] = max(tutorial_progress['completed_steps'], step + 1)
        tutorial_progress['correct_attempts'] += 1
    elif current_pred and confidence > 50:
        feedback = FEEDBACK_RULES[TUTORIAL_STEPS[step]['letter']]['incorrect']
    elif confidence > 0:
        feedback = 'Low confidence. Try positioning your hand clearly in the green box.'
    else:
        feedback = 'No hand detected. Show your hand in the green box.'
    progress = {
        'completed': tutorial_progress['completed_steps'],
        'total': len(TUTORIAL_STEPS),
        'accuracy': (tutorial_progress['correct_attempts'] / tutorial_progress['total_attempts'] * 100) if tutorial_progress['total_attempts'] > 0 else 0
    }
    logger.info(f"Step {step}: Prediction={current_pred}, Confidence={confidence}, Feedback={feedback}, Progress={progress}")
    return jsonify({
        'letter': TUTORIAL_STEPS[step]['letter'],
        'prompt': TUTORIAL_STEPS[step]['prompt'],
        'feedback': feedback,
        'progress': progress
    })

@app.route('/reset_tutorial', methods=['POST'])
def reset_tutorial():
    global tutorial_progress
    tutorial_progress = {'completed_steps': 0, 'correct_attempts': 0, 'total_attempts': 0}
    logger.info("Tutorial progress reset")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8080)