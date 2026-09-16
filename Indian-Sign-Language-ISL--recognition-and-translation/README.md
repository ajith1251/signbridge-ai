# Indian Sign Language (ISL) Recognition and Translation — SignBridge AI

![ISL Recognition](https://img.shields.io/badge/ISL-Recognition-blue)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)

An end-to-end system for recognizing and translating Indian Sign Language (ISL)
gestures into text, and rendering text/speech as ISL alphabet finger-spelling
images — all locally, with no external image or model downloads.

## Features

- **Real-time ISL recognition**: detect and classify 26 static ISL alphabet
  signs (A–Z) from your webcam.
- **Text-to-ISL**: convert typed English text into a finger-spelling image.
- **Speech-to-ISL**: convert speech (browser mic) into a finger-spelling image.
- **Tutorial mode**: guided A–E signing practice with per-letter feedback.
- **Feedback page**: user feedback with optional sentiment tagging.
- **Bilingual foundation** (Phase 1): Kannada text detection and Roman
  transliteration ready for future finger-spelling support.

## Model facts

- Architecture: 2D CNN (Keras), stored as `model.h5`.
- **Input shape: `50×50×1` grayscale**, normalized to `[0, 1]`.
- Output: 26 classes (`A`–`Z`), softmax.
- Training data bundled in this repo: `X_train.npy` (31,200 × 50×50×1) and
  `y_train.npy` (one-hot, 26 classes) — preserved and used by the asset
  generator below.
- The webcam pipeline centers a 300×300 ROI, converts to grayscale, resizes to
  50×50, and runs the CNN; predictions below 70% confidence are ignored.

## Windows setup (exact commands)

From the repository root (`D:\Sign language\Indian-Sign-Language-ISL--recognition-and-translation`):

```bat
:: 1. Create and activate a virtual environment (Python 3.13 recommended)
python -m venv venv
venv\Scripts\activate

:: 2. Install dependencies
pip install -r requirements.txt

:: 3. Generate the A-Z reference assets from the bundled training data
::    (creates static\assets\isl_alphabet\A.png ... Z.png)
python generate_alphabet_assets.py

:: 4. Run the web application
python app.py
```

Then open <http://localhost:8080> in your browser:

| Route | Purpose |
|---|---|
| `/` | Home |
| `/isl_detection` | Real-time webcam ISL detection |
| `/text_to_isl` | Text → finger-spelling image |
| `/speech_to_isl` | Speech → finger-spelling image (browser mic) |
| `/tutorial` | Guided signing tutorial (A–E) |
| `/feedback` | Leave/view feedback |

The sentiment-tagging feature on `/feedback` uses `transformers` + `torch`
(included in `requirements.txt`). If they are not installed the app still runs
and stores feedback untagged.

### Command-line utilities

```bat
:: Text to ISL (interactive or direct)
python text_to_isl.py "HELLO WORLD"

:: Speech to ISL (requires a microphone)
python speech_to_isl.py

:: Verify all 26 alphabet assets exist
python generate_alphabet_assets.py --check
```

### Generating the alphabet assets

`generate_alphabet_assets.py` derives one stable reference image per A–Z label
from the bundled training data (per-pixel class mean, contrast-normalized) and
writes them to `static/assets/isl_alphabet/<LETTER>.png`. The web app and both
CLI tools read those assets — no `data/` folder or downloads are needed.
Re-run it any time the training data changes; output is deterministic.

## Bilingual foundation (Kannada, Phase 1)

`bilingual.py` detects Kannada script and transliterates it to Roman letters:

```text
ನಮಸ್ಕಾರ  -> NAMASKAARA   (lang: kn)
ಕನ್ನಡ    -> KANNADA      (lang: kn)
ನಮಸ್ಕಾರ Hello -> NAMASKAARA HELLO (lang: mixed)
```

The planned feature is **Kannada text → Roman transliteration → ISL alphabet
finger-spelling**. This module is not yet wired into the web UI. Note this is
*not* Kannada Sign Language recognition — the CNN recognizes ISL alphabet signs
only. The `kannada_dataset/` folder is unrelated NLP dataset code kept separate
from the CNN training data.

## Project structure

```
├── app.py                        # Main Flask application (port 8080)
├── isl_assets.py                 # Shared A-Z asset loading + composite builder + cleanup
├── generate_alphabet_assets.py   # Builds A-Z reference images from training data
├── bilingual.py                  # Kannada detection + Roman transliteration (foundation)
├── train_model.py                # CNN training (5-fold CV) - do not run casually
├── model_analysis.py             # Evaluation, confusion matrix
├── predict.py / translator.py    # Desktop recognizer (pyttsx3 TTS)
├── realtime_detection.py         # Desktop real-time detection
├── text_to_isl.py / speech_to_isl.py  # CLI converters (use generated assets)
├── text_splitter.py              # NLTK text preprocessing utility
├── variables.py                  # IMAGE_SIZE=50, MODEL_PATH, LABELS, THRESHOLD
├── model.h5                      # Trained CNN (Git LFS)
├── X_train.npy / y_train.npy     # Bundled training data (Git LFS)
├── static/
│   ├── assets/isl_alphabet/      # Generated A-Z reference images (application-owned)
│   └── isl_output/               # Generated finger-spelling results (auto-cleaned)
└── templates/                    # index, asl_detection, text_to_isl,
                                  # speech_to_isl, tutorial, feedback
```

## Current limitations

- **Static alphabet only.** The model recognizes 26 single-hand ISL alphabet
  poses. Words/sentences are formed by naive concatenation; there is no
  dynamic-gesture or ISL vocabulary (grammar, word order) support.
- **Finger-spelling composites are synthetic.** Text-to-ISL tiles generated
  class-mean reference images; they represent each letter's handshape but are
  not photographic depictions of a real hand.
- **Skin-color-based hand gate.** `detect_hand()` uses HSV skin thresholds;
  accuracy drops in poor lighting or for skin tones outside the tuned range.
- **Word matching is tiny.** Only `HELLO`, `THANK`, `YOU`, `GOOD`, `BYE` are
  recognized as whole words on the detection page.
- **Tutorial covers A–E only**, evaluated on letter + confidence (>65%).
- **Kannada support is transliteration-only** (foundation), not wired into the
  UI, and not KSL recognition.
- **CPU inference.** TensorFlow on native Windows runs CPU-only; the webcam
  loop runs a prediction per frame and may be slow on modest hardware.
- **sentiment feedback** requires the optional `transformers`/`torch` install;
  otherwise feedback is saved without a sentiment tag.
- The repo still contains legacy LFS-tracked sample outputs under
  `static/isl_output/`; new runtime-generated images are gitignored and
  auto-deleted after ~1 hour.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to the open-source computer-vision and sign-language research
  community, and to everyone working on accessibility tooling for the
  hearing-impaired community in India.
