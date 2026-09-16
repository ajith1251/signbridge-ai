# SignBridge AI

A collection of AI-powered sign language recognition and translation projects bridging communication gaps for the hearing-impaired community.

## Projects

### 1. ISL Recognition and Translation (`Indian-Sign-Language-ISL--recognition-and-translation/`)

An end-to-end system for recognizing and translating **Indian Sign Language (ISL)** gestures into text, and rendering text/speech as ISL finger-spelling images — entirely local, no external downloads required.

**Features:**
- Real-time ISL alphabet recognition (A–Z) via webcam
- Text-to-ISL finger-spelling image generation
- Speech-to-ISL via browser microphone
- Guided signing tutorial (A–E) with per-letter feedback
- Bilingual foundation: Kannada text detection and Roman transliteration
- User feedback system with optional sentiment analysis

**Tech stack:** Python 3.13, TensorFlow/Keras, OpenCV, Flask, Pillow

**Quick start:**
```bash
cd Indian-Sign-Language-ISL--recognition-and-translation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_alphabet_assets.py
python app.py
# Open http://localhost:8080
```

See [full ISL project README](Indian-Sign-Language-ISL--recognition-and-translation/README.md) for details on routes, CLI tools, model architecture, and limitations.

---

### 2. ASLearn (`i-w/ASLearn/`)

An interactive web application for learning the **American Sign Language (ASL)** alphabet with real-time gesture recognition.

**Features:**
- Real-time ASL alphabet recognition via webcam
- Easy Mode: displays hand sign reference images for guidance
- Hard Mode: sign without reference images for practice
- Web-based interface with multiple learning levels

**Tech stack:** Python, TensorFlow/Keras, OpenCV, MediaPipe, Flask

**Quick start:**
```bash
cd i-w/ASLearn
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Download model from: https://drive.google.com/file/d/1uf1nlVk13368ZBGeYaFcdwvu-KAN_1_I
python app.py
# Open http://127.0.0.1:5000
```

See [full ASLearn README](i-w/ASLearn/README.md) for methodology, training details, and model performance.

---

### 3. Kannada News Dataset (`kannada_dataset/`)

A Kannada news headline classification dataset (~6,300 headlines) for NLP research in under-represented South Indian languages.

**Categories:** Sports, Tech, Entertainment  
**Splits:** Train (5,167) / Validation (1,293)  
**License:** CC-BY-SA-4.0

See [full dataset README](kannada_dataset/README.md) for dataset card and citations.

---

## Repository Structure

```
signbridge-ai/
├── Indian-Sign-Language-ISL--recognition-and-translation/   # ISL recognition & translation
├── i-w/ASLearn/                                             # ASL alphabet learning app
├── kannada_dataset/                                         # Kannada news NLP dataset
└── README.md
```

## Getting Started

Each project is self-contained with its own `requirements.txt`. See individual READMEs for project-specific setup instructions. All projects run locally with Python and require a webcam for sign language recognition features.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## Acknowledgments

Built with open-source tools from the computer vision, deep learning, and sign language research communities, dedicated to improving accessibility for the hearing-impaired.