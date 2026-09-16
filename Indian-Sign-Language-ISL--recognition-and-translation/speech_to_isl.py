"""
Speech -> ISL finger-spelling (CLI).

Captures speech via the microphone, recognizes it with Google Speech
Recognition, then renders the recognized text with the generated local
alphabet assets (static/assets/isl_alphabet/, built by
generate_alphabet_assets.py). All paths are resolved relative to the
repository root via isl_assets - no hardcoded drive paths.

Usage:
    python speech_to_isl.py
"""

import string
import sys

import cv2
import numpy as np
import speech_recognition as sr

from isl_assets import (
    TARGET_SIZE,
    assets_available,
    build_composite,
    load_letter_image,
    missing_assets,
)

ISL_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def clean_input(text):
    allowed_chars = set(string.ascii_letters + " ")
    cleaned = "".join(ch for ch in text if ch in allowed_chars)
    return cleaned.upper()


def text_to_isl_images(text):
    text = clean_input(text)
    images = []

    for char in text:
        if char == " ":
            images.append(np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255)
            continue
        if char not in ISL_LABELS:
            print(f"Skipping unsupported character: {char}")
            continue

        img = load_letter_image(char, TARGET_SIZE)
        if img is None:
            print(f"No reference asset for letter: {char}")
            continue
        images.append(img)

    return images


def display_isl_word(text):
    isl_images = text_to_isl_images(text)
    if not isl_images:
        print("No valid ISL images to display.")
        return

    combined_image = build_composite(isl_images)
    if combined_image is None:
        print("No valid ISL images to display.")
        return

    cv2.imshow("ISL Translation", combined_image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""


def main():
    if not assets_available():
        print("Alphabet assets are missing:", ", ".join(missing_assets()))
        print("Generate them first with: python generate_alphabet_assets.py")
        sys.exit(1)

    raw_text = recognize_speech()
    if not raw_text:
        return

    cleaned_text = clean_input(raw_text)
    if not any(ch.isalpha() for ch in cleaned_text):
        print("No valid letters found or no images available.")
        return

    display_isl_word(cleaned_text)


if __name__ == "__main__":
    main()
